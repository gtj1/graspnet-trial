import sys
import time
import cv2
import roslibpy
import numpy as np
import pyrealsense2 as rs

from typing import Any, cast
from pyquaternion import Quaternion
from itertools import product

class AuboClient:
    client: roslibpy.Ros
    publisher: roslibpy.Topic
    
    def __init__(self, ip: str, port: int, topic: str, timeout: float = 10.0):
        self.client = roslibpy.Ros(host=ip, port=port)
        print('Connecting to Aubo at {}:{} ...'.format(ip, port))
        self.client.run()
        
        # wait for connection
        waited = 0.0
        while not self.client.is_connected and waited < timeout:
            time.sleep(0.1)
            waited += 0.1

        if not self.client.is_connected:
            print('Failed to connect to Aubo within {}s'.format(timeout))
            sys.exit(1)
            
        print('Connected. Preparing to publish to {}.'.format(topic))
        self.publisher = roslibpy.Topic(self.client, topic, 'geometry_msgs/PoseStamped')
        
        
    def shutdown(self):
        try:
            self.client.terminate()
        except Exception:
            pass
        print('\nExiting.')
        sys.exit(0)
        
    def set_pose(self, position: tuple[float, float, float], orientation: tuple[float, float, float, float]):
        # construct a PoseStamped message and publish it once
        t = time.time()
        secs = int(t)
        nsecs = int((t - secs) * 1e9)

        message: dict[str, Any] = {
            'header': {
                'stamp': {'secs': secs, 'nsecs': nsecs},
                'frame_id': 'world'
            },
            'pose': {
                'position': {
                    'x': position[0],
                    'y': position[1],
                    'z': position[2]
                },
                'orientation': {
                    'w': orientation[0],
                    'x': orientation[1],
                    'y': orientation[2],
                    'z': orientation[3]
                }
            }
        }
        self.publisher.publish(roslibpy.Message(message))

class Camera:
    profile: rs.pipeline_profile
    pipeline: rs.pipeline
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, warmup: int = 30):
        pipeline = rs.pipeline()
        self.pipeline = pipeline
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 0, width, height, rs.format.bgr8, fps)
        self.profile = pipeline.start(cfg)
        try:
            # warm up
            for _ in range(warmup):
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                color = frames.get_color_frame()
                if not color:
                    raise RuntimeError("无法获取彩色帧")
        except:
            pipeline.stop()
            sys.exit(1)
        
    def capture(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("无法获取彩色帧")
        img = np.asanyarray(color.get_data())
        img = cv2.flip(img, -1)
        return img
    
    def shutdown(self):
        self.pipeline.stop()

x0 = 0.5
y0 = 0.0
z0 = 0.5

aubo = AuboClient(ip='127.0.0.1', port=9090, topic='/realtime_pose_goal')
camera = Camera()

def capture_at_pose(
    dx: float,
    dy: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.array([x0, y0, z0]) + np.array([dx, dy, dz])
    orientation = (Quaternion(1, -dy / dz, dx / dz, 0) ** 0.7 * Quaternion(0, 0, 1, 0)).normalised.elements # type: ignore
    aubo.set_pose(position.tolist(), orientation.tolist())
    print('Published pose to Aubo.')
    time.sleep(3)

    img = camera.capture()
    
    cv2.imwrite(f'capture_dx{dx:+.2f}_dy{dy:+.2f}_dz{dz:+.2f}.png', img)
    
    return (
        Quaternion(orientation[0], orientation[1], orientation[2], orientation[3]).rotation_matrix,
        position,
        img
    )

        
def main():
    # mat, _, _ = capture_at_pose(-0.1, 0, 0.4)
    # print(mat)
    # return
    
    dx_list = (-0.1, 0.0, 0.1)
    dy_list = (-0.2, -0.1, 0.0, 0.1, 0.2)
    dz_list = (0.3, 0.35, 0.4, 0.45)
    
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for dx, dy, dz in product(dx_list, dy_list, dz_list):
        results.append(capture_at_pose(dx, dy, dz))

    # 检测棋盘格并进行相机内参与手眼标定
    pattern_size = (7, 7)  # 内角点数量 (8x8 棋盘 -> 7x7 内角点)
    square_size = 0.036  # 单位：米（与机器人位姿单位保持一致）
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []     # 世界坐标中的棋盘角点
    imgpoints = []     # 图像坐标中的角点
    R_gripper2base_list = []  # 机器人末端相对于基座的旋转（从 results）
    t_gripper2base_list = []  # 机器人末端相对于基座的平移（从 results）

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for idx, (R_world_tool, pos_world, img) in enumerate(results):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            print(f"Chessboard not found in image {idx}, skipping.")
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        
        # 在图像上绘制检测到的棋盘角点并保存
        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern_size, corners, True)
        save_name = f'capture_chess_idx{idx:02d}.png'
        cv2.imwrite(save_name, vis)
        print(f"Saved chessboard image: {save_name}")

        # results 中的 R_world_tool 是工具（末端）在世界系下的旋转矩阵，pos_world 是末端在世界系下的位置
        R_gripper2base_list.append(np.asarray(R_world_tool, dtype=np.float64))
        t_gripper2base_list.append(np.asarray(pos_world, dtype=np.float64).reshape(3, 1))

    if len(objpoints) < 3:
        print("有效视图不足，至少需要 3 张成功检测棋盘格的图片进行标定。")
        return
    
    # 相机内参标定
    image_size = (gray.shape[1], gray.shape[0])
    ret, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)
    print("相机内参矩阵:\n", camera_matrix)
    print("畸变系数:\n", dist_coefs.ravel())

    # 将每个视角的棋盘在相机坐标系下的旋转和平移（target2cam）
    R_target2cam = []
    t_target2cam = []
    for rvec, tvec in zip(rvecs, tvecs):
        Rmat, _ = cv2.Rodrigues(rvec)
        R_target2cam.append(Rmat)
        t_target2cam.append(tvec.reshape(3, 1))

    # 手眼标定：求解相机到末端（tool）之间的变换
    # 输入：R_gripper2base_list, t_gripper2base_list, R_target2cam, t_target2cam
    # 输出：R_cam2gripper, t_cam2gripper （将相机坐标系变换到末端坐标系）
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base_list, t_gripper2base_list,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    print("R_cam2gripper:\n", R_cam2gripper)
    print("t_cam2gripper:\n", t_cam2gripper.ravel())

    # 保存标定结果
    np.savez('calib_hand_eye.npz',
                camera_matrix=camera_matrix,
                dist_coefs=dist_coefs,
                R_cam2gripper=R_cam2gripper,
                t_cam2gripper=t_cam2gripper)
    print("标定结果已保存到 calib_hand_eye.npz")

    aubo.shutdown()

if __name__ == '__main__':
    main()