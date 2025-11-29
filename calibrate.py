import os
import sys
import time
import cv2
import roslibpy
import numpy as np
import pyrealsense2 as rs

from typing import Any, Callable, Iterable, Sequence, TypedDict
from pyquaternion import Quaternion
from itertools import product
import pickle

# roslaunch rosbridge_server rosbridge_websocket.launch

class Position(TypedDict):
    x: float
    y: float
    z: float
    
class Orientation(TypedDict):
    w: float
    x: float
    y: float
    z: float

class Pose(TypedDict):
    position: Position
    orientation: Orientation

Vector3 = Position

type PositionGenerator = Iterable[Position]
type PoseGenerator = Iterable[Pose]

type MatLike = cv2.typing.MatLike
type Image = MatLike


def identity[T](x: T) -> T:
    return x

class AuboClient:
    client: roslibpy.Ros
    publisher: roslibpy.Topic
    
    def __init__(self, ip: str, port: int, topic: str, timeout: float = 10.0):
        self.client = roslibpy.Ros(host=ip, port=port)
        print('Connecting to Aubo at {}:{} ...'.format(ip, port))
        self.client.run()
        
        # wait for connection
        waited = 0.0
        while not self.client.is_connected and waited < timeout: # type: ignore
            time.sleep(0.1)
            waited += 0.1

        if not self.client.is_connected:  # type: ignore
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
        
    def set_pose(self, pose: Pose) -> None:
        # construct a PoseStamped message and publish it once
        t = time.time()
        secs = int(t)
        nsecs = int((t - secs) * 1e9)

        message: dict[str, Any] = {
            'header': {
                'stamp': {'secs': secs, 'nsecs': nsecs},
                'frame_id': 'world'
            },
            'pose': pose
        }
        self.publisher.publish(roslibpy.Message(message)) # type: ignore

class Camera:
    profile: rs.pipeline_profile
    pipeline: rs.pipeline
    preprocess: Callable[[Image], Image]
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, warmup: int = 30, preprocess: Callable[[Image], Image] = identity):
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
        self.preprocess = preprocess
        
    def capture(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("无法获取彩色帧")
        img = np.asanyarray(color.get_data())
        img = self.preprocess(img)
        return img
    
    def shutdown(self):
        self.pipeline.stop()

def look_at(origin: Position, target: Position, up: Vector3, body: Orientation) -> Orientation:
    forward = np.array([target['x'] - origin['x'],
                        target['y'] - origin['y'],
                        target['z'] - origin['z']])
    forward = forward / np.linalg.norm(forward)
    
    up_vec = np.array([up['x'], up['y'], up['z']])
    up_vec = up_vec / np.linalg.norm(up_vec)
    
    right = np.cross(forward, up_vec)
    right = right / np.linalg.norm(right)
    
    true_up = np.cross(right, forward)
    
    rot_matrix = np.array([
        [right[0], true_up[0], -forward[0]],
        [right[1], true_up[1], -forward[1]],
        [right[2], true_up[2], -forward[2]]
    ])
    
    quat = Quaternion(matrix=rot_matrix) * Quaternion(**body) # type: ignore
    
    return {
        'w': float(quat.w), 'x': float(quat.x), 'y': float(quat.y), 'z': float(quat.z) # type: ignore
    }


class HandEyeSystem:
    hand: AuboClient
    eye: Camera
    
    def __init__(self, hand: AuboClient, eye: Camera):
        self.hand = hand
        self.eye = eye
        
    def capture_at_pose(self, pose: Pose, wait: float) -> Image:
        self.hand.set_pose(pose)
        print('Published pose to Aubo.')
        time.sleep(wait)
        img = self.eye.capture()
        return img
    
class CharucoCalibrator:
    board: cv2.aruco.CharucoBoard
    detector: cv2.aruco.CharucoDetector
    
    def __init__(self, board: cv2.aruco.CharucoBoard):
        self.board = board
        self.detector = cv2.aruco.CharucoDetector(self.board)
        
    def detect(self, img: Image) -> tuple[MatLike, MatLike] | None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(img)
        if len(charuco_ids) < 4:
            return None
        del marker_corners, marker_ids
        return charuco_corners, charuco_ids
    
    def draw_detected(self, img: Image, corners: MatLike, ids: MatLike) -> Image:
        vis = img.copy()
        cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)
        return vis
    
    def calibrate_camera(self, corners_list: list[MatLike], ids_list: list[MatLike], image_size: tuple[int, int]) -> tuple[MatLike, MatLike, Sequence[MatLike], Sequence[MatLike]]:
        if len(corners_list) < 3:
            raise ValueError("At least 3 views are required for calibration.")
        ret, camera_matrix, dist_coefs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            corners_list, ids_list, self.board,
            image_size, np.array([]), np.array([]))
        if not ret:
            raise RuntimeError("Camera calibration failed.")
        return camera_matrix, dist_coefs, rvecs, tvecs
        

def main():
    aubo = AuboClient(ip='127.0.0.1', port=9090, topic='/realtime_pose_goal')
    
    width, height = 640, 480
    camera = Camera(width=width, height=height, preprocess=lambda img: cv2.flip(img, -1))
    system = HandEyeSystem(hand=aubo, eye=camera)
    
    calibrator = CharucoCalibrator(
        board=cv2.aruco.CharucoBoard(
            (8, 8), 0.05, 0.025, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        )
    )
    
    x_list = (0.4, 0.5, 0.6)
    y_list = (-0.12, 0.0, 0.12)
    z_list = (0.8, 0.87, 0.95)
    
    target = Position(x=0.45, y=0.0, z=0.5)
    up = Vector3(x=1.0, y=0.0, z=0.0)
    body = Orientation(w=0.0, x=-1/np.sqrt(2), y=1/np.sqrt(2), z=0.0)
    
    wait = 8.0  # seconds to wait after moving to a pose
    
    corner_results: list[tuple[Pose, tuple[MatLike, MatLike]]] = []
    image_output_dir = 'temp_calib_captures'
    
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)
        
    points = list(product(x_list, y_list, z_list))
    
    for x, y, z in points:
        position = Position(x=x, y=y, z=z)
        orientation = look_at(
            origin=position,
            target=target,
            up=up,
            body=body
        )
        pose = Pose(position=position, orientation=orientation)
        
        image = system.capture_at_pose(pose, wait=wait)
        print(image.shape)
        cv2.imwrite(os.path.join(image_output_dir, f'raw_x{x:.4f}_y{y:.4f}_z{z:.4f}.png'), image)
        
        detection = calibrator.detect(image)
        
        if detection is None:
            print(f"Charuco board not found at pose {pose}, skipping.")
            continue
        
        corner_results.append((pose, detection))
        corners, ids = detection
        vis = calibrator.draw_detected(image, corners, ids)
        
        filename = os.path.join(image_output_dir, f'capture_x{x:.4f}_y{y:.4f}_z{z:.4f}.png')
        cv2.imwrite(filename, vis)
    
    save_path = os.path.join('corner_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(corner_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    poses = [item[0] for item in corner_results]
    corners_list = [item[1][0] for item in corner_results]
    ids_list = [item[1][1] for item in corner_results]
    
    camera_matrix, dist_coefs, rvecs, tvecs = calibrator.calibrate_camera(
        corners_list, ids_list, (width, height)
    )
        
    R_target2cam_list = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]
    t_target2cam_list = [tvec.reshape(3, 1) for tvec in tvecs]
    
    R_gripper2base_list = [Quaternion(**pose['orientation']).rotation_matrix for pose in poses] # type: ignore
    t_gripper2base_list = [
        np.array([
            pose['position']['x'],
            pose['position']['y'],
            pose['position']['z']
        ]).reshape(3, 1)
        for pose in poses
    ]
    
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base_list, t_gripper2base_list, # type: ignore
        R_target2cam_list, t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    with open('transformations.pkl', 'wb') as f:
        pickle.dump({
            'R_gripper2base': R_gripper2base_list,
            't_gripper2base': t_gripper2base_list,
            'R_target2cam': R_target2cam_list,
            't_target2cam': t_target2cam_list,
            'R_cam2gripper': R_cam2gripper,
            't_cam2gripper': t_cam2gripper
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved transformations to transformations.pkl")
    
    print("R_cam2gripper:\n", R_cam2gripper)
    print("t_cam2gripper:\n", t_cam2gripper.ravel())

    np.savez('calib_hand_eye.npz',
                camera_matrix=camera_matrix,
                dist_coefs=dist_coefs,
                R_cam2gripper=R_cam2gripper,
                t_cam2gripper=t_cam2gripper)
    print("标定结果已保存到 calib_hand_eye.npz")

    aubo.shutdown()
    

if __name__ == '__main__':
    main()