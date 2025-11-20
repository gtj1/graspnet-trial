import numpy as np
import pyrealsense2 as rs
import open3d as o3d

def main():
    # RealSense 管线和配置
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Open3D 可视化器和点云容器
    vis = o3d.visualization.Visualizer()
    vis.create_window("RealSense Colored PointCloud", width=960, height=540)
    pcd = o3d.geometry.PointCloud()
    added = False

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 用 rs.pointcloud 生成点（自动使用 depth->color 映射）
            pc = rs.pointcloud()
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)

            # 顶点 (x,y,z)
            vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

            # 纹理坐标，用于从 color_frame 采样颜色
            tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
            color_image = np.asanyarray(color_frame.get_data())  # HxWx3 BGR

            h, w, _ = color_image.shape
            # 将纹理坐标 (s,t) 映射到像素坐标 (u,v)
            u = np.clip((tex[:, 0] * w).astype(np.int32), 0, w - 1)
            v = np.clip((tex[:, 1] * h).astype(np.int32), 0, h - 1)

            # 从 BGR -> RGB 并归一化到 [0,1]
            colors = color_image[v, u, ::-1].astype(np.float32) / 255.0

            # 去除无效点（深度为0通常会生成零点）
            valid = np.logical_and(np.isfinite(vtx).all(axis=1), (vtx != 0).any(axis=1))
            vtx = vtx[valid]
            colors = colors[valid]

            # 更新 Open3D 点云并渲染
            pcd.points = o3d.utility.Vector3dVector(vtx)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            if not added:
                vis.add_geometry(pcd)
                render_opt = vis.get_render_option()
                render_opt.background_color = np.asarray([0, 0, 0])
                render_opt.point_size = 1.0
                added = True
            else:
                vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()

            # 当窗口被关闭时退出
            if not vis.poll_events():
                break

    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()
        pipeline.stop()

if __name__ == "__main__":
    main()