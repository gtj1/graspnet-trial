import pickle
import numpy as np

from typing import Any, Literal
from pyquaternion import Quaternion
from calibrate import Pose

type TransformationMatrix = np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float64]]

def pose_to_matrix(pose: Pose) -> TransformationMatrix:
    """Convert a Pose object to a 4x4 transformation matrix."""

    R = Quaternion(**pose['orientation']).rotation_matrix  # type: ignore
    t = np.array([
        pose['position']['x'],
        pose['position']['y'],
        pose['position']['z']
    ]).reshape(3, 1)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t
    
    return T # type: ignore


transformations_path = 'transformations.pkl'

with open(transformations_path, 'rb') as file:
    transformations_result: dict[str, Any] = pickle.load(file)

R_gripper2base_list = transformations_result['R_gripper2base']
t_gripper2base_list = transformations_result['t_gripper2base']

R_target2cam_list = transformations_result['R_target2cam']
t_target2cam_list = transformations_result['t_target2cam']

R_cam2gripper = transformations_result['R_cam2gripper']
t_cam2gripper = transformations_result['t_cam2gripper']

transformation_gripper2base_list: list[TransformationMatrix] = [
    np.block([
        [R_gripper2base, t_gripper2base.reshape(3, 1)],
        [np.zeros((1, 3)), 1]
    ]) for R_gripper2base, t_gripper2base in zip(R_gripper2base_list, t_gripper2base_list)
]

transformation_target2cam_list: list[TransformationMatrix] = [
    np.block([
        [R_target2cam, t_target2cam.reshape(3, 1)],
        [np.zeros((1, 3)), 1]
    ]) for R_target2cam, t_target2cam in zip(R_target2cam_list, t_target2cam_list)
]

transformation_cam2gripper: TransformationMatrix = np.block([
    [R_cam2gripper, t_cam2gripper.reshape(3, 1)],
    [np.zeros((1, 3)), 1]
])

transformation_cam2target_list: list[TransformationMatrix] = [
    np.linalg.inv(T_target2cam) for T_target2cam in transformation_target2cam_list
]

transformation_cam2base_list: list[TransformationMatrix] = [
    T_gripper2base @ transformation_cam2gripper for T_gripper2base in transformation_gripper2base_list
]

trajectory_by_base = [
    T_cam2base[:3, 3] for T_cam2base in transformation_cam2base_list
]

trajectory_by_target = [
    T_cam2target[:3, 3] for T_cam2target in transformation_cam2target_list
]

trajectory_by_base = np.array(trajectory_by_base)
trajectory_by_target = np.array(trajectory_by_target)

trajectory_by_base -= trajectory_by_base.mean(axis=0)
trajectory_by_target -= trajectory_by_target.mean(axis=0)

U, S, Vt = np.linalg.svd(trajectory_by_base.T @ trajectory_by_target)

trajectory_by_base_aligned = trajectory_by_base @ U
trajectory_by_target_aligned = trajectory_by_target @ Vt.T

error = trajectory_by_base_aligned - trajectory_by_target_aligned
rmse = np.sqrt(np.mean(np.sum(error**2, axis=1)))

print(f'RMSE between trajectories: {rmse:.6f} units')