import os
import random
from typing import Optional
import numpy as np
import torch

from transforms3d.quaternions import quat2mat

from .type import Grasp
from .constants import PC_MAX, PC_MIN

def to_pose(
    trans: Optional[np.ndarray] = None, rot: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert translation and rotation into a 4x4 pose matrix.

    Parameters
    ----------
    trans: Optional[np.ndarray]
        Translation vector, shape (3,).
    rot: Optional[np.ndarray]
        Rotation matrix, shape (3, 3).

    Returns
    -------
    np.ndarray
        4x4 pose matrix.
    """
    ret = np.eye(4)
    if trans is not None:
        ret[:3, 3] = trans
    if rot is not None:
        ret[:3, :3] = rot
    return ret

def transform_grasp_pose(
    grasp: Grasp,
    est_trans: np.ndarray,
    est_rot: np.ndarray,
    cam_trans: np.ndarray,
    cam_rot: np.ndarray,
) -> Grasp:
    """
    Transform grasp from the object frame into the robot frame

    Parameters
    ----------
    grasp: Grasp
        The grasp to be transformed.
    est_trans: np.ndarray
        Estimated translation vector in the camera frame.
    est_rot: np.ndarray
        Estimated rotation matrix in the camera frame.
    cam_trans: np.ndarray
        Camera's translation vector in the robot frame.
    cam_rot: np.ndarray
        Camera's rotation matrix in the robot frame.

    Returns
    -------
    Grasp
        The transformed grasp in the robot frame.
    """
    # raise NotImplementedError
    return Grasp(
        trans=cam_rot @ (est_rot @ grasp.trans + est_trans) + cam_trans,
        rot=cam_rot @ est_rot @ grasp.rot,
        width=grasp.width,
    )

def rand_rot_mat() -> np.ndarray:
    """
    Generate a random rotation matrix with shape (3, 3) uniformly.
    """
    while True:
        quat = np.random.randn(4)
        if np.linalg.norm(quat) > 1e-6:
            break
    quat /= np.linalg.norm(quat)
    return quat2mat(quat)


def theta_to_2d_rot(theta: float) -> np.ndarray:
    """
    Convert a 2D rotation angle into a rotation matrix.

    Parameters
    ----------
    theta : float
        The rotation angle in radians.

    Returns
    -------
    np.ndarray
        The resulting 2D rotation matrix (2, 2).
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rot_dist(r1: np.ndarray, r2: np.ndarray) -> float:
    """
    The relative rotation angle between two rotation matrices.

    Parameters
    ----------
    r1 : np.ndarray
        The first rotation matrix (3, 3).
    r2 : np.ndarray
        The second rotation matrix (3, 3).

    Returns
    -------
    float
        The relative rotation angle in radians.
    """
    return np.arccos(np.clip((np.trace(r1 @ r2.T) - 1) / 2, -1, 1))

def get_pc(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Convert depth image into point cloud using intrinsics

    All points with depth=0 are filtered out

    Parameters
    ----------
    depth: np.ndarray
        Depth image, shape (H, W)
    intrinsics: np.ndarray
        Intrinsics matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        Point cloud with shape (N, 3)
    """
    # Get image dimensions
    height, width = depth.shape
    # Create meshgrid for pixel coordinates
    v, u = np.meshgrid(range(height), range(width), indexing="ij")
    # Flatten the arrays
    u = u.flatten()
    v = v.flatten()
    depth_flat = depth.flatten()
    # Filter out invalid depth values
    valid = depth_flat > 0
    u = u[valid]
    v = v[valid]
    depth_flat = depth_flat[valid]
    # Create homogeneous pixel coordinates
    pixels = np.stack([u, v, np.ones_like(u)], axis=0)
    # Convert pixel coordinates to camera coordinates
    rays = np.linalg.inv(intrinsics) @ pixels
    # Scale rays by depth
    points = rays * depth_flat
    return points.T


def calculate_table_height(pc, z_min=0.65, z_max=0.75):
    # 筛选出 z 值在范围内的点
    filtered_points = pc[
        (pc[:, 0] > PC_MIN[0])
        & (pc[:, 0] < PC_MAX[0])
        & (pc[:, 1] > PC_MIN[1])
        & (pc[:, 1] < PC_MAX[1])
        & (pc[:, 2] >= z_min)
        & (pc[:, 2] <= z_max)
    ]
    if len(filtered_points) == 0:
        raise ValueError("筛选后没有点，检查输入或调整 z 值范围")
    
    # 计算 z 均值
    table_height = np.mean(filtered_points[:, 2])
    return table_height

def get_workspace_mask(pc: np.ndarray) -> np.ndarray:
    """Get the mask of the point cloud in the workspace."""
    table_height = calculate_table_height(pc)
    pc_mask = (
        (pc[:, 0] > PC_MIN[0])
        & (pc[:, 0] < PC_MAX[0])
        & (pc[:, 1] > PC_MIN[1])
        & (pc[:, 1] < PC_MAX[1])
        & (pc[:, 2] > table_height + 0.005)
        & (pc[:, 2] < PC_MAX[2])
    )
    return pc_mask

def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed: int
        Random seed between 0 and 2**32 - 1
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)