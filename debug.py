import os
from typing import Dict, Optional
import random
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader

from src.config import Config
from src.constants import DEPTH_IMG_SCALE, TABLE_HEIGHT, PC_MAX, PC_MIN, OBJ_INIT_TRANS
from src.utils import get_pc, get_workspace_mask
from src.vis import Vis
from src.robot.cfg import get_robot_cfg
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"


def random_sampling(points: np.ndarray, M: int) -> np.ndarray:
    N = points.shape[0]
    if M >= N:
        return points.copy()
    # np.random.choice 默认是不放回抽样，需要指定 replace=False
    indices = np.random.choice(N, size=M, replace=False)
    return points[indices]

def visualize_pc(points: np.ndarray):
    # points = points[get_workspace_mask(points)]
    points = random_sampling(points, 100000)
    print(points[:, 0].mean(), points[:, 1].mean(), points[:, 2].mean())
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        # 点大小、透明度可以自己调
        marker=dict(
            size=1,
            opacity=0.8,
            color=points[:, 2],  # 根据 z 值着色
            colorscale='Viridis',
        )
    )

    layout = go.Layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=600,
        title='Plotly 3D 点云示例'
    )

    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()

obj_pose = np.load("data/train/20250606_025830/object_pose.npy")
camera_pose = np.load("data/train/20250606_025830/camera_pose.npy")
depth_array = (
    np.array(
        cv2.imread("data/train/20250606_025830/depth.png", cv2.IMREAD_UNCHANGED)
    )
    / DEPTH_IMG_SCALE
)

full_pc_camera = get_pc(
    depth_array, get_robot_cfg("galbot").camera_cfg[1].intrinsics
)
full_pc_world = (
    np.einsum("ab,nb->na", camera_pose[:3, :3], full_pc_camera)
    + camera_pose[:3, 3]
)


full_coord = np.einsum(
    "ba,nb->na", obj_pose[:3, :3], full_pc_world - obj_pose[:3, 3]
)

pc_mask = get_workspace_mask(full_pc_world)
sel_pc_idx = np.random.randint(0, np.sum(pc_mask), 1024)
visualize_pc(full_pc_world[pc_mask])

pc_camera = full_pc_camera[pc_mask][sel_pc_idx]
coord = full_coord[pc_mask][sel_pc_idx]
rel_obj_pose = np.linalg.inv(camera_pose) @ obj_pose