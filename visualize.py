import numpy as np
import plotly.graph_objects as go
import os
from src.utils import get_pc, get_workspace_mask
from src.constants import DEPTH_IMG_SCALE, TABLE_HEIGHT, PC_MAX, PC_MIN, OBJ_INIT_TRANS
import cv2
import plotly.io as pio
pio.renderers.default = "browser"

def plot_pointclouds_with_poses(
    points1: np.ndarray,
    points2: np.ndarray,
    gt_pose: np.ndarray,
    pred_pose: np.ndarray,
    axis_length: float = 1.0
):
    """
    在同一图中可视化两个三维点云和两个位姿矩阵代表的坐标系（Ground Truth & Predicted）。

    参数:
      points1: (N,3) numpy 数组，点云 1 坐标
      points2: (M,3) numpy 数组，点云 2 坐标
      gt_pose: (4,4) numpy 数组，Ground Truth 位姿
      pred_pose: (4,4) numpy 数组，Predicted 位姿
      axis_length: 坐标轴长度
    """
    def _frame_traces(pose: np.ndarray, colors: list, name_prefix: str):
        R = pose[:3, :3]
        t = pose[:3, 3]
        origin = t.reshape((1,3))
        axes = np.eye(3) * axis_length
        axes_transformed = (R @ axes.T).T + t
        traces = []
        labels = ['X', 'Y', 'Z']
        for i in range(3):
            traces.append(
                go.Scatter3d(
                    x=[origin[0,0], axes_transformed[i,0]],
                    y=[origin[0,1], axes_transformed[i,1]],
                    z=[origin[0,2], axes_transformed[i,2]],
                    mode='lines',
                    line=dict(width=5, color=colors[i], dash='solid' if name_prefix=='GT' else 'dash'),
                    name=f"{name_prefix}-{labels[i]}"
                )
            )
        return traces

    # 点云 1
    scatter_pc1 = go.Scatter3d(
        x=points1[:,0], y=points1[:,1], z=points1[:,2],
        mode='markers',
        marker=dict(size=2, color='gray'),
        name='Point Cloud 1'
    )

    # 点云 2
    scatter_pc2 = go.Scatter3d(
        x=points2[:,0], y=points2[:,1], z=points2[:,2],
        mode='markers',
        marker=dict(size=2, color='blue'),
        name='Point Cloud 2'
    )

    # GT & Pred 坐标系
    gt_colors = ['red', 'green', 'blue']
    pred_colors = ['magenta', 'lime', 'cyan']
    gt_traces = _frame_traces(gt_pose, gt_colors, 'GT')
    pred_traces = _frame_traces(pred_pose, pred_colors, 'Pred')

    # 组合并绘图
    fig = go.Figure(data=[scatter_pc1, scatter_pc2] + gt_traces + pred_traces)
    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        title='三维点云及 Ground Truth vs Predicted 位姿'
    )
    fig.show()

def random_sampling(points: np.ndarray, M: int) -> np.ndarray:
    N = points.shape[0]
    if M >= N:
        return points.copy()
    # np.random.choice 默认是不放回抽样，需要指定 replace=False
    indices = np.random.choice(N, size=M, replace=False)
    return points[indices]


if __name__ == "__main__":
    fdir = "data/val/20250607_111925/"

    obj_pose = np.load(os.path.join(fdir, "object_pose.npy"))
    
    camera_pose = np.load(os.path.join(fdir, "camera_pose.npy"))
    print(camera_pose.shape, obj_pose.shape)
    depth_array = (
        np.array(
            cv2.imread(os.path.join(fdir, "depth.png"), cv2.IMREAD_UNCHANGED)
        )
        / DEPTH_IMG_SCALE
    )

    full_pc_camera = get_pc(
        depth_array, np.array([[649.5874633789062, 0, 644.9450073242188], [0, 648.9038696289062, 355.34857177734375], [0, 0, 1]])
    )
    full_pc_world = (
        np.einsum("ab,nb->na", camera_pose[:3, :3], full_pc_camera)
        + camera_pose[:3, 3]
    )
    full_coord = np.einsum(
        "ba,nb->na", obj_pose[:3, :3], full_pc_world - obj_pose[:3, 3]
    )
    rel_obj_pose = np.linalg.inv(camera_pose) @ obj_pose

    pc_mask = get_workspace_mask(full_pc_world)
    sel_pc_idx = np.random.randint(0, np.sum(pc_mask), 1024)

    pc_camera = full_pc_camera[pc_mask][sel_pc_idx]
    pc_world = full_pc_world[pc_mask][sel_pc_idx]
    coord = full_coord[pc_mask][sel_pc_idx]

    plot_pointclouds_with_poses(random_sampling(full_pc_world, 50000), pc_world, obj_pose, camera_pose @ rel_obj_pose, axis_length=0.1)

    
    pc=pc_camera.astype(np.float32)
    coord=coord.astype(np.float32)
    trans=rel_obj_pose[:3, 3].astype(np.float32)
    rot=rel_obj_pose[:3, :3].astype(np.float32)
    camera_pose=camera_pose.astype(np.float32)
    obj_pose_in_world=obj_pose.astype(np.float32)

    print(f"obj_pose_shape: {obj_pose.shape}")
    print(f"camera_pose_shape: {camera_pose.shape}")
    