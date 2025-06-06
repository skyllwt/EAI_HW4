import argparse
import os
from typing import Optional, Tuple, List
import numpy as np
import cv2
from pyapriltags import Detector

from src.type import Grasp
from src.utils import to_pose, get_pc, get_workspace_mask
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv
from src.sim.wrapper_env import get_grasps
from src.test.load_test import load_test_data
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
pio.renderers.default = "browser"
import debugpy


# debugpy.listen(("localhost", 5678))
# print("等待调试器连接...")
# debugpy.wait_for_client()
# print("调试器已连接")


def detect_driller_pose(img, depth, camera_matrix, camera_pose, *args, **kwargs):
    """
    Detects the pose of driller, you can include your policy in args
    """
    # implement the detection logic here
    # 
    pose = np.eye(4)
    return pose



def detect_marker_pose(
        detector: Detector, 
        img: np.ndarray, 
        camera_params: tuple, 
        camera_pose: np.ndarray,
        tag_size: float = 0.12
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
  
    detections = detector.detect(gray, estimate_tag_pose=True, 
                                camera_params=camera_params, 
                                tag_size=tag_size)
    
    if len(detections) > 0:
       
        detection = detections[0]
   
        trans_marker_camera = detection.pose_t[:, 0]
        rot_marker_camera = detection.pose_R
  
        rot_camera_world = camera_pose[:3, :3]
        trans_camera_world = camera_pose[:3, 3]
        
        trans_marker_world = rot_camera_world @ trans_marker_camera + trans_camera_world
        rot_marker_world = rot_camera_world @ rot_marker_camera
        
        return trans_marker_world, rot_marker_world
    
    return None, None


def plan_move_qpos(begin_qpos, end_qpos, steps=50) -> np.ndarray:
    delta_qpos = (end_qpos - begin_qpos) / steps
    cur_qpos = begin_qpos.copy()
    traj = []
    
    for _ in range(steps):
        cur_qpos += delta_qpos
        traj.append(cur_qpos.copy())
    
    return np.array(traj)
def execute_plan(env: WrapperEnv, plan):
    """Execute the plan in the environment."""
    for step in range(len(plan)):
        env.step_env(
            humanoid_action=plan[step],
        )


def random_sampling(points: np.ndarray, M: int) -> np.ndarray:
    N = points.shape[0]
    if M >= N:
        return points.copy()
    # np.random.choice 默认是不放回抽样，需要指定 replace=False
    indices = np.random.choice(N, size=M, replace=False)
    return points[indices]

def visualize_pc(points: np.ndarray):
    # points = points[get_workspace_mask(points)]
    points = random_sampling(points, 50000)
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



def main():
    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=0)
    parser.add_argument("--reset_wait_steps", type=int, default=100)
    parser.add_argument("--test_id", type=int, default=0)

    args = parser.parse_args()

    detector = Detector(
        families="tagStandard52h13",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    env_config = WrapperEnvConfig(
        humanoid_robot=args.robot,
        obj_name=args.obj,
        headless=args.headless,
        ctrl_dt=args.ctrl_dt,
        reset_wait_steps=args.reset_wait_steps,
    )


    env = WrapperEnv(env_config)
    

    env.launch()
    env.reset(humanoid_qpos=env.sim.humanoid_robot_cfg.joint_init_qpos)
    humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos[:7]
    Metric = {
        'obj_pose': False,
        'drop_precision': False,
        'quad_return': False,
    }
    
    head_init_qpos = np.array([-0.05,0.35]) # you can adjust the head init qpos to find the driller

    env.step_env(humanoid_head_qpos=head_init_qpos)
    
    observing_qpos = humanoid_init_qpos + np.array([0.01,0,0,0,0,0,0]) # you can customize observing qpos to get wrist obs
    init_plan = plan_move_qpos(humanoid_init_qpos, observing_qpos, steps = 20)
    execute_plan(env, init_plan)


    obs_wrist = env.get_obs(camera_id=1) # wrist camera
    rgb, depth, camera_pose = obs_wrist.rgb, obs_wrist.depth, obs_wrist.camera_pose
    wrist_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[1].intrinsics
    # env.debug_save_obs(obs_wrist, "tmp_data")

    full_pc_camera = get_pc(
        depth, wrist_camera_matrix
    )
    full_pc_world = (
        np.einsum("ab,nb->na", camera_pose[:3, :3], full_pc_camera)
        + camera_pose[:3, 3]
    )

    visualize_pc(full_pc_world)

    pc_mask = get_workspace_mask(full_pc_world)
    input("dafasf")

    visualize_pc(full_pc_world[pc_mask])
    print(pc_mask.sum())


    driller_pose = detect_driller_pose(rgb, depth, wrist_camera_matrix, camera_pose)
    # metric judgement
    Metric['obj_pose'] = env.metric_obj_pose(driller_pose)
    
    driller_pose = env.get_driller_pose()

    input("Press Enter to continue...")
    env.close()
    return

if __name__ == "__main__":
    main()