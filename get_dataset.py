import argparse
from typing import Optional, Tuple, List
import numpy as np
import cv2
from tqdm import tqdm
from pyapriltags import Detector

from src.constants import OBSERVING_QPOS_DELTA
from src.type import Grasp
from src.utils import to_pose
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv
from src.sim.wrapper_env import get_grasps
from src.test.load_test import load_test_data


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


TESTING = True
DISABLE_GRASP = False
DISABLE_MOVE = False


def main():
    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=0)
    parser.add_argument("--reset_wait_steps", type=int, default=100)
    parser.add_argument("--n", type=int, default=1, help="Number of datasets to generate")

    args = parser.parse_args()

    num = args.n

    env_config = WrapperEnvConfig(
        humanoid_robot=args.robot,
        obj_name=args.obj,
        headless=args.headless,
        ctrl_dt=args.ctrl_dt,
        reset_wait_steps=args.reset_wait_steps,
    )

    env = WrapperEnv(env_config)


    for i in tqdm(range(num), desc="Generating datasets"):
        while True:
            table_trans = np.array([
                np.random.uniform(0.55, 0.61),  # x 在 [0.55, 0.61] 随机
                np.random.uniform(0.38, 0.47),  # y 在 [0.38, 0.47] 随机
                np.random.uniform(0.67, 0.74)   # z 在 [0.67, 0.74] 随机
            ])
            size_option1 = np.array([0.68, 0.36, 0.02])
            size_option2 = np.array([0.72, 0.42, 0.02])
            table_size = size_option1 if np.random.rand() < 0.75 else size_option2
            # print(table_trans, table_size)
            table_pose = np.eye(4)
            table_pose[:3, 3] = table_trans

            env.set_table_obj_config(
                table_pose=table_pose,
                table_size=table_size,
                obj_pose=None
            )
            env.launch()
            env.reset(humanoid_qpos=env.sim.humanoid_robot_cfg.joint_init_qpos)
            humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos[:7]
            
            head_init_qpos = np.array([-0.05,0.35]) # you can adjust the head init qpos to find the driller

            env.step_env(humanoid_head_qpos=head_init_qpos)
            if env.get_driller_pose()[2,3] > 0.6:
                break
        
        # print(env.config.table_pose)
        
        observing_qpos = humanoid_init_qpos + OBSERVING_QPOS_DELTA
        # 为了保证能看到物体，调整qpos
        init_plan = plan_move_qpos(humanoid_init_qpos, observing_qpos, steps=20)
        execute_plan(env, init_plan)

        obs_wrist = env.get_obs(camera_id=1)
        env.save_obs(obs_wrist)
    
        env.close()

    

if __name__ == "__main__":
    main()