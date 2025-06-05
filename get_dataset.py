import argparse
from typing import Optional, Tuple, List
import numpy as np
import cv2
from tqdm import tqdm
from pyapriltags import Detector

from src.type import Grasp
from src.utils import to_pose
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv
from src.sim.wrapper_env import get_grasps


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
        env.launch()
        env.reset(humanoid_qpos=env.sim.humanoid_robot_cfg.joint_init_qpos)
        humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos[:7]
        
        head_init_qpos = np.array([-0.05,0.35]) # you can adjust the head init qpos to find the driller

        env.step_env(humanoid_head_qpos=head_init_qpos)
        
        observing_qpos = humanoid_init_qpos + np.random.uniform(low=-0.03, high=0.03, size=7)
        init_plan = plan_move_qpos(humanoid_init_qpos, observing_qpos, steps=20)
        execute_plan(env, init_plan)

        obs_wrist = env.get_obs(camera_id=1)  # wrist camera
        env.save_obs(obs_wrist)
        env.close()

    

if __name__ == "__main__":
    main()