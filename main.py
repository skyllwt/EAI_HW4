import argparse
from typing import Optional, Tuple, List
import numpy as np
import cv2
from pyapriltags import Detector

from src.type import Grasp
from src.utils import to_pose
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv
from src.sim.wrapper_env import get_grasps
from src.test.load_test import load_test_data



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


def _yaw_from_rot(R: np.ndarray) -> float:
 
    return np.arctan2(R[1, 0], R[0, 0])

def forward_quad_policy(current_trans, current_yaw, target_trans, target_yaw, *args, **kwargs):
    
    # 位置误差（在全局坐标系）
    position_error_global = target_trans[:2] - current_trans[:2]
    
    # 将全局误差转换到机器人坐标系
    cos_yaw = np.cos(current_yaw)
    sin_yaw = np.sin(current_yaw)
    
    
    R_robot_to_world = np.array([
        [ cos_yaw, -sin_yaw],
        [ sin_yaw,  cos_yaw]
    ])

    R_world_to_robot = R_robot_to_world.T

    position_error_robot = R_world_to_robot @ position_error_global

    # 控制增益 - 注意符号！
    kp_forward = 1.2    # 前后方向增益
    kp_lateral = 1.0    # 侧向移动增益
    kp_angular = 1.2    # 旋转增益
    
    # 计算速度命令 - 注意符号！
    # 如果目标在前方(position_error_robot[0]>0)，应该正向前进
    vx = kp_forward * position_error_robot[0]
    
    # 如果目标在左侧(position_error_robot[1]>0)，应该向左移动
    vy = kp_lateral * position_error_robot[1]
    
    # 角度误差处理不变
    angle_error = target_yaw - current_yaw
    angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
    wz = kp_angular * angle_error

    max_linear_speed = 0.6
    max_angular_speed = 1.0
    vx = np.clip(vx, -max_linear_speed, max_linear_speed)
    vy = np.clip(vy, -max_linear_speed, max_linear_speed)
    wz = np.clip(wz, -max_angular_speed, max_angular_speed)

    return np.array([vx, vy, wz])


def is_close(pose1, pose2, threshold=0.05, thresh_yaw=np.deg2rad(5)):
    if pose1 is None or pose2 is None:
        return False
    dp = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
    dyaw = abs(_yaw_from_rot(pose1[:3, :3]) -
               _yaw_from_rot(pose2[:3, :3]))
    dyaw = (dyaw + np.pi) % (2*np.pi) - np.pi   # wrap to [-π, π]
    return (dp < threshold) and (abs(dyaw) < thresh_yaw)


def backward_quad_policy(
        pose_container_world: np.ndarray,
        target_container_pose: np.ndarray,
        v_max: float = 0.30,          
        w_max: float = 1.00,          
        Kp_xy: float = 1.2,           
        Kp_yaw: float = 3.0,          
        slow_dist: float = 0.30       
    ) -> np.ndarray:
    # ----------- 边界情况：没观测到 tag 时保持静止 -----------------
    if pose_container_world is None or target_container_pose is None:
        return np.zeros(3, dtype=np.float32)

    # ----------- 1) 位置误差（只考虑平面 XY） ---------------------
    cur_pos  = pose_container_world[:3, 3]
    goal_pos = target_container_pose[:3, 3]
    d_xy     = goal_pos[:2] - cur_pos[:2]            # shape (2,)

    # ----------- 2) 朝向误差（绕 Z） -----------------------------
    cur_yaw  = _yaw_from_rot(pose_container_world[:3, :3])
    goal_yaw = _yaw_from_rot(target_container_pose[:3, :3])
    d_yaw    = np.arctan2(np.sin(goal_yaw - cur_yaw),
                          np.cos(goal_yaw - cur_yaw))  # wrap → [-π, π]

    # ----------- 3) 近目标时自动减小上限 --------------------------
    dist = np.linalg.norm(d_xy)
    if dist < slow_dist:
        v_max *= 0.5
        w_max *= 0.5

    # ----------- 4) 比例控制 + 饱和 ------------------------------
    vx = np.clip(Kp_xy * d_xy[0], -v_max, v_max)
    vy = np.clip(Kp_xy * d_xy[1], -v_max, v_max)
    wz = np.clip(Kp_yaw * d_yaw,  -w_max, w_max)

    return np.array([vx, vy, wz], dtype=np.float32)



def plan_grasp(env: WrapperEnv, grasp: Grasp, grasp_config, *args, **kwargs) -> Optional[List[np.ndarray]]:
    """Try to plan a grasp trajectory for the given grasp. The trajectory is a list of joint positions. Return None if the trajectory is not valid."""
    # implement
    reach_steps = grasp_config['reach_steps']
    lift_steps = grasp_config['lift_steps']
    delta_dist = grasp_config['delta_dist']

    traj_reach = []
    traj_lift = []
    succ = False
    if not succ: return None

    return [np.array(traj_reach), np.array(traj_lift)]

def plan_move(env: WrapperEnv, begin_qpos, begin_trans, begin_rot, end_trans, end_rot, steps = 50, *args, **kwargs):
    """Plan a trajectory moving the driller from table to dropping position"""
    # implement
    traj = []

    succ = False
    if not succ: return None
    return traj

def open_gripper(env: WrapperEnv, steps = 10):
    for _ in range(steps):
        env.step_env(gripper_open=1)
def close_gripper(env: WrapperEnv, steps = 10):
    for _ in range(steps):
        env.step_env(gripper_open=0)
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


def rotation_matrix_to_euler_angles(R):
    """将旋转矩阵转换为欧拉角 (roll, pitch, yaw)"""
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def yaw_robot_in_world(R_world_obj):
    """返回“机器狗真正前方”在世界坐标系里的 yaw 角"""
    return rotation_matrix_to_euler_angles(R_world_obj)[2] - np.pi/2

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
    if TESTING:
        data_dict = load_test_data(args.test_id)
        env.set_table_obj_config(
            table_pose=data_dict['table_pose'],
            table_size=data_dict['table_size'],
            obj_pose=data_dict['obj_pose']
        )
        env.set_quad_reset_pos(data_dict['quad_reset_pos'])

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


    # --------------------------------------step 1: move quadruped to dropping position--------------------------------------
    if not DISABLE_MOVE:
        forward_steps = 1000  # number of steps that quadruped walk to dropping position
        steps_per_camera_shot = 5  # number of steps per camera shot
        head_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[0].intrinsics
        head_camera_params = (head_camera_matrix[0, 0], head_camera_matrix[1, 1], 
                            head_camera_matrix[0, 2], head_camera_matrix[1, 2])
        
      
        table_pose = data_dict['table_pose']
        target_trans = table_pose[:3, 3] + table_pose[:3, :3] @ np.array([0.6,-0.5, 0])
          
        target_rot = table_pose[:3, :3]
        target_yaw = yaw_robot_in_world(table_pose[:3, :3])
        
        estimated_quad_trans = env.sim.default_quad_pose[:3].copy()
        estimated_quad_yaw =  target_yaw # 初始 yaw

        
        last_valid_trans = None
        last_valid_rot = None
        head_scan_direction = 0.1  
        head_scan_angle = 0.0  
        scan_range = 0.8 
        scan_speed = 0.1 

        
        for step in range(forward_steps):
            current_pose_source = "estimated"

            if step % steps_per_camera_shot == 0:
                obs_head = env.get_obs(camera_id=0)  

                # 这里可以保存图像
                # if step % 20 == 0:
                #     env.debug_save_obs(obs_head, f'data/head_{step:04d}') 

                trans_marker_world, rot_marker_world = detect_marker_pose(
                    detector, 
                    obs_head.rgb, 
                    head_camera_params,
                    obs_head.camera_pose,
                    tag_size=0.12
                )
                
                if trans_marker_world is not None:
                    # 更新有效的标记位置
                    last_valid_trans = trans_marker_world
                    last_valid_rot = rot_marker_world
                    
                    # 计算箱子的全局位置（相对于标记）
                    trans_container_world = rot_marker_world @ np.array([0, 0.31, 0.02]) + trans_marker_world
                    rot_container_world = rot_marker_world
                    quad_yaw_world      = yaw_robot_in_world(rot_container_world)

                    pose_container_world = to_pose(trans_container_world, rot_container_world)

                    estimated_quad_trans = trans_container_world
                    estimated_quad_yaw =  quad_yaw_world 

                    current_pose_source = "observed"
                    
                    # 重置头部扫描
                    head_scan_angle = 0.0
                    head_scan_direction = abs(head_scan_direction) * np.sign(head_scan_direction) # 确保方向一致
                else:
                    estimated_rot_matrix = np.array([
                    [np.cos(estimated_quad_yaw + np.pi/2), -np.sin(estimated_quad_yaw + np.pi/2), 0], # 加上基准调整
                    [np.sin(estimated_quad_yaw + np.pi/2), np.cos(estimated_quad_yaw + np.pi/2),  0],
                    [0,                                   0,                                   1]
                ])

                    trans_container_world = estimated_rot_matrix @ np.array([0, 0.31, 0.02]) + estimated_quad_trans
                    rot_container_world = estimated_rot_matrix # 容器的旋转就是四足机器人的旋转

            
            
            current_yaw_for_policy = rotation_matrix_to_euler_angles(rot_container_world)[2]
            quad_command = forward_quad_policy(
                trans_container_world, 
                quad_yaw_world,
                target_trans,
                target_yaw
            )
            
            # 更新四足机器人的估计姿态 (死推)
            # 这是基于 quad_command 和 ctrl_dt 的预测
            vx_robot = quad_command[0]
            vy_robot = quad_command[1]
            wz_robot = quad_command[2]

            # 转换为世界坐标系的速度
            cos_yaw_est = np.cos(estimated_quad_yaw)
            sin_yaw_est = np.sin(estimated_quad_yaw)
            vx_world =  vx_robot * np.cos(estimated_quad_yaw) \
                - vy_robot * np.sin(estimated_quad_yaw)
            vy_world =  vx_robot * np.sin(estimated_quad_yaw) \
                 + vy_robot * np.cos(estimated_quad_yaw)

            estimated_quad_trans[0] += vx_world * env.config.ctrl_dt
            estimated_quad_trans[1] += vy_world * env.config.ctrl_dt
            estimated_quad_yaw += wz_robot * env.config.ctrl_dt
            estimated_quad_yaw = np.arctan2(np.sin(estimated_quad_yaw), np.cos(estimated_quad_yaw)) # 确保在[-pi, pi]

            # 头部扫描逻辑
            move_head = False
            head_qpos = head_init_qpos
            if trans_marker_world is None: # 如果没有检测到标记
                move_head = True
                head_scan_angle += head_scan_direction * scan_speed
                
                # 限制扫描范围，并反转方向
                if abs(head_scan_angle) > scan_range / 2:
                    head_scan_direction *= -1 # 反转方向
                    head_scan_angle = np.clip(head_scan_angle, -scan_range / 2, scan_range / 2) # 限制在边界
                
                head_qpos = np.array([head_scan_angle, head_init_qpos[1]]) # 保持俯仰角不变

            
            # 执行动作
            env.step_env(
                humanoid_head_qpos=head_qpos if move_head else None,
                quad_command=quad_command
            )
            
            # 检查是否到达目标位置
            position_error = np.linalg.norm(target_trans[:2] - trans_container_world[:2])
            yaw_error = abs(target_yaw - current_yaw_for_policy)  

            # 调试输出
            # print(f"Step {step}: Position error: {position_error:.4f}, Yaw error: {yaw_error:.4f}")
            # print(f"{target_trans[:2]} vs {trans_container_world[:2]}") 
            # print(f"cmd0:{quad_command[0]:.4f}, cmd1:{quad_command[1]:.4f}")

            if position_error < 0.05 :  
                print(f"Reached target position at step {step}")
                break
                

    # --------------------------------------step 2: detect driller pose------------------------------------------------------
    if not DISABLE_GRASP:
        obs_wrist = env.get_obs(camera_id=1) # wrist camera
        rgb, depth, camera_pose = obs_wrist.rgb, obs_wrist.depth, obs_wrist.camera_pose
        wrist_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[1].intrinsics
        driller_pose = detect_driller_pose(rgb, depth, wrist_camera_matrix, camera_pose)
        # metric judgement
        Metric['obj_pose'] = env.metric_obj_pose(driller_pose)


    # --------------------------------------step 3: plan grasp and lift------------------------------------------------------
    if not DISABLE_GRASP:
        obj_pose = driller_pose.copy()
        grasps = get_grasps(args.obj) 
        grasps0_n = Grasp(grasps[0].trans, grasps[0].rot @ np.diag([-1,-1,1]), grasps[0].width)
        grasps2_n = Grasp(grasps[2].trans, grasps[2].rot @ np.diag([-1,-1,1]), grasps[2].width)
        valid_grasps = [grasps[0], grasps0_n, grasps[2], grasps2_n] # we have provided some grasps, you can choose to use them or yours
        grasp_config = dict( 
            reach_steps=0,
            lift_steps=0,
            delta_dist=0, 
        ) # the grasping design in assignment 2, you can choose to use it or design yours

        for obj_frame_grasp in valid_grasps:
            robot_frame_grasp = Grasp(
                trans=obj_pose[:3, :3] @ obj_frame_grasp.trans
                + obj_pose[:3, 3],
                rot=obj_pose[:3, :3] @ obj_frame_grasp.rot,
                width=obj_frame_grasp.width,
            )
            grasp_plan = plan_grasp(env, robot_frame_grasp, grasp_config)
            if grasp_plan is not None:
                break
        if grasp_plan is None:
            print("No valid grasp plan found.")
            env.close()
            return
        reach_plan, lift_plan = grasp_plan

        pregrasp_plan = plan_move_qpos(env, observing_qpos, reach_plan[0], steps=50) # pregrasp, change if you want
        execute_plan(env, pregrasp_plan)
        open_gripper(env)
        execute_plan(env, reach_plan)
        close_gripper(env)
        execute_plan(env, lift_plan)


    # --------------------------------------step 4: plan to move and drop----------------------------------------------------
    if not DISABLE_GRASP and not DISABLE_MOVE:
        # implement your moving plan
        #
        move_plan = plan_move(
            env=env,
        ) 
        execute_plan(env, move_plan)
        open_gripper(env)


    # --------------------------------------step 5: move quadruped backward to initial position------------------------------
    if not DISABLE_MOVE:
        # implement
        #
        backward_steps = 1000 # customize by yourselves
        for step in range(backward_steps):
            # same as before, please implement this
            #
            quad_command = backward_quad_policy()
            env.step_env(
                quad_command=quad_command
            )
        

    # test the metrics
    Metric["drop_precision"] = Metric["drop_precision"] or env.metric_drop_precision()
    Metric["quad_return"] = Metric["quad_return"] or env.metric_quad_return()

    print("Metrics:", Metric) 

    print("Simulation completed.")
    env.close()

if __name__ == "__main__":
    main()