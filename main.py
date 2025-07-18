import argparse
from typing import Optional, Tuple, List
import numpy as np
import cv2
from pyapriltags import Detector

from src.type import Grasp
from src.utils import to_pose, get_pc, get_workspace_mask
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv
from src.sim.wrapper_env import get_grasps
from src.test.load_test import load_test_data
import src.pose_detection.pose_detector as pose_detector
from src.constants import OBSERVING_QPOS_DELTA
from visualize import plot_pointclouds_with_poses, random_sampling


def detect_driller_pose(img, depth, camera_matrix, camera_pose):
    """
    Detects the pose of driller, you can include your policy in args
    """
    full_pc_camera = get_pc(depth, camera_matrix)
    full_pc_world = (
        np.einsum("ab,nb->na", camera_pose[:3, :3], full_pc_camera)
        + camera_pose[:3, 3]
    )
    pc_mask = get_workspace_mask(full_pc_world)
    sel_pc_idx = np.random.randint(0, np.sum(pc_mask), 1024)
    pc_camera = full_pc_camera[pc_mask][sel_pc_idx]
    rel_pose = pose_detector.detect_pose(pc_camera)
    pose = camera_pose @ rel_pose
    # plot_pointclouds_with_poses(
    #     random_sampling(full_pc_world, 50000), 
    #     full_pc_world[pc_mask][sel_pc_idx], 
    #     gt, 
    #     pose
    # )
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
    
    #world 
    position_error_global = target_trans[:2] - current_trans[:2]
    
    #robot
    cos_yaw = np.cos(current_yaw)
    sin_yaw = np.sin(current_yaw)
    
    
    R_robot_to_world = np.array([
        [ cos_yaw, -sin_yaw],
        [ sin_yaw,  cos_yaw]
    ])

    R_world_to_robot = R_robot_to_world.T

    position_error_robot = R_world_to_robot @ position_error_global


    kp_forward = 1.2
    kp_lateral = 1.0   
    kp_angular = 1.2    
    
    
    vx = kp_forward * position_error_robot[0]
    
    
    vy = kp_lateral * position_error_robot[1]
    
    
    angle_error = target_yaw - current_yaw
    angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
    wz = kp_angular * angle_error

    max_linear_speed = 3.0
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
        cur_pose: np.ndarray,
        goal_pose: np.ndarray,
        v_max: float = 3.0,
        w_max: float = 1.2,
        Kp_xy: float = 4.0,
        Kp_yaw: float = 3.0       
    ) -> np.ndarray:

    if cur_pose is None or goal_pose is None:
        return np.zeros(3, dtype=np.float32)

    dp   = goal_pose[:2, 3] - cur_pose[:2, 3]         

    # yaw  = _yaw_from_rot(cur_pose [:3, :3])
    # dyaw = _yaw_from_rot(goal_pose[:3, :3]) - yaw
    # dyaw = (dyaw + np.pi) % (2*np.pi) - np.pi         
    yaw  = yaw_robot(cur_pose [:3, :3])     
    goal = yaw_robot(goal_pose[:3, :3])     
    dyaw = (goal - yaw + np.pi) % (2*np.pi) - np.pi

    # 世界 → 机器人坐标
    Rwr = np.array([[ np.cos(yaw),  np.sin(yaw)],
                    [-np.sin(yaw),  np.cos(yaw)]])
    dp_r = Rwr @ dp                                    

    vx =  Kp_xy * dp_r[0]
    vy =  Kp_xy * dp_r[1]
    wz =  Kp_yaw* dyaw

    vx = np.clip(vx, -v_max, v_max)
    vy = np.clip(vy, -v_max, v_max)
    wz = np.clip(wz, -w_max, w_max)
    return np.array([vx, vy, wz], dtype=np.float32)

def calculate_trajectory_length(traj):
    """计算轨迹在关节空间中的总长度"""
    if len(traj) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(traj)):
        # 计算相邻两个关节配置之间的欧几里得距离
        step_length = np.linalg.norm(traj[i] - traj[i-1])
        total_length += step_length
    
    return total_length

def plan_grasp(env: WrapperEnv, grasp: Grasp, grasp_config, *args, **kwargs) -> Optional[List[np.ndarray]]:
    """Try to plan a grasp trajectory for the given grasp. The trajectory is a list of joint positions. Return None if the trajectory is not valid."""
    # implement
    reach_steps = grasp_config['reach_steps']
    lift_steps = grasp_config['lift_steps']
    delta_dist = grasp_config['delta_dist']
    max_traj_length = grasp_config['max_traj_length']
    max_ik_attempts = grasp_config['max_ik_attempts']
    
    robot_model = env.humanoid_robot_model
    robot_cfg = env.humanoid_robot_cfg
    initial_arm_joints = robot_cfg.joint_init_qpos[robot_cfg.joint_arm_indices]
    
    target_grasp_trans, target_grasp_rot = grasp.trans, grasp.rot
    
    for attempt in range(max_ik_attempts):
        # 每次尝试时稍微扰动初始配置，以获得不同的IK解
        if attempt == 0:
            init_qpos_for_ik = initial_arm_joints
        else:
            # 添加小的随机扰动
            noise_scale = 0.1 * attempt  # 随着尝试次数增加扰动
            noise = np.random.uniform(-noise_scale, noise_scale, initial_arm_joints.shape)
            init_qpos_for_ik = initial_arm_joints + noise
        
        success_grasp, q_grasp_target = robot_model.ik(
            trans=target_grasp_trans, rot=target_grasp_rot,
            init_qpos=init_qpos_for_ik,
        )
        
        if not success_grasp:
            continue  # 如果IK失败，尝试下一个初始配置
        
        # 规划轨迹
        traj_reach = plan_move_qpos(
            begin_qpos=initial_arm_joints,
            end_qpos=q_grasp_target,
            steps=reach_steps
        )
        traj_reach = np.vstack([initial_arm_joints[None, :], traj_reach])
        
        # 检查轨迹长度
        reach_length = calculate_trajectory_length(traj_reach)
        print(f"Attempt {attempt+1}: Reach trajectory length: {reach_length:.3f} (max: {max_traj_length})")
        
        if reach_length <= max_traj_length:
            # 找到了满足长度要求的解，继续规划lift轨迹
            lift_offset_base = np.array([0, 0, delta_dist])
            target_lift_trans = target_grasp_trans + lift_offset_base
            target_lift_rot = target_grasp_rot
            success_lift, q_lift_target = robot_model.ik(
                trans=target_lift_trans,
                rot=target_lift_rot,
                init_qpos=q_grasp_target,
            )
            if not success_lift:
                print(f"Attempt {attempt+1}: IK failed for lift pose.")
                continue  # lift IK失败，尝试下一个
            
            traj_lift = plan_move_qpos(
                begin_qpos=q_grasp_target,
                end_qpos=q_lift_target,
                steps=lift_steps
            )
            traj_lift = np.vstack([q_grasp_target[None, :], traj_lift])
            
            print(f"Successfully found valid grasp plan on attempt {attempt+1}")
            return [np.array(traj_reach), np.array(traj_lift)]
        else:
            print(f"Attempt {attempt+1}: Trajectory too long ({reach_length:.3f} > {max_traj_length}), trying next IK solution")
    
    print(f"Failed to find valid grasp plan after {max_ik_attempts} attempts")
    return None

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

def yaw_robot(R):            
    return np.arctan2(R[1,0], R[0,0]) - np.pi/2

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
    
    observing_qpos = humanoid_init_qpos + OBSERVING_QPOS_DELTA # you can customize observing qpos to get wrist obs
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
    
        target_trans = table_pose[:3, 3] + table_pose[:3, :3] @ np.array([-0.35,-0.5, 0])
          
        table_heading = yaw_robot_in_world(table_pose[:3, :3])
        target_yaw    = (table_heading - np.pi/2) % (2*np.pi)   


        estimated_quad_trans = env.sim.default_quad_pose[:3].copy()
        estimated_quad_yaw =  target_yaw 

        
        last_valid_trans = None
        last_valid_rot = None
        head_scan_direction = 0.1  
        head_scan_angle = 0.0  
        scan_range = 0.8 
        scan_speed = 0.1 

        initial_container_pose =  env._init_container_pose.copy()
  
        

        pitch_ang  = head_init_qpos[1]     
        pitch_dir  = -1                   
        PITCH_STEP = 0.05                
        PITCH_UP   = 0.50                
        PITCH_DN   = 0.05                

        yaw_ang    = 0.0                  
        yaw_dir    = 1                    
        YAW_STEP   = 0.03                
        YAW_RANGE  = 0.25         
        last_seen_step = -1
        lost_thr=3         
        # ------------------------------------------------

        for step in range(forward_steps):

            tag_found = False              
            quad_cmd             = np.zeros(3)      
       
           
            if step % steps_per_camera_shot == 0:
                obs_head = env.get_obs(camera_id=0)
                # if step %20 ==0:
                #     env.debug_save_obs(obs_head, f'data/head_{step:04d}') 

                trans_marker_world, rot_marker_world = detect_marker_pose(
                    detector, obs_head.rgb, head_camera_params,
                    obs_head.camera_pose, tag_size=0.12)

                if trans_marker_world is not None:         
                    tag_found = True
                    last_seen_step = step
                    
                    trans_container_world = rot_marker_world @ np.array([0,0.31,0.02]) + trans_marker_world
                    rot_container_world   = rot_marker_world
                    quad_yaw_world        = yaw_robot_in_world(rot_container_world)
                    pose_container_world  = to_pose(trans_container_world, rot_container_world)

                    
                    quad_command = forward_quad_policy(
                        trans_container_world, quad_yaw_world,
                        target_trans,        target_yaw)

               
                    
                else:                                      
                    quad_command = np.zeros(3)             

            
            if step - last_seen_step > lost_thr:
                
                pitch_ang += pitch_dir * PITCH_STEP
                if pitch_ang > PITCH_UP or pitch_ang < PITCH_DN:
                    pitch_dir *= -1                        
                    pitch_ang  = np.clip(pitch_ang, PITCH_DN, PITCH_UP)

             
                yaw_ang += yaw_dir * YAW_STEP
                if abs(yaw_ang) > YAW_RANGE:
                    yaw_dir *= -1
                    yaw_ang  = np.clip(yaw_ang, -YAW_RANGE, YAW_RANGE)

            
            env.step_env(
                humanoid_head_qpos=np.array([yaw_ang, pitch_ang]),
                quad_command=quad_command
            )

           
            if tag_found:
                pos_err = np.linalg.norm(target_trans[:2] - trans_container_world[:2])
                if pos_err < 0.05:
                    print(f"Reached target position at step {step}")
                    break

    # --------------------------------------step 2: detect driller pose------------------------------------------------------

    
    if not DISABLE_GRASP:
        obs_wrist = env.get_obs(camera_id=1) # wrist camera
        # env.debug_save_obs(obs_wrist, "tmp_data")
        rgb, depth, camera_pose = obs_wrist.rgb, obs_wrist.depth, obs_wrist.camera_pose
        wrist_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[1].intrinsics
        driller_pose = detect_driller_pose(rgb, depth, wrist_camera_matrix, camera_pose)
        # metric judgement
        Metric['obj_pose'] = env.metric_obj_pose(driller_pose)
    
    
    # driller_pose = env.get_driller_pose()

    # --------------------------------------step 3: plan grasp and lift------------------------------------------------------
    if not DISABLE_GRASP:
        obj_pose = driller_pose.copy()
        grasps = get_grasps(args.obj) 
        grasps0_n = Grasp(grasps[0].trans, grasps[0].rot @ np.diag([-1,-1,1]), grasps[0].width)
        grasps2_n = Grasp(grasps[2].trans, grasps[2].rot @ np.diag([-1,-1,1]), grasps[2].width)
        valid_grasps = [grasps[0], grasps0_n, grasps[2], grasps2_n] # we have provided some grasps, you can choose to use them or yours
        grasp_config = dict( 
            reach_steps=50,
            lift_steps=30,
            delta_dist=0.05, 
            max_traj_length=2.5,
            max_ik_attempts=10,
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

        pregrasp_plan = plan_move_qpos(observing_qpos, reach_plan[0], steps=50) # pregrasp, change if you want
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
        print("\n===  Step-5: go home  ===")

  
        home_pose = env._init_container_pose.copy()
        # print(f"[dbg] home_pose:\n{home_pose}")
        home_yaw  = yaw_robot_in_world(home_pose[:3, :3]) + np.pi
        R_home    = np.array([[np.cos(home_yaw), -np.sin(home_yaw), 0],
                              [np.sin(home_yaw),  np.cos(home_yaw), 0],
                              [0,                 0,                1]])
        home_pose[:3, :3] = R_home
        
        backward_steps        = 800
        steps_per_cam_shot    = 5
        head_cam_mat          = env.sim.humanoid_robot_cfg.camera_cfg[0].intrinsics
        head_cam_params       = (head_cam_mat[0, 0], head_cam_mat[1, 1],
                                 head_cam_mat[0, 2], head_cam_mat[1, 2])

        pitch_ang  = head_init_qpos[1]   
        pitch_dir  = 1                 
        PITCH_STEP = 0.05
        PITCH_UP   = 0.50
        PITCH_DN   = 0.10

        yaw_ang    = 0.0                
        yaw_dir    = 1                  
        YAW_STEP   = 0.03
        YAW_RANGE  = 0.25
        last_seen_step = -1
        lost_thr=3
        # ------------------------------------------------------

        for step in range(backward_steps):

            tag_found            = False
            quad_cmd             = np.zeros(3)      
            pose_container_world = None

           
            if step % steps_per_cam_shot == 0:
                obs_head = env.get_obs(camera_id=0)
                # if step % 20 == 0:                       
                #     env.debug_save_obs(obs_head, f'data/back_{step:04d}')

                trans_tag, rot_tag = detect_marker_pose(
                    detector, obs_head.rgb, head_cam_params,
                    obs_head.camera_pose, tag_size=0.12)

                if trans_tag is not None:                
                    tag_found = True
                    last_seen_step = step

                    # —— 当前容器 pose
                    trans_cont = rot_tag @ np.array([0, -0.31, 0.02]) + trans_tag
                    pose_container_world = to_pose(trans_cont, rot_tag)

                    # —— 计算回家指令
                    quad_cmd = backward_quad_policy(pose_container_world, home_pose)

                    # print(f"[dbg] pose_container_wolrd:\n{pose_container_world}")
                    # print(f"[dbg] home_pose:\n{home_pose}")
                    # print(f"[dbg] quad_cmd: {quad_cmd}")

                    

            # ---------- 没检测到 tag ➜ 更新扫描角 ----------
            if step - last_seen_step > lost_thr:
                # ① 俯仰主扫
                pitch_ang += pitch_dir * PITCH_STEP
                if pitch_ang > PITCH_UP or pitch_ang < PITCH_DN:
                    pitch_dir *= -1
                    pitch_ang  = np.clip(pitch_ang, PITCH_DN, PITCH_UP)

                # ② 水平微摆
                yaw_ang += yaw_dir * YAW_STEP
                if abs(yaw_ang) > YAW_RANGE:
                    yaw_dir *= -1
                    yaw_ang  = np.clip(yaw_ang, -YAW_RANGE, YAW_RANGE)

            # ---------------- 执行动作 -----------------
            env.step_env(
                humanoid_head_qpos=np.array([yaw_ang, pitch_ang]),
                quad_command=quad_cmd
            )

            # --------- 判断是否已经回到家 ---------
            if tag_found:
                err_xy = np.linalg.norm(trans_cont[:2] - home_pose[:2, 3])
                # print(f"[dbg] GT dist_xy = {err_xy:.3f}")  # 调试输出
                # gt_now = env.get_container_pose()
                # print(f"[dbg] real_dist = {np.linalg.norm(gt_now[:2,3]-home_pose[:2,3]):.3f}")
                if err_xy < 0.07:                     
                    print(f"  ✓ quad returned after {step} steps")
                    break


    # test the metrics
    Metric["drop_precision"] = Metric["drop_precision"] or env.metric_drop_precision()
    Metric["quad_return"] = Metric["quad_return"] or env.metric_quad_return()

    print("Metrics:", Metric) 

    print("Simulation completed.")
    env.close()

if __name__ == "__main__":
    main()