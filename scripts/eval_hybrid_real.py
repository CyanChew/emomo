import argparse
import copy
import pprint
import numpy as np
import torch
import pyrallis
import signal
import sys
import threading

from envs.common_real_env_cfg import RealEnvConfig
from teleop.policies import TeleopPolicy
from scipy.spatial.transform import Rotation as R
import common_utils

import os
import time
from itertools import count
from constants import POLICY_CONTROL_PERIOD
from common_utils.eval_utils import (
    check_for_interrupt,
)

from envs.utils.camera_utils import pcl_from_obs
from interactive_scripts.dataset_recorder import ActMode
from scripts.train_waypoint import load_waypoint
from scripts.train_dense import load_model

# Global vars
env = None
cleanup_lock = threading.Lock()
cleanup_done = False
arm_moved = False

# Signal handler for Ctrl+C
def handle_signal(signum, frame):
    global cleanup_done, env
    with cleanup_lock:
        if cleanup_done:
            print("[Force Exit] Cleanup already started. Forcing exit.")
            sys.exit(1)
        print("\n[Signal] Ctrl+C or Ctrl+\ received. Cleaning up...")
        cleanup_done = True
        if env is not None:
            try:
                env.close()
                print("Closed env.")
            except Exception as e:
                print(f"[Error] Failed to close env cleanly: {e}")
        sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGQUIT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def run_dense_mode(env, dense_policy, dense_dataset, recorder, prev_obs):
    cached_actions = []
    mode = ActMode.Dense.value

    consecutive_modes_required = 5
    WAYPOINT_THRESH = 0.5
    TERMINATE_THRESH = 1.9
    mode_history = []

    start_time = time.time()

    for step_idx in count():
        # Enforce desired control freq
        step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
        while time.time() < step_end_time:
            time.sleep(0.0001)

        with env.stopwatch.time('get_obs'):
            obs = env.get_obs()

        with env.stopwatch.time('teleop_policy.step'):
            teleop_intervention = env.teleop_policy.step(obs)

        if teleop_intervention == 'end_episode':
            return ActMode.Terminate.value, prev_obs

        with env.stopwatch.time('process_obs'):
            dense_obs = dense_dataset.process_observation(obs)
            for k, v in dense_obs.items():
                dense_obs[k] = v.cuda()
    
        if len(cached_actions) == 0:
            with env.stopwatch.time('inference'):
                action_seq = dense_policy.act(dense_obs)
            for action in action_seq.split(1, dim=0):
                cached_actions.append(action.squeeze(0))

        action = cached_actions.pop(0)

        if not dense_dataset.cfg.wbc:
            ee_pos, ee_quat, gripper_pos, base_pose, raw_mode = action.split([3, 4, 1, 3, 1])
            base_pose = base_pose.detach().cpu().numpy()

        else:
            ee_pos, ee_quat, gripper_pos, raw_mode = action.split([3, 4, 1, 1])
            base_pose = np.zeros(3)

        print(raw_mode)
        mode_history.append(raw_mode)
        if len(mode_history) >= consecutive_modes_required:
            if np.all(np.array(mode_history) < WAYPOINT_THRESH):
                return ActMode.ArmWaypoint.value if env.cfg.wbc else ActMode.BaseWaypoint.value, prev_obs
            elif np.all(np.array(mode_history) > TERMINATE_THRESH):
                return ActMode.Terminate.value, prev_obs
            mode_history = []
        

        ee_pos = ee_pos.detach().cpu().numpy()
        ee_quat = ee_quat.detach().cpu().numpy()
        gripper_pos = gripper_pos.detach().cpu().numpy()
        raw_mode = raw_mode.detach().cpu().numpy()

        if dense_dataset.cfg.delta_actions: # absolute
            ee_pos += obs['arm_pos']
            ee_rot = R.from_quat(ee_quat) * R.from_quat(obs['arm_quat'])
            ee_quat = ee_rot.as_quat()
            ee_quat /= np.linalg.norm(ee_quat)
            if ee_quat[3] < 0.0:  # Enforce quaternion uniqueness
                np.negative(ee_quat, out=ee_quat)
            base_pose += obs['base_pose']

        if dense_dataset.cfg.wbc:
            base_pose = np.zeros(3)

        action = {
            "base_pose": base_pose,
            "arm_pos": ee_pos,
            "arm_quat": ee_quat,
            "gripper_pos": gripper_pos,
        }
        #print('here dense', action)
        env.step(action)

        if recorder is not None:
            recorder.add_numpy(obs, ["base1_image", "base2_image", "wrist_image"], color=(255, 140, 0))

    return mode, prev_obs  # unreachable, but keeps the signature

def run_waypoint_mode(env, waypoint_policy, num_pass, recorder, curr_mode):
    global arm_moved
    global wpt
    obs = env.get_obs()
    interrupt = False

    while curr_mode in [ActMode.ArmWaypoint.value, ActMode.BaseWaypoint.value]:
        if recorder:
            recorder.add_numpy(obs, ["base1_image", "base2_image", "wrist_image"])

        points, colors = pcl_from_obs(obs, env.cfg)
        proprio = np.hstack((obs["arm_pos"], obs["arm_quat"], obs["gripper_pos"], obs["base_pose"]))

        with torch.no_grad():
            _, pos_cmd, rot_cmd, gripper_cmd, next_mode = waypoint_policy.inference(
                torch.from_numpy(points).float(),
                torch.from_numpy(colors).float(),
                torch.from_numpy(proprio).float(),
                num_pass=num_pass,
            )

            if waypoint_policy.cfg.use_euler:
                quat_cmd = R.from_euler("xyz", rot_cmd).as_quat()
            else:
                quat_cmd = rot_cmd

        if (not env.cfg.wbc) and quat_cmd[0] < 0.55:
            curr_mode = ActMode.BaseWaypoint.value

        if curr_mode == ActMode.ArmWaypoint.value:
            if quat_cmd[3] < 0:
                np.negative(quat_cmd, out=quat_cmd)

            if not arm_moved:
                arm_moved = True

                if env.cfg.wbc:
                    reference_pos = obs["arm_pos"] + [0.2, 0.0, 0.19]
                else:
                    reference_pos = obs["arm_pos"] + [0.2, 0.0, 0.19]

                reference_quat = obs["arm_quat"]
                reached, err, interrupt = env.move_to_arm_waypoint(reference_pos, reference_quat, 1.0, phone_interventions=True, recorder=recorder)

            reached, err, interrupt = env.move_to_arm_waypoint(pos_cmd, quat_cmd, gripper_cmd, phone_interventions=True, recorder=recorder)

        else:

            base_pose = np.array([pos_cmd[0], pos_cmd[1], R.from_quat(quat_cmd).as_euler("xyz")[2]])
            reached, err, interrupt = env.move_to_base_waypoint(base_pose, phone_interventions=True, recorder=recorder)

        obs = env.get_obs()
        curr_mode = next_mode

        if interrupt:
            return ActMode.Terminate.value, obs

    return curr_mode, obs

def eval_hybrid(
    waypoint_policy,
    dense_policy,
    dense_dataset,
    env,
    episode_idx,
    num_pass,
    save_dir,
    record,
):
    global arm_moved
    assert not waypoint_policy.training
    assert not dense_policy.training

    env.reset()
    env.teleop_policy.reset()
    arm_moved = False

    recorder = None
    if record:
        recorder = common_utils.Recorder(save_dir)

    mode = ActMode.ArmWaypoint.value if env.cfg.wbc else ActMode.BaseWaypoint.value
    obs = env.get_obs()

    while mode != ActMode.Terminate.value:
        if mode in [ActMode.ArmWaypoint.value, ActMode.BaseWaypoint.value]:
            mode, obs = run_waypoint_mode(env, waypoint_policy, num_pass, recorder, curr_mode=mode)
        elif mode == ActMode.Dense.value:
            mode, obs = run_dense_mode(env, dense_policy, dense_dataset, recorder, obs)

    if recorder and not (input('Save?') in ['n', 'N']):
        recorder.save(f"{episode_idx}", fps=10)
        return True

    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env_cfg", type=str, required=True)
    parser.add_argument("-w", "--waypoint_model", type=str, required=True)
    parser.add_argument("-d", "--dense_model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=99999)
    parser.add_argument("--num_episode", type=int, default=20)
    parser.add_argument("--num_pass", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="rollouts")
    parser.add_argument("--record", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    try:
        env_cfg = pyrallis.load(RealEnvConfig, open(args.env_cfg, "r"))
        if env_cfg.wbc:
            from envs.real_env_wbc import RealEnv
        else:
            from envs.real_env_base_arm import RealEnv

        waypoint_policy = load_waypoint(args.waypoint_model, device="cuda").cuda()
        waypoint_policy.eval()

        dense_policy, dense_dataset, _ = load_model(args.dense_model, "cuda", load_only_one=True)
        dense_policy.eval()

        if dense_policy.cfg.use_ddpm:
            common_utils.cprint(f"Warning: overriding to use ddim with 10 steps")
            dense_policy.cfg.use_ddpm = 0
            dense_policy.cfg.ddim.num_inference_timesteps = 10

        env = RealEnv(env_cfg)
        if env.teleop_policy is None: 
            env.teleop_policy = TeleopPolicy()

        N = 0
        if os.path.exists('rollouts'):
            N = len([fn for fn in os.listdir('rollouts') if 'mp4' in fn])

        for episode_idx in range(N, args.num_episode + N):
            env.drive_to_reset()
            eval_hybrid(
                waypoint_policy,
                dense_policy,
                dense_dataset,
                env,
                episode_idx,
                num_pass=args.num_pass,
                save_dir=args.save_dir,
                record=args.record,
            )
            print(f"[{episode_idx+1}/{args.num_episode}]")
            print(common_utils.wrap_ruler("", max_len=80))

    except Exception as e:
        print(f"[Error] Unhandled exception: {e}")

    finally:
        if not cleanup_done:
            try:
                if env is not None:
                    env.stopwatch.summary()
                    env.close()
                    print("Closed env.")
            except Exception as e:
                print(f"[Error] Cleanup failed in finally block: {e}")

    ### Example Usage
    # python scripts/eval_hybrid_real.py -w exps/waypoint/pillow_base_arm/latest.pt -d exps/dense/pillow_base_arm_delta_allcams/latest.pt --num_episode 10 -e envs/cfgs/real_base_arm.yaml
    ###

    ### Example Usage
    # python scripts/eval_hybrid_real.py -w exps/waypoint/pillow_wbc/latest.pt -d exps/dense/pillow_wbc_delta_allcams/latest.pt --num_episode 10 -e envs/cfgs/real_wbc.yaml

    # python scripts/eval_hybrid_real.py -w exps/waypoint/remote_base_arm/latest.pt -d exps/dense/remote_base_arm_delta_allcams/latest.pt --num_episode 10 -e envs/cfgs/real_base_arm.yaml

    # python scripts/eval_hybrid_real.py -w exps/waypoint/remote_wbc/latest.pt -d exps/dense/remote_wbc_delta_allcams/latest.pt --num_episode 10 -e envs/cfgs/real_wbc.yaml

    # python scripts/eval_hybrid_real.py -w exps/waypoint/wiping_wbc/latest.pt -d exps/dense/wiping_wbc_delta_allcams/latest.pt --num_episode 10 -e envs/cfgs/real_wbc.yaml
    # python scripts/eval_hybrid_real.py -w exps/waypoint/wiping_wbc/latest.pt -d exps/dense/wiping_base_arm_delta_allcams/latest.pt --num_episode 20 -e envs/cfgs/real_base_arm.yaml
    ### 

