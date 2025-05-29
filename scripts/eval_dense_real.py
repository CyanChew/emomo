import argparse
import copy
import numpy as np
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
from scripts.train_dense import load_model
from common_utils.eval_utils import (
    check_for_interrupt,
)

# Global vars
env = None
cleanup_lock = threading.Lock()
cleanup_done = False

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

def eval_dense(
    dense_policy,
    dense_dataset,
    env,
    episode_idx,
    save_dir,
    record,
):
    assert not dense_policy.training

    env.reset()
    env.teleop_policy.reset()

    if record:
        assert save_dir is not None
        recorder = common_utils.Recorder(save_dir)
    cached_actions = []

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
            break

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
        env.step(action)

        if recorder is not None:
            recorder.add_numpy(obs, ["base1_image", "base2_image", "wrist_image"], color=(255, 140, 0))

    if recorder is not None:
        recorder.save(f"{episode_idx}", fps=10)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dense_model", type=str, required=True, help="dense model path")
    parser.add_argument("--num_episode", type=int, default=10)
    parser.add_argument("--num_pass", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default='rollouts')
    parser.add_argument("--record", type=int, default=1)
    parser.add_argument("-e", "--env_cfg", type=str, default="envs/cfgs/real_base_arm.yaml")
    args = parser.parse_args()

    try:
        env_cfg = pyrallis.load(RealEnvConfig, open(args.env_cfg, "r"))
        if env_cfg.wbc:
            from envs.real_env_wbc import RealEnv
        else:
            from envs.real_env_base_arm import RealEnv

        print(f">>>>>>>>>>{args.dense_model}<<<<<<<<<<")
        dense_policy, dense_dataset, _ = load_model(args.dense_model, "cuda", load_only_one=True)
        dense_policy.eval()

        if dense_policy.cfg.use_ddpm:
            common_utils.cprint(f"Warning: override to use ddim with step 10")
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
            eval_dense(
                dense_policy,
                dense_dataset,
                env,
                episode_idx,
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

    # python scripts/eval_dense_real.py -d exps/dense/pillow_base_arm_delta_allcams/latest.pt -e envs/cfgs/real_base_arm.yaml

    # python scripts/eval_dense_real.py -d exps/dense/pillow_wbc_delta_allcams/latest.pt -e envs/cfgs/real_wbc.yaml

    # python scripts/eval_dense_real.py -d exps/dense/remote_base_arm_delta_allcams/latest.pt -e envs/cfgs/real_base_arm.yaml

    # python scripts/eval_dense_real.py -d exps/dense/remote_wbc_delta_allcams/latest.pt -e envs/cfgs/real_wbc.yaml

    # python scripts/eval_dense_real.py -d exps/dense/wiping_wbc_delta_allcams/latest.pt -e envs/cfgs/real_wbc.yaml
    # python scripts/eval_dense_real.py -d exps/dense/wiping_base_arm_delta_allcams/latest.pt -e envs/cfgs/real_base_arm.yaml
    ###
