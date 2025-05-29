import argparse
import copy
import numpy as np
import pyrallis
import signal
import sys
import threading

from envs.common_real_env_cfg import RealEnvConfig
from teleop.policies import TeleopPolicy
from scripts.train_waypoint import load_waypoint
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

def eval_waypoint(
    policy: WaypointTransformer,
    env,
    episode_idx,
    num_pass: int,
    save_dir,
    record: bool,
):
    assert not policy.training
    env.reset()
    env.teleop_policy.reset()

    recorder = None
    if record:
        assert save_dir is not None
        recorder = common_utils.Recorder(save_dir)


    curr_mode = ActMode.ArmWaypoint.value if env.cfg.wbc else ActMode.BaseWaypoint.value
    start_time = time.time()

    for step_idx in count():
        loop_start = time.time()
        if teleop_intervention == 'end_episode':
            break

        obs = env.get_obs()
        if recorder is not None:
            recorder.add_numpy(obs, ["base1_image", "wrist_image"], color=(255, 140, 0))

        points, colors = pcl_from_obs(obs, env.cfg)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        proprio = np.hstack((obs["arm_pos"], obs["arm_quat"], obs["gripper_pos"], obs["base_pose"]))

        with torch.no_grad():
            _, pos_cmd, rot_cmd, gripper_cmd, mode = policy.inference(
                torch.from_numpy(points).float(),
                torch.from_numpy(colors).float(),
                torch.from_numpy(proprio).float(),
                num_pass=num_pass,
            )
            if policy.cfg.use_euler:
                quat_cmd = R.from_euler('xyz', rot_cmd).as_quat()
            else:
                quat_cmd = rot_cmd

        if curr_mode == ActMode.ArmWaypoint.value:
            if quat_cmd[3] < 0:
                np.negative(quat_cmd, out=quat_cmd)
            reached, err, _ = env.move_to_arm_waypoint(pos_cmd, quat_cmd, gripper_cmd, phone_interventions=True, recorder=recorder)

        elif curr_mode == ActMode.BaseWaypoint.value:
            base_pose = np.array([pos_cmd[0], pos_cmd[1], R.from_quat(quat_cmd).as_euler('xyz')[2]])
            reached, err, _ = env.move_to_base_waypoint(base_pose, phone_interventions=True, recorder=recorder)

        else:
            break

        curr_mode = mode

        elapsed = time.time() - loop_start
        sleep_time = POLICY_CONTROL_PERIOD - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    if recorder is not None:
        recorder.save(f"s{episode_idx}", fps=10)
    return

def _eval_waypoint_multi_episode(
    policy: WaypointTransformer,
    num_pass: int,
    env_cfg: MujocoEnvConfig,
    seed: int,
    num_episode: int,
    save_dir,
    record: int,
):
    scores = []
    num_steps = []
    for seed in range(seed, seed + num_episode):
        score, num_step = eval_waypoint(
            policy,
            env_cfg,
            seed=seed,
            num_pass=num_pass,
            save_dir=save_dir,
            record=record,
        )
        scores.append(score)
        num_steps.append(num_step)
    return np.mean(scores), np.mean(num_steps)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model path")
    parser.add_argument("--seed", type=int, default=99999)
    parser.add_argument("--num_episode", type=int, default=10)
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--num_pass", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default='rollouts')
    parser.add_argument("--record", type=int, default=1)
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/cube.yaml")
    parser.add_argument("--headless", action='store_true')
    args = parser.parse_args()

    try:
        env_cfg = pyrallis.load(RealEnvConfig, open(args.env_cfg, "r"))
        if env_cfg.wbc:
            from envs.real_env_wbc import RealEnv
        else:
            from envs.real_env_base_arm import RealEnv

        print(f">>>>>>>>>>{args.model}<<<<<<<<<<")
        policy = load_waypoint(args.model, device='cuda')
        policy.train(False)
        policy = policy.cuda()

        env = RealEnv(env_cfg)

        if env.teleop_policy is None: 
            env.teleop_policy = TeleopPolicy()

        if os.path.exists('rollouts'):
            N = len([fn for fn in os.listdir('rollouts') if 'mp4' in fn])

        for episode_idx in range(N, args.num_episode + N):
            env.drive_to_reset()
            eval_waypoint(
                policy,
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
    # python scripts/eval_waypoint_real.py -d exps/waypoint/pillow_base_arm/latest.pt -e envs/cfgs/real_base_arm.yaml
    ###
