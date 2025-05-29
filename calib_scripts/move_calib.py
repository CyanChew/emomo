import argparse, pickle, os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from envs.real_env_base_arm import RealEnv
from envs.common_real_env import RealEnvConfig
import pyrallis

"""
This script collects RGB-D observations and arm poses from two fixed cameras
("base1" and "base2") to support hand-eye calibration using an ArUco marker.

Instructions:
1. Print the marker PDF file `aruco_marker_id0_50mm.pdf`, which is generated using
   the `generate_calib_marker_pdf.py` script. Paste the printed marker onto a small
   piece of cardboard and place it securely inside the robot gripper.
2. Run this script. It will prompt you when it is time to close the gripper.
3. After confirming, the gripper will close around the marker, and the robot will move
   to a series of randomly sampled poses around a reference position.
4. At each pose, the script records:
   - RGB and depth images from both cameras
   - Arm end-effector position and orientation
   - Camera intrinsics for each camera

The collected data is saved to `cam_to_robot_data.pkl`.

This file is used as input to `solve_calib.py`, which estimates the camera-to-robot
extrinsic transform by aligning the observed marker pose in each camera to the known
end-effector pose.
"""

def collect_data_for_camera(env, cam_name, N=15):
    intrinsics = env.get_intrinsics(cam_name)
    if cam_name == 'base2':
        reference_pos = np.array([0.45, 0.0, 0.3])
    else:
        reference_pos = np.array([0.48, 0.0, -0.05])

    reference_quat = env.get_state_obs()["arm_quat"]
    euler = R.from_quat(reference_quat).as_euler('xyz')

    np.random.seed(1)
    position_offsets = np.random.uniform(low=-0.05, high=0.05, size=(N, 3))
    euler_offsets = np.random.uniform(low=-0.1, high=0.1, size=(N, 3))

    samples = []
    for i in range(N):
        print(f"[{cam_name}] [{i+1}/{N}] Moving to random pose...")
        target_pos = reference_pos + position_offsets[i]
        target_euler = euler + euler_offsets[i]
        target_quat = R.from_euler('xyz', target_euler).as_quat()
        if target_quat[3] < 0:
            np.negative(target_quat, out=target_quat)

        reached, err, _ = env.move_to_arm_waypoint(target_pos, target_quat, target_gripper_pos=0.0)
        time.sleep(0.3)
        print(reached, err)
        obs = env.get_obs()

        sample = {
            f"{cam_name}_image": obs[f"{cam_name}_image"],
            f"{cam_name}_depth": obs[f"{cam_name}_depth"],
            "arm_pos": obs["arm_pos"],
            "arm_quat": obs["arm_quat"]
        }
        samples.append(sample)

    return intrinsics, samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/real_base_arm.yaml")
    parser.add_argument("--out_path", type=str, default="calib_scripts/cam_to_robot_data.pkl")
    args = parser.parse_args()

    env_cfg = pyrallis.load(RealEnvConfig, open(args.env_cfg, "r"))
    env = RealEnv(env_cfg)

    try:
        env.reset()
        input('Ready to close gripper?')
        env.set_gripper(0.25)
        input('Ready to start motion?')

        all_intrinsics = {}
        all_samples = []

        for cam_name in ["base1", "base2"]:
            intrinsics, samples = collect_data_for_camera(env, cam_name)
            all_intrinsics[cam_name] = intrinsics
            all_samples.extend(samples)

        data_to_save = {
            "cam_names": ["base1", "base2"],
            "intrinsics": all_intrinsics,
            "samples": all_samples,
        }

        os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
        with open(args.out_path, "wb") as f:
            pickle.dump(data_to_save, f)

        print(f"Saved {len(all_samples)} samples and intrinsics to {args.out_path}")
        env.reset()

    finally:
        env.close()
        print("Closed env.")
