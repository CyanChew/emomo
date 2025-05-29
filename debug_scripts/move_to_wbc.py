import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from envs.common_real_env import RealEnvConfig
from envs.real_env_wbc import RealEnv
import pyrallis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/real_wbc.yaml")
    args = parser.parse_args()
    env_cfg = pyrallis.load(RealEnvConfig, open(args.env_cfg, "r"))

    try:
        env = RealEnv(env_cfg)
        env.reset()

        N = 1

        obs = env.get_state_obs()
        reference_quat = env.get_state_obs()["arm_quat"]
        reference_pos = np.array([0.5, 0.0, 0.3])

        np.random.seed(42)
        euler = R.from_quat(reference_quat).as_euler('xyz')
        euler_offsets = np.random.uniform(low=-0.1, high=0.1, size=(N, 3))  # ~17 deg perturbations
        position_offsets = np.random.uniform(low=-0.05, high=0.05, size=(N, 3))

        for i in range(N):
             target_pos = reference_pos + position_offsets[i]
             target_euler = euler + euler_offsets[i]
             target_quat = R.from_euler('xyz', target_euler).as_quat()
             if target_quat[3] < 0:
                 np.negative(target_quat, out=target_quat)

             print(f"\nMoving to pose {i+1}/{N}: pos={target_pos.round(3)}, quat={target_quat.round(3)}")

             reached, error = env.move_to_arm_waypoint(target_pos, target_quat, target_gripper_pos=0.0)
             #print(reached, error)
             #time.sleep(0.3)

        env.reset()

    finally:
        print("Closing env...")
        env.close()
        print("Closed env.")
