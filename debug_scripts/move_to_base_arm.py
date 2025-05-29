import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from envs.common_real_env import RealEnvConfig
from teleop.policies import TeleopPolicy
from envs.real_env_base_arm import RealEnv
import pyrallis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/real_base_arm.yaml")
    args = parser.parse_args()
    env_cfg = pyrallis.load(RealEnvConfig, open(args.env_cfg, "r"))

    try:
        env = RealEnv(env_cfg)
        if env.teleop_policy is None: 
            env.teleop_policy = TeleopPolicy()
            print(env.teleop_policy.teleop_state)
            env.teleop_policy.reset()
            print(env.teleop_policy.teleop_state)

        env.reset()

        obs = env.get_state_obs()

        reference_pos = obs["arm_pos"] + [0.2, 0.0, 0.0]
        reference_quat = obs["arm_quat"]
        print(reference_pos, reference_quat)

        reached, error, _ = env.move_to_arm_waypoint(reference_pos, reference_quat, 1.0, phone_interventions=True)

        #base_pose = obs['base_pose']
        #new_base_pose = [base_pose[0] + 0.2,  base_pose[1] + 0.0, base_pose[2] + np.pi/6]
        #new_base_pose = [base_pose[0] + 0.2,  base_pose[1] + 0.0, base_pose[2] + np.pi/4]
        #new_base_pose = [base_pose[0] + 0.3,  base_pose[1] + 0.0, base_pose[2]]
        #reached, error = env.move_to_base_waypoint(new_base_pose, phone_interventions=True)

        print(reached, error)

    finally:
        print("Closing env...")
        env.close()
        print("Closed env.")
