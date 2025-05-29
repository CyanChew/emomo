from envs.common_real_env import CommonRealEnv, RealEnvConfig

class RealEnv(CommonRealEnv):
    def __init__(self, cfg: RealEnvConfig):
        super().__init__(cfg)
        assert(not self.cfg.wbc)

    def step_arm_only(self, action):
        self.arm.execute_action(action)   # Non-blocking

    def step(self, action):
        self.base.execute_action(action)  # Non-blocking
        self.arm.execute_action(action)   # Non-blocking

if __name__ == '__main__':
    import argparse
    import pyrallis
    import time
    from common_utils import Stopwatch
    from constants import POLICY_CONTROL_PERIOD

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/real_base_arm.yaml")

    args = parser.parse_args()
    env_cfg = pyrallis.load(RealEnvConfig, open(args.env_cfg, "r"))

    # Test open close env
    #try:
    #    env = RealEnv(env_cfg)
    #    env.reset()
    #finally:
    #    env.close()
    #    print('Successfully closed env.')

    ## Test get obs
    try:
        env = RealEnv(env_cfg)
        env.reset()
        stopwatch = Stopwatch()
        while True:
            with stopwatch.time('get_obs'):
                obs = env.get_obs()
            lat_ms = stopwatch.times['get_obs'][-1]
            bar_len = int(min(lat_ms / 2, 50))  # Scale: 2ms = 1 char, max 50 chars
            bar = '*' * bar_len
            print(f"[get_obs] {lat_ms:6.1f} ms | {bar}")
            time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    finally:
        env.close()
        print('Successfully closed env.')

    ## Test collect episode
    #try:
    #    env = RealEnv(env_cfg)
    #    env.collect_episode()
    #finally:
    #    env.close()
    #    print('Successfully closed env.')

    # Test replay episode
    #try:
    #    env = RealEnv(env_cfg)
    #    env.replay_episode('dev1/demo00000.pkl', replay_mode="absolute")
    #finally:
    #    env.close()
    #    print('Successfully closed env.')
