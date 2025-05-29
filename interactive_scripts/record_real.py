import pyrallis
import argparse
import signal
import sys
import threading
from envs.common_real_env_cfg import RealEnvConfig

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env_cfg", type=str, default="envs/cfgs/real_base_arm.yaml")
    args = parser.parse_args()
    print('env cfg', args.env_cfg)

    try:
        # Load config
        with open(args.env_cfg, "r") as f:
            env_cfg = pyrallis.load(RealEnvConfig, f)

        # Dynamically import correct RealEnv
        if env_cfg.wbc:
            from envs.real_env_wbc import RealEnv
            print("Using WBC env.")
        else:
            from envs.real_env_base_arm import RealEnv
            print("Using Base+Arm env.")

        # Create env
        env = RealEnv(env_cfg)

        # Run episodes
        for i in range(10):
            env.collect_episode()
            env.drive_to_reset()

    except Exception as e:
        print(f"[Error] Unhandled exception: {e}")
    finally:
        if not cleanup_done:
            try:
                if env is not None:
                    env.close()
                    print("Closed env.")
            except Exception as e:
                print(f"[Error] Cleanup failed in finally block: {e}")

