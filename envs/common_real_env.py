from constants import BASE_RPC_HOST, BASE_RPC_PORT, ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from constants import BASE_CAMERA_SERIAL, POLICY_CONTROL_PERIOD
import time
from envs.utils.cameras import KinovaCamera, LogitechCamera, RealSenseCamera
from envs.utils.arm_server import ArmManager
from envs.utils.base_server import BaseManager
from itertools import count
from interactive_scripts.dataset_recorder import ActMode, DatasetRecorder
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from teleop.policies import TeleopPolicy
from colorama import Fore, Style
import pyrallis
import pickle
from common_utils import Stopwatch
import os
import common_utils
import numpy as np
from envs.common_real_env_cfg import CameraConfig, RealEnvConfig


class CommonRealEnv:
    def __init__(self, cfg: RealEnvConfig):
        # RPC server connection for base
        base_manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            base_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception('Could not connect to base RPC server, is base_server.py running?') from e

        # RPC server connection for arm
        arm_manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            arm_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception('Could not connect to arm RPC server, is arm_server.py running?') from e

        # RPC proxy objects
        self.base = base_manager.Base(max_vel=(0.5, 0.5, 1.57), max_accel=(0.5, 0.5, 1.57))
        self.arm = arm_manager.Arm()

        self.cfg = cfg

        # Camera name 
        self.cameras = {}

        for cam_cfg in self.cfg.cameras:
            cam_name = cam_cfg.name
            cam_type = cam_cfg.type.lower()
            if cam_type == "kinova":
                cam = KinovaCamera()
            elif cam_type == "realsense":
                assert cam_cfg.serial is not None, f"Camera '{cam_name}' of type 'realsense' must specify a serial"
                cam = RealSenseCamera(cam_cfg.serial)
            elif cam_type == "logitech":
                assert cam_cfg.serial is not None, f"Camera '{cam_name}' of type 'logitech' must specify a serial"
                cam = LogitechCamera(cam_cfg.serial)
            else:
                raise ValueError(f"Unknown camera type: {cam_type}")

            self.cameras[cam_name] = cam

        self.data_folder = cfg.data_folder
        self.recorder = DatasetRecorder(self.data_folder)
        self.teleop_policy = None

        self.cached_intrinsics = dict()
        self.cached_extrinsics = dict()
        self.stopwatch = Stopwatch()

    def get_state_obs(self):
        obs = {}
        obs.update(self.base.get_state())
        obs.update(self.arm.get_state())
        return obs

    def get_cam_obs(self):
        obs = {}
        for name, cam in self.cameras.items():
            if isinstance(cam, RealSenseCamera):
                rgb_image, depth_image = cam.get_rgbd()
                obs[f"{name}_image"] = rgb_image
                obs[f"{name}_depth"] = depth_image
            else:
                obs[f"{name}_image"] = cam.get_image()
        return obs

    def get_obs(self):
        obs = self.get_cam_obs()
        obs.update(self.get_state_obs())
        return obs

    def get_intrinsics(self, cam_name: str):
        cam = self.cameras[cam_name]
        if isinstance(cam, RealSenseCamera):
            if cam_name in self.cached_intrinsics:
                return self.cached_intrinsics[cam_name]
    
            intrinsics = cam.get_intrinsics()
            self.cached_intrinsics[cam_name] = intrinsics
    
            # Save to pickle file if calibration dir is set
            if self.cfg.calib_dir:
                os.makedirs(self.cfg.calib_dir, exist_ok=True)
                intr_path = os.path.join(self.cfg.calib_dir, f"{cam_name}_intrinsics.pkl")
                with open(intr_path, "wb") as f:
                    pickle.dump(intrinsics, f)
                print(f"[INFO] Saved intrinsics for {cam_name} to {intr_path}")
    
            return intrinsics
        else:
            raise ValueError(f"Camera '{cam_name}' does not support intrinsics")

    def get_extrinsics(self, cam_name: str):
        cam = self.cameras[cam_name]
        if isinstance(cam, RealSenseCamera):
            if cam_name in self.cached_extrinsics:
                return self.cached_extrinsics[cam_name]
    
            if not self.cfg.calib_dir:
                raise RuntimeError("No calibration_files path set in config")
    
            extr_path = os.path.join(self.cfg.calib_dir, f"{cam_name}_extrinsics.pkl")
            if not os.path.exists(extr_path):
                raise FileNotFoundError(f"Extrinsics for {cam_name} not found at {extr_path}")
    
            with open(extr_path, "rb") as f:
                extrinsics = pickle.load(f)
            self.cached_extrinsics[cam_name] = extrinsics
            return extrinsics
        else:
            raise ValueError(f"Camera '{cam_name}' does not support extrinsics")

    def reset(self):
        print('Resetting base...')
        self.base.reset()

        print('Resetting arm...')
        self.arm.reset()

        print('Robot has been reset')

    def step_arm_only(self, action):
        raise('Subclasses need to implement this')

    def step_base_only(self, action):
        self.base.execute_action(action)  # Non-blocking

    def step(self, action):
        raise('Subclasses need to implement this')

    def close(self):
        self.base.close()
        self.arm.close()
        for cam in self.cameras.values():
            cam.close()

    def _dump_or_check_env_cfg(self):
        cfg_path = os.path.join(self.data_folder, "env_cfg.yaml")
        if not os.path.exists(cfg_path):
            print(f"saving env cfg to {cfg_path}")
            pyrallis.dump(self.cfg, open(cfg_path, "w"))  # type: ignore
        else:
            assert common_utils.check_cfg(RealEnvConfig, cfg_path, self.cfg), \
                f"Error: {self.data_folder} contains a different config than the current one"

    def drive_to_reset(self):
        if self.teleop_policy is None:
            self.teleop_policy = TeleopPolicy()

        self.reset() # Reset arm
        print(Fore.MAGENTA + "Arm was reset..." + Style.RESET_ALL)
        print(Fore.MAGENTA + "Drive base to reset pose..." + Style.RESET_ALL)
        self.teleop_policy.reset()

        start_time = time.time()

        for step_idx in count():
            # Enforce desired control freq
            step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
            while time.time() < step_end_time:
                time.sleep(0.0001)

            # Get latest observation
            with self.stopwatch.time('get_obs'):
                obs = self.get_obs()

            lat_ms = self.stopwatch.times['get_obs'][-1]

            with self.stopwatch.time('teleop_policy.step'):
                action = self.teleop_policy.step(obs)

            lat_ms += self.stopwatch.times['teleop_policy.step'][-1]

            if lat_ms > 100.0:
                print(Fore.RED + f"[ERROR] teleop_policy.step + get_obs exceeded 100ms ({lat_ms:.1f} ms). Exiting episode." + Style.RESET_ALL)
                self.stopwatch.summary()
                #return USED TO BE THIS

            # No action if teleop not enabled
            if action is None:
                continue

            # Execute valid action on robot
            if isinstance(action, dict) and 'base_pose' in action:
                self.step_base_only(action)
            elif action == 'end_episode':
                print(Fore.MAGENTA + "Base driven back, hit Reset to collect next episode" + Style.RESET_ALL)
            elif action == 'reset_env':
                print(Fore.MAGENTA + "Continuing to collect next episode..." + Style.RESET_ALL)
                break

    def collect_episode(self):
        self._dump_or_check_env_cfg()
        # Reset
        if self.teleop_policy is None:
            self.teleop_policy = TeleopPolicy()

        print(Fore.GREEN + "*********************" + Style.RESET_ALL)
        print(Fore.GREEN + "Recording episode: %d" % self.recorder.episode_idx + Style.RESET_ALL)
        self.reset()
        self.teleop_policy.reset()

        episode_ended = False
        start_time = time.time()

        prev_obs = self.get_obs()  # Only capture observations at 10Hz

        for step_idx in count():
            # Enforce desired control freq
            step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
            while time.time() < step_end_time:
                time.sleep(0.0001)

            # Get latest observation
            with self.stopwatch.time('get_obs'):
                obs = self.get_obs()

            lat_ms = self.stopwatch.times['get_obs'][-1]
            with self.stopwatch.time('teleop_policy.step'):
                action = self.teleop_policy.step(obs)

            lat_ms += self.stopwatch.times['teleop_policy.step'][-1]
            if lat_ms > 90.0:
                print(Fore.RED + f"[ERROR] teleop_policy.step + get_obs exceeded 90ms ({lat_ms:.1f} ms). Exiting episode." + Style.RESET_ALL)
                self.stopwatch.summary()
                return

            # No action if teleop not enabled
            if action is None:
                prev_obs = obs
                continue

            # Execute valid action on robot
            if isinstance(action, dict):
                self.step(action)

                with self.stopwatch.time('process_action'):
                    action_quat = action["arm_quat"]
                    if action_quat[3] < 0.0:
                        np.negative(action_quat, out=action_quat)

                    record_action_arm_pos = action["arm_pos"]
                    record_action = np.concatenate(
                        [
                            record_action_arm_pos,
                            action_quat,
                            action["gripper_pos"],
                            action["base_pose"]
                        ]
                    )

                with self.stopwatch.time('process_delta_action'):
                    delta_pos = record_action[:3] - prev_obs['arm_pos']
                    delta_rot = R.from_quat(action["arm_quat"]) * R.from_quat(prev_obs["arm_quat"]).inv()
                    delta_quat = delta_rot.as_quat()
                    delta_base_pose = record_action[-3:] - prev_obs["base_pose"]
                    # Delta action to record
                    record_delta_action = np.concatenate(
                        [delta_pos, delta_quat, action["gripper_pos"], delta_base_pose]
                    )

                if not episode_ended:
                    with self.stopwatch.time('record action'):
                        # Record executed action
                        self.recorder.record(
                            ActMode.Dense,
                            obs,
                            action=record_action,
                            delta_action=record_delta_action,
                            teleop_mode=action["teleop_mode"]
                        )

            # Episode ended
            elif not episode_ended and action == 'end_episode':
                episode_ended = True
                print('Episode ended')
                if not (input('Save episode?') in ['n', 'N']):
                    print('Saving')
                    self.recorder.end_episode(save=True)
                else:
                    print('Not saving')
                print('Teleop is now active. Press "Reset env" in the web app when ready to proceed.')

            # Ready for env reset
            elif action == 'reset_env':
                break

            prev_obs = obs

        self.stopwatch.summary()

    def set_gripper(self, width):
        obs = self.get_state_obs()
        arm_pos, arm_quat, base_pose = obs["arm_pos"], obs["arm_quat"], obs["base_pose"]
        # Execute action
        action = {
            "arm_pos": arm_pos, \
            "arm_quat": arm_quat, \
            "gripper_pos": np.array([width]), \
            "base_pose": base_pose if not self.cfg.wbc else np.zeros(3)
            }
        self.step(action)
        time.sleep(POLICY_CONTROL_PERIOD)  # Maintain control rate

    def move_to_arm_waypoint(self, target_arm_pos, target_arm_quat, target_gripper_pos, step_size=0.065, threshold_pos=0.01, threshold_quat=0.01, phone_interventions=False, recorder=None):
        """
        Moves the robot arm towards a target position and orientation using interpolation.
    
        Args:
            target_arm_pos (array-like): [x, y, z] target for the arm end-effector.
            target_arm_quat (array-like): [x, y, z, w] target quaternion for arm orientation.
            target_gripper_pos (float): Target gripper position (0.0 closed, 1.0 open).
            step_size (float): Maximum step size per iteration.
            threshold_pos (float): Position error threshold for stopping.
            threshold_quat (float): Quaternion error threshold for stopping.
    
        Returns:
            bool: True if the target is reached.
        """
        if phone_interventions:
            assert(self.teleop_policy is not None)

        # Ensure consistent quaternion sign
        if target_arm_quat[3] < 0:
            np.negative(target_arm_quat, out=target_arm_quat)
    
        reached = False
        pos_error_norm = np.inf
        MAX_STEP = 25
        step = 0
        interrupt = False

        while not reached:
            # Get current position and orientation
            if recorder is not None:
                obs = self.get_obs()
                recorder.add_numpy(obs, ["base1_image", "base2_image", "wrist_image"])
            else:
                obs = self.get_state_obs()

            if phone_interventions:
                with self.stopwatch.time('teleop_policy.step'):
                    teleop_intervention = self.teleop_policy.step(obs)
                if teleop_intervention == 'end_episode':
                    interrupt = True
                    break

            curr_arm_pos, curr_arm_quat, curr_base_pose = obs["arm_pos"], obs["arm_quat"], obs["base_pose"]
    
            # Compute position error
            pos_error = target_arm_pos - curr_arm_pos
            pos_error_norm = np.linalg.norm(pos_error)
    
            # Compute quaternion error
            quat_error = 1 - abs(np.dot(curr_arm_quat, target_arm_quat))

            print(f"[Step {step}] pos_err: {pos_error_norm:.4f}, quat_err: {quat_error:.4f}")

            if pos_error_norm < threshold_pos and quat_error < threshold_quat:
                reached = True
                break

            elif step > MAX_STEP:
                break

            # Compute interpolated position step
            step_vec = step_size * pos_error / (pos_error_norm + 1e-6)  # Avoid division by zero
            next_pos = curr_arm_pos + step_vec if pos_error_norm > step_size else target_arm_pos
    
            # Compute interpolated quaternion step using Slerp
            key_times = [0, 1]  # Define key times
            key_rots = R.from_quat([curr_arm_quat, target_arm_quat])  # Define key rotations
            slerp = Slerp(key_times, key_rots)  # Create Slerp object
            interp_ratio = min(step_size / (pos_error_norm + 1e-6), 1.0)  # Normalize step size
            next_quat = slerp([interp_ratio]).as_quat()[0]  # Interpolated quaternion
    
            # Execute action
            action = {
                "arm_pos": next_pos, \
                "arm_quat": next_quat, \
                "gripper_pos": 1 if obs['gripper_pos'] > 0.3 else obs['gripper_pos'], \
                "base_pose": curr_base_pose if not self.cfg.wbc else np.zeros(3)
                }
            self.step(action)

            time.sleep(POLICY_CONTROL_PERIOD)  # Maintain control rate
            step += 1

        ## FIXME: Move the gripper, after reaching a waypoint
        #for _ in range(10):  # Hack: Execute gripper action for 10 timesteps
        #    self.step({"gripper_pos": target_gripper_pos})
        #    time.sleep(POLICY_CONTROL_PERIOD)
    
        return reached, pos_error_norm, interrupt

    def move_to_base_waypoint(self, target_base_pose, threshold_pos=0.01, threshold_theta=0.01, phone_interventions=False, recorder=None):
        """
        Smoothly moves the robot base to a target [x, y, theta] pose via interpolation.
    
        Args:
            target_base_pose (array-like): [x, y, theta] target for the base.
            threshold_pos (float): Position error threshold for stopping.
            threshold_theta (float): Rotation error threshold (in radians) for stopping.
    
        Returns:
            bool: True if the target is reached.
        """
        interrupt = False
        if phone_interventions:
            assert(self.teleop_policy is not None)

        obs = self.get_state_obs()
        curr_base_pose = np.array(obs["base_pose"])
        MAX_STEP = 100
        ALPHA = 0.1  # interpolation factor (0 < ALPHA <= 1)
        step = 0
    
        while True:
            if recorder is not None:
                obs = self.get_obs()
                recorder.add_numpy(obs, ["base1_image", "base2_image", "wrist_image"])
            else:
                obs = self.get_state_obs()

            if phone_interventions:
                with self.stopwatch.time('teleop_policy.step'):
                    teleop_intervention = self.teleop_policy.step(obs)
                if teleop_intervention == 'end_episode':
                    interrupt = True
                    break

            curr_base_pose = np.array(obs["base_pose"])
    
            # Compute errors
            pos_error = target_base_pose[:2] - curr_base_pose[:2]
            theta_error = target_base_pose[2] - curr_base_pose[2]
            pos_error_norm = np.linalg.norm(pos_error)
    
            print(f"[Step {step}] pos_err: {pos_error_norm:.4f}, theta_err: {theta_error:.4f}")
    
            if pos_error_norm < threshold_pos and abs(theta_error) < threshold_theta:
                return True, pos_error_norm, interrupt
            elif step > MAX_STEP:
                break
    
            # Interpolate linearly toward target
            next_pose = curr_base_pose.copy()
            next_pose[:2] += ALPHA * pos_error
            next_pose[2] += ALPHA * theta_error
    
            # Execute base-only action
            #print(next_pose)
            self.step_base_only({"base_pose": next_pose})
    
            time.sleep(POLICY_CONTROL_PERIOD)
            step += 1
    
        return False, pos_error_norm, interrupt

