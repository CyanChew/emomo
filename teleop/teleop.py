import mujoco
import mujoco.viewer
import numpy as np
from teleop.policies import TeleopPolicy

policy = TeleopPolicy()
policy.reset()  # Wait for user to press "Start episode"

xml = """
<mujoco>
  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texrepeat="5 5"/>
  </asset>
  <worldbody>
    <light directional="true"/>
    <geom name="floor" size="0 0 .05" type="plane" material="groundplane"/>
    <body name="target" pos="0 0 .5" mocap="true">
      <geom type="box" size=".05 .05 .05" rgba=".6 .3 .3 .5"/>
    </body>
  </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mocap_id = m.body('target').mocapid[0]
with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as viewer:
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    while viewer.is_running():
        mujoco.mj_step(m, d)
        obs = {
            'base_pose': np.zeros(3),
            'arm_pos': d.mocap_pos[mocap_id],
            'arm_quat': d.mocap_quat[mocap_id][[1, 2, 3, 0]],  # (w, x, y, z) -> (x, y, z, w)
            'gripper_pos': np.zeros(1),
        }
        action = policy.step(obs)
        if action == 'reset_env':
            break
        if isinstance(action, dict):
            d.mocap_pos[mocap_id] = action['arm_pos'][[1, 0, 2]] * [-1, 1, 1]
            d.mocap_quat[mocap_id] = action['arm_quat'][[3, 1, 0, 2]] * [1, -1, 1, 1]
        viewer.sync()
