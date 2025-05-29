import pyrallis
import argparse
from envs.utils.camera_utils import pcl_from_obs
from envs.common_mj_env import MujocoEnvConfig
import open3d as o3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/open_wbc.yaml")
    args = parser.parse_args()
    env_cfg = pyrallis.load(MujocoEnvConfig, open(args.env_cfg, "r"))

    if env_cfg.wbc:
        from envs.mj_env_wbc import MujocoEnv
    else:
        from envs.mj_env_base_arm import MujocoEnv
    
    env = MujocoEnv(env_cfg)
    env.reset()

    obs = env.get_obs()
    merged_points, merged_colors = pcl_from_obs(obs, env_cfg)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(merged_points)
    point_cloud.colors = o3d.utility.Vector3dVector(merged_colors)
    o3d.visualization.draw_geometries([point_cloud], window_name="Point Cloud Viewer")
