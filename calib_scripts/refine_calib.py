import os
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import open3d as o3d
import numpy as np
import argparse
import pickle
import pyrallis
from envs.common_real_env_cfg import RealEnvConfig
from envs.real_env_base_arm import RealEnv
from envs.utils.camera_utils import pcl_from_obs

"""
This script refines the extrinsic transform of the secondary camera ('base2')
*relative to* the primary camera ('base1') to improve point cloud alignment.

Usage:
1. First, run `solve_calib` to compute initial camera-to-robot transforms.
2. Place an object with regular, easily matchable geometry (e.g., a box or other object
   with flat, visible surfaces) within the shared view of both cameras.
3. Run this script to:
   - Capture RGB-D observations from both cameras.
   - Convert them to point clouds in the robot frame.
   - Run ICP to align `base2`'s point cloud to `base1`'s.
   - Compute a refined transform and save it.

The goal is to visually stitch the two point clouds into a single, coherent scene
without gaps or misalignments, improving spatial consistency for downstream tasks
(e.g., 3D reconstruction or manipulation).
"""

def transform_pcd(pcd, transform):
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    new_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    new_pcd.transform(transform)
    return new_pcd

def run_icp(source_pcd, target_pcd, threshold=0.04):
    print("Source points:", np.asarray(source_pcd.points).shape)
    print("Target points:", np.asarray(target_pcd.points).shape)
    
    print("Source centroid:", np.mean(np.asarray(source_pcd.points), axis=0))
    print("Target centroid:", np.mean(np.asarray(target_pcd.points), axis=0))

    icp_result = o3d.pipelines.registration.registration_icp(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=threshold,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    print("[ICP] Fitness:", icp_result.fitness)
    print("[ICP] RMSE:", icp_result.inlier_rmse)
    return icp_result.transformation

def save_extrinsics(path, extrinsics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(extrinsics, f)
    print(f"[INFO] Saved refined extrinsics to {path}")

def main(env_cfg_path):
    with open(env_cfg_path, "r") as f:
        cfg = pyrallis.load(RealEnvConfig, f)
    env = RealEnv(cfg)
    env.reset()

    cam_names = ["base1", "base2"]
    intrinsics = {name: env.get_intrinsics(name) for name in cam_names}
    extrinsics = {name: env.get_extrinsics(name) for name in cam_names}

    print("[INFO] Capturing point clouds...")
    obs = env.get_obs()
    base1_points, base1_colors = pcl_from_obs(obs, cfg, cam_names=["base1"])
    base2_points, base2_colors = pcl_from_obs(obs, cfg, cam_names=["base2"])

    if base1_points is None or base2_points is None:
        print("[ERROR] One of the point clouds is empty. Make sure both cameras are streaming.")
        return

    # Build Open3D point clouds
    pcd_base1_robot = o3d.geometry.PointCloud()
    pcd_base1_robot.points = o3d.utility.Vector3dVector(base1_points)
    pcd_base1_robot.colors = o3d.utility.Vector3dVector(base1_colors)

    pcd_base2_robot = o3d.geometry.PointCloud()
    pcd_base2_robot.points = o3d.utility.Vector3dVector(base2_points)
    pcd_base2_robot.colors = o3d.utility.Vector3dVector(base2_colors)

    print("[INFO] Running ICP to align base2 to base1 (in robot frame)...")
    T_icp = run_icp(pcd_base2_robot, pcd_base1_robot)

    refined_T_base2 = T_icp @ extrinsics["base2"]

    # Save refined extrinsics
    base2_default_path = os.path.join(cfg.calib_dir, "base2_extrinsics.pkl")
    base2_backup_path = os.path.join(cfg.calib_dir, "base2_extrinsics_orig.pkl")
    
    # Backup the original extrinsics if not already backed up
    if not os.path.exists(base2_backup_path):
        print(f"[INFO] Backing up original extrinsics to {base2_backup_path}")
        with open(base2_default_path, "rb") as f_in, open(base2_backup_path, "wb") as f_out:
            f_out.write(f_in.read())
    else:
        print(f"[INFO] Backup already exists at {base2_backup_path}")
    
    # Overwrite base2_extrinsics.pkl with the refined result
    save_extrinsics(base2_default_path, refined_T_base2)

    # Visualize
    pcd_base2_aligned = transform_pcd(pcd_base2_robot, T_icp)
    o3d.visualization.draw_geometries([
        pcd_base1_robot.paint_uniform_color([0, 1, 0]),
        pcd_base2_aligned.paint_uniform_color([1, 0, 0]),
    ])

    env.close()
    print('Closed env.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env_cfg", type=str, default="envs/cfgs/real_base_arm.yaml")
    args = parser.parse_args()
    main(args.env_cfg)

