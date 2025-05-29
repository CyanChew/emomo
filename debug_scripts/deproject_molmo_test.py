import pyrallis
import numpy as np
import argparse
from envs.utils.camera_utils import pcl_from_obs, deproject_pixel_to_3d
from envs.common_mj_env import MujocoEnvConfig
import open3d as o3d
import pygame
import cv2

def select_pixel_manually(img_rgb):
    pygame.init()
    height, width, _ = img_rgb.shape
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Click a pixel (hover to see coords)")

    font = pygame.font.SysFont("Arial", 20)
    surface = pygame.surfarray.make_surface(np.transpose(img_rgb, (1, 0, 2)))  # (W, H, C)

    clicked_pixel = None
    running = True

    while running:
        screen.blit(surface, (0, 0))
        mouse_x, mouse_y = pygame.mouse.get_pos()

        # Draw red crosshair
        pygame.draw.line(screen, (255, 0, 0), (mouse_x, 0), (mouse_x, height), 1)
        pygame.draw.line(screen, (255, 0, 0), (0, mouse_y), (width, mouse_y), 1)

        # Draw coordinates in top-left
        coord_text = font.render(f"({mouse_x}, {mouse_y})", True, (255, 0, 0))
        screen.blit(coord_text, (10, 10))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    clicked_pixel = (mouse_x, mouse_y)
                    print(f"Clicked pixel: ({mouse_x}, {mouse_y})")
                    running = False
                    break

    pygame.quit()
    return clicked_pixel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--env_cfg", type=str, default="envs/cfgs/open_wbc.yaml")
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/cube_wbc.yaml")
    args = parser.parse_args()
    env_cfg = pyrallis.load(MujocoEnvConfig, open(args.env_cfg, "r"))

    if env_cfg.wbc:
        from envs.mj_env_wbc import MujocoEnv
    else:
        from envs.mj_env_base_arm import MujocoEnv

    env = MujocoEnv(env_cfg)
    env.seed(0)
    env.reset()

    obs = env.get_obs()
    base1_img = obs['base1_image']
    clicked_uv = select_pixel_manually(base1_img) # NOTE: This should come from MolMo

    if clicked_uv is not None:

        # Deproject clicked pixel
        click_3d = deproject_pixel_to_3d(obs, clicked_uv, camera_name='base1', env_cfg=env.cfg) # Camera name is which camera image the pixel is in
        print(f"3D point in robot frame: {click_3d}")

        # Get point cloud
        points, colors = pcl_from_obs(obs, env.cfg)

        # Compute soft distance-weighted click labels
        radius = 0.05  # NOTE: you should use the same as what is used in the waypoint dataset, 0.05 by default
        dists = np.linalg.norm(points - click_3d[None], axis=1)
        click_probs = np.clip((radius - dists), 0, None)
        assert(click_probs.max() > 0)
        click_probs /= click_probs.max()

        # Apply red heatmap based on soft click for visualization purposes
        red = np.array([1.0, 0.0, 0.0])
        vis_colors = colors * (1 - click_probs[:, None]) + red[None] * click_probs[:, None]

        # Visualize
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(vis_colors)
        o3d.visualization.draw_geometries([point_cloud], window_name="Click Heatmap Viewer")

        ### For salient-point conditioned eval_waypoint.py, you'd then do:

        #proprio = obs["proprio"]
        #_, pos_cmd, rot_cmd, gripper_cmd, mode = policy.inference(
        #    torch.from_numpy(points).float(),
        #    torch.from_numpy(colors).float(),
        #    torch.from_numpy(click_probs).float(),
        #    torch.from_numpy(proprio).float(),
        #    num_pass=num_pass,
        #)

        ###
