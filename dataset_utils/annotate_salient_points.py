import sys
import pyrallis
import open3d as o3d
import pickle
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from interactive_scripts.dataset_recorder import ActMode
from envs.utils.camera_utils import pcl_from_obs

class PointCloudViewer(QMainWindow):
    def __init__(self, merged_points=None, merged_colors=None, index=None):
        super().__init__()
        self.merged_points = merged_points
        self.merged_colors = merged_colors
        self.index = index
        self.clicked_point = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Point Cloud Viewer")
        self.setGeometry(400, 400, 200, 200)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Label to display point cloud index
        self.index_label = QLabel(f"Point Cloud Index: {self.index}")
        self.index_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.index_label)

        # Label to display clicked point
        self.point_label = QLabel("Clicked Point: None")
        self.point_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.point_label)

    def display_point_cloud(self):
        self.vis = o3d.visualization.VisualizerWithEditing()
        self.vis.create_window("Point Cloud Viewer", width=2000, height=1000)

        # Create and zoom into point cloud
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(self.merged_points)
        self.point_cloud.colors = o3d.utility.Vector3dVector(self.merged_colors)
        self.vis.add_geometry(self.point_cloud)
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.3)  # Zoom in

        self.vis.run()
        picked_indices = self.vis.get_picked_points()
        self.vis.destroy_window()

        if picked_indices:
            self.display_picked_points(picked_indices)

    def display_picked_points(self, picked_indices):
        picked_points = np.asarray(self.point_cloud.points)[picked_indices]
        if len(picked_points) > 0:
            self.clicked_point = picked_points[0]
            # Update label
            self.update_label(self.clicked_point)

    def update_label(self, point):
        self.point_label.setText(f"Clicked Point: {point}")

    def get_clicked_point(self):
        return self.clicked_point

def label_salient_points(episode_fn, env_cfg):
    with open(episode_fn, "rb") as fp:
        demo = pickle.load(fp)

    app = QApplication(sys.argv)

    for t, step in enumerate(list(demo)):
        if step['mode'] in [ActMode.ArmWaypoint, ActMode.BaseWaypoint]:

            merged_points, merged_colors = pcl_from_obs(step["obs"], env_cfg)

            viewer = PointCloudViewer(merged_points=merged_points, merged_colors=merged_colors, index=t)
            viewer.display_point_cloud()

            clicked_point = viewer.get_clicked_point()
            if clicked_point is not None:
                action = step["action"][8:11] if step["mode"] == ActMode.BaseWaypoint else step["action"][:3]
                print(f"t: {t}, click: {clicked_point}, action: {action}, mode: {step['mode']}")
                step["click"] = np.array(clicked_point)  # Update obs with the clicked point
    
    print('Done, saving %s'%episode_fn)
    with open(episode_fn, "wb") as f:
        pickle.dump(demo, f)
    print('Saved %s'%episode_fn)
    print('****')
    #sys.exit(app.exec_())

if __name__ == "__main__":
    # Example usage
    data_dir = 'dev1_relabeled'
    try:
        from envs.common_mj_env import MujocoEnvConfig
        env_cfg = pyrallis.load(MujocoEnvConfig, open(os.path.join(data_dir, "env_cfg.yaml"), "r"))
    except:
        from envs.common_real_env_cfg import RealEnvConfig, load_calibration_matrices
        env_cfg = pyrallis.load(RealEnvConfig, open(os.path.join(data_dir, "env_cfg.yaml"), "r"))

    for fn in sorted(os.listdir(data_dir)):
        if 'pkl' in fn:
            label_salient_points(os.path.join(data_dir, fn), env_cfg)
