import pickle
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R

"""
This script solves for the extrinsic calibration between each camera and the robot base
by detecting an ArUco marker attached to the robot's end-effector. It assumes prior data
has been collected using `cam_to_robot_data.pkl`, where each sample contains RGB-D
images and robot end-effector poses.

Steps:
1. Detect ArUco marker (ID 0) in each RGB frame.
2. Estimate marker pose relative to the camera using solvePnP.
3. Pair marker positions (in camera frame) with known end-effector positions (in robot frame).
4. Solve for the rigid transform (camera_T_robot) using SVD.
5. Save the resulting transform to a calibration file for each camera.
"""

def estimate_pose_single_marker(corner, marker_length, camera_matrix, dist_coeffs):
    """
    Estimate the pose (rvec, tvec) of a single ArUco marker.

    :param corner: (1, 4, 2) numpy array of image points (single marker corners)
    :param marker_length: length of the marker side (in meters)
    :param camera_matrix: camera intrinsics (3x3)
    :param dist_coeffs: distortion coefficients (length 5 or 8)
    :return: rvec, tvec
    """
    # Define 3D marker corners in marker frame
    obj_points = np.array([
        [-marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2, -marker_length / 2, 0],
        [-marker_length / 2, -marker_length / 2, 0]
    ], dtype=np.float32)

    # solvePnP expects shape (N,1,3) and (N,1,2)
    obj_points = obj_points.reshape((4, 1, 3))
    img_points = corner.reshape((4, 1, 2))

    success, rvec, tvec = cv.solvePnP(
        obj_points, img_points, camera_matrix, dist_coeffs,
        flags=cv.SOLVEPNP_IPPE_SQUARE
    )

    if not success:
        raise RuntimeError("solvePnP failed")

    return rvec, tvec

def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def detect_aruco_pose(rgb, intrinsics, marker_length):
    gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    #parameters = cv.aruco.DetectorParameters_create()
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None

    # Filter to only the marker with ID 0
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id == 0:
            rvec, tvec = estimate_pose_single_marker(
            corners[i], marker_length, intrinsics["matrix"], intrinsics["distortion"]
            )
            #rvec, tvec = rvecs[0][0], tvecs[0][0]

            # Draw 3D axis
            cv.drawFrameAxes(rgb, intrinsics['matrix'], intrinsics['distortion'], rvec, tvec, marker_length * 0.5)
            cv.imshow('img', cv.cvtColor(rgb, cv.COLOR_RGB2BGR))
            cv.waitKey(0)  # Press any key to continue
            return rvec, tvec.squeeze(), rgb

    return None  # Marker ID 0 not found

def pose_to_matrix(pos, quat):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = pos
    return T

def estimate_rigid_transform(A, B):
    # A, B are (N, 3) matrices of corresponding points (marker in cam frame, EE in robot frame)
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R_est = Vt.T @ U.T
    if np.linalg.det(R_est) < 0:
        Vt[-1, :] *= -1
        R_est = Vt.T @ U.T

    t_est = centroid_B - R_est @ centroid_A
    T = np.eye(4)
    T[:3, :3] = R_est
    T[:3, 3] = t_est
    return T

def main(data_path, marker_length=0.04):
    data = load_data(data_path)
    cam_names = data["cam_names"]
    print(cam_names)
    intrinsics = data["intrinsics"]
    samples = data["samples"]

    vis_frames = []
    for cam_name in cam_names:
        cam_pts = []
        robot_pts = []
        detected_tvecs = []

        for i, s in enumerate(samples):
            if not (f"{cam_name}_image" in s):
                continue
            result = detect_aruco_pose(s[f"{cam_name}_image"], intrinsics[cam_name], marker_length)
            if result is None:
                print(f"[{i}] No ArUco detected.")
                continue
            rvec, tvec, img = result
            vis_frames.append(img)
            detected_tvecs.append(tvec)

            R_cam = cv.Rodrigues(rvec)[0]
            T_marker_cam = np.eye(4)
            T_marker_cam[:3, :3] = R_cam
            T_marker_cam[:3, 3] = tvec

            T_ee_robot = pose_to_matrix(s["arm_pos"], s["arm_quat"])

            cam_pts.append(T_marker_cam[:3, 3])       # marker position in cam frame
            robot_pts.append(T_ee_robot[:3, 3])       # EE position in robot frame

        cam_pts = np.array(cam_pts)
        robot_pts = np.array(robot_pts)

        assert len(cam_pts) >= 3, "Need at least 3 detections to solve for transform."

        camera_T_robot = estimate_rigid_transform(cam_pts, robot_pts)
        print("Estimated camera_T_robot:\n", camera_T_robot)

        # Sanity check: transform all marker positions to robot frame
        detected_tvecs = np.array(detected_tvecs)
        transformed_pts = []

        for tvec in detected_tvecs:
            tvec_hom = np.append(tvec, 1.0)  # [x, y, z, 1]
            p_robot = camera_T_robot @ tvec_hom
            transformed_pts.append(p_robot[:3])

        transformed_pts = np.array(transformed_pts)

        print("\nSanity check: transformed marker positions vs ground truth EE positions:")
        for i in range(len(transformed_pts)):
            print(f"Detected (robot frame): {transformed_pts[i]}")
            print(f"Ground truth EE pos:    {robot_pts[i]}")
            print("Diff:", np.linalg.norm(transformed_pts[i] - robot_pts[i]), "\n")

        save_path = f"calib_files/{cam_name}_extrinsics.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(camera_T_robot, f)
        print(f"\n[INFO] Saved extrinsics to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="calib_scripts/cam_to_robot_data.pkl")
    parser.add_argument("--marker_length", type=float, default=0.05)
    args = parser.parse_args()
    main(args.data_path, args.marker_length)
