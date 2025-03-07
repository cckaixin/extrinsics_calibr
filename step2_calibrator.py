import cv2
import numpy as np
import yaml
import argparse
import os
from glob import glob

def load_board(config_file):
    """Load ChArUco board configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    squares_x = config.get("squares_x", 5)
    squares_y = config.get("squares_y", 7)
    square_length_mm = config.get("square_length_mm", 30) / 1000.0  # convert to meters
    marker_length_mm = config.get("marker_length_mm", 24) / 1000.0  # convert to meters
    aruco_dict_name = config.get("aruco_dict", "DICT_4X4_50")

    # Get the corresponding ArUco dictionary
    aruco_dict = getattr(cv2.aruco, aruco_dict_name, None)
    if aruco_dict is None:
        raise ValueError(f"Invalid ArUco dictionary: {aruco_dict_name}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
    charuco_board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_length_mm, marker_length_mm, aruco_dict
    )
    return charuco_board


def get_image_pairs(cam1_dir, cam2_dir):
    """Retrieve sorted image file paths from both camera directories."""
    cam1_images = sorted(glob(os.path.join(cam1_dir, '*.png')))
    cam2_images = sorted(glob(os.path.join(cam2_dir, '*.png')))

    assert len(cam1_images) == len(cam2_images), "Number of images in both cameras do not match."
  
    return list(zip(cam1_images, cam2_images))

def detect_charuco_corners(charuco_board, image, aruco_dict):
    """Detect ChArUco corners in an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.CharucoDetector(charuco_board)

    charucoCorners, charucoIds, markerCorners, markerIds = detector.detectBoard(gray)

    if charucoIds is not None and len(charucoIds) > 3:   # Require at least 4 corners
        return charucoCorners, charucoIds
    else:
        return None, None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Calibrate extrinsic parameters between two RealSense cameras.")
    parser.add_argument("--basecam_id", type=int, required=True, help="Base camera ID")
    parser.add_argument("--subcam_id", type=int, required=True, help="Sub camera ID")
    parser.add_argument("--task_path", type=str, required=True, help="Path to calibration task directory")
    parser.add_argument("--board", type=str, required=True, help="Path to the board YAML configuration file.")
    args = parser.parse_args()

    # configruation
    basecam_id = args.basecam_id
    subcam_id = args.subcam_id
    task_path = args.task_path
    board_config_path = args.board
    intrinsics_file = os.path.join(task_path, f"cali_T_{basecam_id}_{subcam_id}", "intrinsics.yaml")
    cam1_dir = os.path.join(task_path, f"cali_T_{basecam_id}_{subcam_id}", f"cam{basecam_id}")  # cam1 is the base camera in this script
    cam2_dir = os.path.join(task_path, f"cali_T_{basecam_id}_{subcam_id}", f"cam{subcam_id}")   # cam2 is the sub camera in this script
    result_file = os.path.join(task_path, f"cali_T_{basecam_id}_{subcam_id}", "extrinsics.yaml")

    # Load ChArUco board and dictionary
    try:
        charuco_board = load_board(board_config_path)
        aruco_dict = charuco_board.getDictionary    # Prepare ArUco dictionary
        print("ChArUco board loaded successfully.")
    except Exception as e:
        print(f"Error loading board configuration: {e}")
        return

    # Load intrinsics
    with open(intrinsics_file, 'r') as f:
        intrinsics = yaml.safe_load(f)
    try:
        cam1_matrix, cam1_dist = np.array(intrinsics[f"cam{basecam_id}"]['camera_matrix']), np.array(intrinsics[f"cam{basecam_id}"]['dist_coefficients'])
        cam2_matrix, cam2_dist = np.array(intrinsics[f"cam{subcam_id}"]['camera_matrix']), np.array(intrinsics[f"cam{subcam_id}"]['dist_coefficients'])
        print("Camera intrinsics loaded successfully.")
    except Exception as e:
        print(f"Error loading intrinsics: {e}")
        return

    # Load image pairs
    image_pairs = get_image_pairs(cam1_dir, cam2_dir)
    print(f"Found {len(image_pairs)} image pairs.")

    # Collections for calibration
    all_obj_points = []     # 3D points
    all_cam1_img_points = []
    all_cam2_img_points = []

    # Object points (assuming board is placed on Z=0 plane)
    obj_3d = charuco_board.getChessboardCorners()  # Shape: (N, 3)
    print(f"before:obj_3d:{obj_3d}")

    for idx, (cam1_img_path, cam2_img_path) in enumerate(image_pairs):
        print(f"Processing pair {idx+1}: {os.path.basename(cam1_img_path)} & {os.path.basename(cam2_img_path)}")
        cam1_img = cv2.imread(cam1_img_path)
        cam2_img = cv2.imread(cam2_img_path)

        if cam1_img is None or cam2_img is None:
            print(f"Warning: Unable to read images {cam1_img_path} or {cam2_img_path}. Skipping.")
            continue

        # Detect corners in both images
        cam1_corners, cam1_ids = detect_charuco_corners(charuco_board, cam1_img, aruco_dict)
        cam2_corners, cam2_ids = detect_charuco_corners(charuco_board, cam2_img, aruco_dict)

        if cam1_corners is not None and cam2_corners is not None:
            # Find common IDs
            cam1_ids_set = set(cam1_ids.flatten())
            cam2_ids_set = set(cam2_ids.flatten())
            common_ids = cam1_ids_set.intersection(cam2_ids_set)

            if len(common_ids) < 4:
                print(f"Not enough common corners in pair {idx+1}. Skipping.")
                continue

            # Extract corresponding corners
            cam1_common = []
            cam2_common = []
            obj_common = []

            for id in common_ids:
                cam1_idx = np.where(cam1_ids.flatten() == id)[0][0]
                cam2_idx = np.where(cam2_ids.flatten() == id)[0][0]
                cam1_common.append(cam1_corners[cam1_idx])
                cam2_common.append(cam2_corners[cam2_idx])

                # Directly index obj_3d using the id
                if id < len(obj_3d):
                    obj_common.append(obj_3d[id])
                else:
                    print(f"Warning: ID {id} is out of bounds for obj_3d with length {len(obj_3d)}")
            # print(f"cam1_common:{cam1_common}")
            # print(f"cam2_common:{cam2_common}")
            # print(f"obj_common:{obj_common}")
            cam1_common = np.array(cam1_common, dtype=np.float32)   # meaning of this variable is the 2D points in cam1
            cam2_common = np.array(cam2_common, dtype=np.float32)   # meaning of this variable is the 2D points in cam2
            obj_common = np.array(obj_common, dtype=np.float32)     # meaning of this variable is the 3D points

            # Append to collections
            all_obj_points.append(obj_common)
            all_cam1_img_points.append(cam1_common)
            all_cam2_img_points.append(cam2_common)
        else:
            print(f"ChArUco corners not detected in pair {idx+1}. Skipping.")

    if len(all_obj_points) < 5:
        print("Not enough valid image pairs for calibration. Need at least 5.")
        return

    # Stereo calibration
    print("Starting stereo calibration...")
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    flags = (cv2.CALIB_FIX_INTRINSIC)

    ret, cam1_matrix, cam1_dist, cam2_matrix, cam2_dist, R, T, E, F = cv2.stereoCalibrate(
        all_obj_points,
        all_cam1_img_points,
        all_cam2_img_points,
        cam1_matrix,
        cam1_dist,
        cam2_matrix,
        cam2_dist,
        charuco_board.imageSize if hasattr(charuco_board, 'imageSize') else (1280, 720),
        criteria=criteria,
        flags=flags
    )

    if ret < 0:
        print("Stereo calibration failed.")
        return

    print("Stereo calibration successful.")
    print(f"Rotation matrix:\n{R}")
    print(f"Translation vector:\n{T}")

    # Save extrinsic parameters
    extrinsics = {
        f"T_{basecam_id}_{subcam_id}": {
            "R": R.tolist(),
            "T": T.tolist(),
            # "E": E.tolist(),
            # "F": F.tolist(),
            "stereo_calib_error": ret
        }
    }

    with open(result_file, 'w') as f:
        yaml.dump(extrinsics, f)

    print(f"Extrinsic parameters saved to {result_file}")

if __name__ == "__main__":
    main()