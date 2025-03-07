import open3d as o3d
import numpy as np
import yaml
import argparse
import os
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize camera extrinsics with Open3D and export summary.")
    parser.add_argument("--basecam_id", type=int, required=True, help="Base camera ID (e.g., 2)")
    parser.add_argument("--subcam_id", type=int, nargs='+', required=True, help="Sub camera IDs (e.g., 1 3 4)")
    parser.add_argument("--task_path", type=str, required=True, help="Path to calibration task directory")
    return parser.parse_args()


def load_extrinsics(calibration_dir, basecam_id, subcam_id):
    extrinsics_file = os.path.join(calibration_dir, f"extrinsics.yaml")
    if not os.path.exists(extrinsics_file):
        print(f"Extrinsics file not found: {extrinsics_file}")
        sys.exit(1)
    
    with open(extrinsics_file, 'r') as f:
        extrinsics = yaml.safe_load(f)
    
    key = f"T_{basecam_id}_{subcam_id}"
    if key not in extrinsics:
        print(f"Extrinsics key '{key}' not found in {extrinsics_file}")
        sys.exit(1)
    
    R = np.array(extrinsics[key]["R"])
    T = np.array(extrinsics[key]["T"]).reshape(3, 1)
    return R, T


def load_intrinsics(calibration_dir, basecam_id, subcam_id):
    intrinsics_file = os.path.join(calibration_dir, "intrinsics.yaml")
    if not os.path.exists(intrinsics_file):
        print(f"Intrinsics file not found: {intrinsics_file}")
        sys.exit(1)
    
    with open(intrinsics_file, 'r') as f:
        intrinsics = yaml.safe_load(f)
    
    base_cam_intrinsic = intrinsics.get(f"cam{basecam_id}", {}).get("camera_matrix", [])
    sub_cam_intrinsic = intrinsics.get(f"cam{subcam_id}", {}).get("camera_matrix", [])
    
    return base_cam_intrinsic, sub_cam_intrinsic


def create_transformation_matrix(R, T):
    """
    Create a 4x4 homogeneous transformation matrix from rotation and translation.
    """
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = T.flatten()
    return transformation


def invert_transformation(R, T):
    """
    Invert the transformation to get from camera to world frame.
    """
    R_inv = R.T
    T_inv = -R_inv @ T
    return R_inv, T_inv


def visualize_cameras(basecam_id, subcam_ids, task_path):
    # Initialize Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Camera Extrinsics Visualization', width=1280, height=720)
    
    # Create base camera frame (world frame)
    base_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    base_cam_frame.paint_uniform_color([1, 0, 0])  # Red for base camera
    vis.add_geometry(base_cam_frame)
    
    # Define color map for sub-cameras
    color_map = {
        1: [0, 1, 0],   # Green
        3: [0, 0, 1],   # Blue
        4: [1, 1, 0],   # Yellow
        # Add more colors if needed
    }
    
    # Dictionary to store intrinsics for summary
    summary_intrinsics = {}
    
    # Dictionary to store extrinsics for summary
    summary_extrinsics = {}
    
    # Process each sub-camera
    for subcam_id in subcam_ids:
        calibration_dir = os.path.join(task_path, f"cali_T_{basecam_id}_{subcam_id}")
        if not os.path.isdir(calibration_dir):
            print(f"Calibration directory not found: {calibration_dir}")
            continue
        
        # Load extrinsics
        try:
            R, T = load_extrinsics(calibration_dir, basecam_id, subcam_id)
        except SystemExit:
            print(f"Skipping camera {subcam_id} due to missing extrinsics.")
            continue
        
        # Invert the transformation to get camera pose in world frame
        R_wc, T_wc = invert_transformation(R, T)
        
        # Create transformation matrix
        transformation_wc = create_transformation_matrix(R_wc, T_wc)
        
        # Create sub-camera frame
        sub_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        sub_cam_frame.transform(transformation_wc)
        
        # Assign color based on camera ID
        color = color_map.get(subcam_id, [0.5, 0.5, 0.5])  # Default to gray if ID not in map
        sub_cam_frame.paint_uniform_color(color)
        
        # Add to visualization
        vis.add_geometry(sub_cam_frame)
        
        print(f"Added Camera {subcam_id} to visualization.")
        
        # Load intrinsics
        try:
            base_cam_intrinsic, sub_cam_intrinsic = load_intrinsics(calibration_dir, basecam_id, subcam_id)
            summary_intrinsics[f"cam{basecam_id}"] = base_cam_intrinsic
            summary_intrinsics[f"cam{subcam_id}"] = sub_cam_intrinsic
        except SystemExit:
            print(f"Skipping intrinsics collection for camera {subcam_id} due to missing intrinsics.")
            continue
        
        # Store extrinsics for summary
        T_full = create_transformation_matrix(R, T)
        summary_extrinsics[f"T_{basecam_id}_{subcam_id}"] = T_full.tolist()
    
    # Render the scene in a non-blocking way
    vis.run()
    vis.destroy_window()
    
    # Export summary
    export_summary(basecam_id, subcam_ids, task_path, summary_intrinsics, summary_extrinsics)


def export_summary(basecam_id, subcam_ids, task_path, intrinsics, extrinsics, output_file="summary.txt"):
    """
    Export a summary file in the specified format.
    """
    all_cam_ids = [basecam_id] + subcam_ids
    
    with open(os.path.join(task_path, output_file), 'w') as f:
        for cam_id in sorted(all_cam_ids):
            f.write(f"cam{cam_id}\n")
            f.write("    intrinsic\n")
            cam_matrix = intrinsics.get(f"cam{cam_id}", [])
            if not cam_matrix:
                f.write("        []\n")
            else:
                # Format the camera_matrix with proper indentation
                matrix_str = "        " + str(cam_matrix).replace('], [', '],\n        [') + "\n"
                f.write(matrix_str)
            
            if cam_id != basecam_id:
                key = f"T_{basecam_id}_{cam_id}"
                if key in extrinsics:
                    f.write(f'    {key}: \n')
                    T_full = extrinsics[key]
                    for row in T_full[:4]:
                        # Format each row with indentation
                        row_str = "            " + str(row) + ",\n"
                        f.write(row_str)
                else:
                    f.write(f'    "{key}": \n')
                    f.write("            []\n")
            f.write("\n")
    
    print(f"Summary exported to {os.path.join(task_path, output_file)}")

 
def main():
    args = parse_arguments()
    basecam_id = args.basecam_id
    subcam_ids = args.subcam_id
    task_path = args.task_path
    
    if not os.path.isdir(task_path):
        print(f"Task path does not exist: {task_path}")
        sys.exit(1)
    
    visualize_cameras(basecam_id, subcam_ids, task_path)


if __name__ == "__main__":
    main()