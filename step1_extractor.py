import logging
from typing import List, Dict
import pyrealsense2 as rs
import os
import shutil
import numpy as np
import argparse
import cv2
import yaml

serial_dict = {
    1: "234322070242",  # wrist
    2: "233522070688",  # front
    3: "236522070121",  # left
    4: "828112072646",  # right
}

camera_info = {
    "format": rs.format.bgr8,
    "fps": 30,
    "width": 1280,
    "height": 720,
}

def initialise(arg, overwrite_images: bool = True) -> (List[str], List[rs.pipeline]):
    """ Find and initialize all connected RealSense cameras, optionally make new folders for saving captured images.
        Also extracts and saves intrinsic parameters to intrinsics.yaml.
    """
    # Initialize cameras
    paths = []
    pipelines = []
    intrinsics = {}

    basecam_id = arg.basecam_id
    subcam_id = arg.subcam_id

    # Calibration task directory
    calib_task_dir = os.path.join(arg.task_path, f"cali_T_{basecam_id}_{subcam_id}")
    intrinsics_file_path = os.path.join(calib_task_dir, "intrinsics.yaml")

    for cam_id in [basecam_id, subcam_id]:
        serial = serial_dict.get(cam_id)
        if not serial:
            logging.warning(f"No serial number found for camera ID {cam_id}. Skipping.")
            continue

        logging.info(f"Camera ID {cam_id} with serial {serial} found, starting stream.")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        try:
            config.enable_stream(
                rs.stream.color,
                camera_info["width"],
                camera_info["height"],
                camera_info["format"],
                camera_info["fps"]
            )
            pipeline_profile = pipeline.start(config)
        except RuntimeError:
            logging.error(f"Failed to start camera ID {cam_id} with serial {serial}.")
            raise RuntimeError(f"Cannot start camera ID {cam_id} with serial {serial}.")

        pipelines.append(pipeline)
        cam_dir = os.path.join(calib_task_dir, f"cam{cam_id}")
        paths.append(cam_dir)

        # Extract intrinsic parameters
        try:
            # Wait for the camera to emit a few frames to ensure it's streaming
            for _ in range(5):
                pipeline.wait_for_frames()

            profile = pipeline_profile.get_stream(rs.stream.color)
            intrinsics_rs = profile.as_video_stream_profile().get_intrinsics()

            # Map RealSense intrinsics to OpenCV format
            camera_matrix = [
                [intrinsics_rs.fx, 0.0, intrinsics_rs.ppx],
                [0.0, intrinsics_rs.fy, intrinsics_rs.ppy],
                [0.0, 0.0, 1.0]
            ]

            # RealSense distortion coefficients might have more than 5 coefficients depending on the model
            # OpenCV typically uses 5: [k1, k2, p1, p2, k3]
            # Adjust based on the distortion model
            distortion_coefficients = list(intrinsics_rs.coeffs[:5])

            # Store in intrinsics dictionary
            intrinsics[f"cam{cam_id}"] = {
                "camera_matrix": camera_matrix,
                "dist_coefficients": distortion_coefficients
            }

            logging.info(f"Extracted intrinsics for cam{cam_id}:")
            logging.info(f"Camera Matrix: {camera_matrix}")
            logging.info(f"Distortion Coefficients: {distortion_coefficients}")

        except Exception as e:
            logging.error(f"Failed to extract intrinsics for camera ID {cam_id}: {e}")
            raise e

    # Make output folders
    if overwrite_images:
        for path in paths:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

    # Save intrinsics to YAML
    try:
        with open(intrinsics_file_path, 'w') as f:
            yaml.dump(intrinsics, f)
        logging.info(f"Intrinsic parameters saved to {intrinsics_file_path}")
    except Exception as e:
        logging.error(f"Failed to save intrinsic parameters to YAML: {e}")
        raise e

    return paths, pipelines

def save_frames(frames: List[np.ndarray], paths: List[str]) -> None:
    assert len(frames) == len(paths), "Number of frames and paths do not match"
    for i in range(len(frames)):
        save_dir = paths[i]
        fnum = f"{len(os.listdir(save_dir)):04d}"
        filename = os.path.join(save_dir, f"{fnum}.png")
        cv2.imwrite(filename, frames[i])
    logging.info(f"Saved frame {fnum}.")

def capture_realsense_sync(paths: List[str], cams: List[rs.pipeline]) -> None:
    logging.info("Press 'c' to capture, 'q' to quit.")
    try:
        while True:
            frames = []
            # Get frames
            for cam in cams:
                rs_frames = cam.wait_for_frames()
                rs_col_frame = rs_frames.get_color_frame()
                if not rs_col_frame:
                    logging.warning("No color frame received.")
                    continue
                frames.append(np.asanyarray(rs_col_frame.get_data()))

            if not frames:
                logging.warning("No frames captured.")
                continue

            # Display frames side by side
            concatenated = np.hstack(frames)
            cv2.imshow("View", concatenated)
            k = cv2.waitKey(1)

            # Capture
            if k in [ord('c'), ord("C")]:
                save_frames(frames, paths)

            # Break loop
            elif k in [ord('q'), ord('Q')]:
                break

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")

    finally:
        # Close
        logging.info("Stopping camera pipelines.")
        for cam in cams:
            cam.stop()
        cv2.destroyAllWindows()
        logging.info("Closed all windows.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract and save images from RealSense cameras along with intrinsic parameters.")
    parser.add_argument("--basecam_id", type=int, required=True, help="Base camera ID (e.g., 1-4)")
    parser.add_argument("--subcam_id", type=int, required=True, help="Sub camera ID (e.g., 1-4)")
    parser.add_argument("--task_path", type=str, required=True, help="Path for storing images and calibration data")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing image directories")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        paths, pipelines = initialise(args, overwrite_images=args.overwrite)
        capture_realsense_sync(paths, pipelines)
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()