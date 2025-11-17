"""Application runner: initialize components and run the main loop.

This module wires together the camera, pose estimator, optional
Kalman smoother, angle calculator and CSV logger. It keeps the
main loop simple and focuses on orchestrating data flow.
"""

import cv2
import logging
from src.camera.realsense_camera import RealSenseCamera
from src.pose.pose_estimator import PoseEstimator
from src.filters.kalman_smoother import KalmanSmoother
from src.utils.helpers import get_mean_depth, deproject

from src.utils.angle_calculator import AngleCalculator
from src.utils.csv_writer import CSVLogger


def run_system(use_kalman=True, show_depth=False, show_angles=False, model=1):
    """Main function to run the full system.

    Parameters mirror the CLI flags and allow programmatic control in
    addition to the command line.
    """

    logging.info("Initializing camera...")

    # Initialize objects
    cam = RealSenseCamera(verbose=True)
    pose_est = PoseEstimator(model)
    kalman = KalmanSmoother() if use_kalman else None

    angle_calc = AngleCalculator()
    logger = CSVLogger()

    logging.info(f"Kalman filter {'ENABLED' if use_kalman else 'DISABLED'}")

    try:
        while True:
            # Get synchronized color image and depth frame from camera
            color_image, depth_frame = cam.get_frames()
            if color_image is None:
                # skip iteration if frames were not available
                continue

            h, w, _ = color_image.shape
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            # Pose estimation on the latest color frame
            results = pose_est.estimate(color_image)
            annotated_image = pose_est.draw_landmarks(color_image, results)

            # Extract 3D coordinates for detected landmarks
            landmarks_dict = {}
            if results and results.pose_landmarks:
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    if not (0 <= px < w and 0 <= py < h):
                        continue

                    # Sample a small patch around the pixel to reduce noise
                    depth = get_mean_depth(depth_frame, px, py, w, h)
                    if depth is None:
                        continue

                    # Convert pixel+depth to 3D camera coordinates. `deproject`
                    # may return None on failure, so guard against that.
                    xyz = deproject(depth_intrin, px, py, depth)
                    if xyz is None:
                        continue
                    X, Y, Z = xyz

                    # Optionally smooth using Kalman filter per-joint
                    if use_kalman and kalman:
                        X, Y, Z = kalman.update(id, X, Y, Z)

                    landmarks_dict[id] = (X, Y, Z)

                    if show_depth:
                        # Overlay depth (XYZ) near the landmark for debugging
                        cv2.putText(color_image,
                                    f"{id}: ({X:.2f}, {Y:.2f}, {Z:.2f})",
                                    (px, py - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4, (0, 255, 255), 1, cv2.LINE_AA)

                # Calculate joint angles from 3D landmarks
                angles = angle_calc.calculate(landmarks_dict)

                # Log angles to CSV for later analysis
                logger.log(angles)

                if show_angles:
                    # Overlay angles on the image for selected joints
                    for joint, angle in angles.items():
                        if angle is not None:
                            # Map joint names to landmark IDs for placement
                            from src.utils.landmarks import JOINT_TO_LANDMARK_ID
                            lm_id = JOINT_TO_LANDMARK_ID.get(joint)
                            if lm_id is None:
                                continue
                            px = int(results.pose_landmarks.landmark[lm_id].x * w)
                            py = int(results.pose_landmarks.landmark[lm_id].y * h)
                            cv2.putText(color_image,
                                        f"{joint}: {angle:.1f}",
                                        (px, py - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # Display skeleton
            cv2.imshow("3D Pose Skeleton", annotated_image)

            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        logging.info("Interrupted by user. Shutting down.")

    finally:
        logger.close()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_system()
