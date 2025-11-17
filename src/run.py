import cv2
from src.camera.realsense_camera import RealSenseCamera
from src.pose.pose_estimator import PoseEstimator
from src.filters.kalman_smoother import KalmanSmoother
from src.utils.helpers import get_mean_depth, deproject

from src.utils.angle_calculator import AngleCalculator
from src.utils.csv_writer import CSVLogger
from src.utils.angle_plotter import AnglePlotter


def run_system(use_kalman=True, show_depth=False, show_angles=False, model=1):
    """Main function to run the full system."""

    print("[INFO] Initializing camera...")

    # Initialize objects
    cam = RealSenseCamera(verbose=True)
    pose_est = PoseEstimator(model)
    kalman = KalmanSmoother() if use_kalman else None

    angle_calc = AngleCalculator()
    logger = CSVLogger()

    print(f"[INFO] Kalman filter {'ENABLED' if use_kalman else 'DISABLED'}")

    try:
        while True:
            color_image, depth_frame = cam.get_frames()
            if color_image is None:
                continue

            h, w, _ = color_image.shape
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            # Pose estimation
            results = pose_est.estimate(color_image)
            annotated_image = pose_est.draw_landmarks(color_image, results)

            # Extract 3D coordinates
            landmarks_dict = {}
            if results.pose_landmarks:
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    if not (0 <= px < w and 0 <= py < h):
                        continue

                    depth = get_mean_depth(depth_frame, px, py, w, h)
                    if depth is None:
                        continue

                    X, Y, Z = deproject(depth_intrin, px, py, depth)

                    if use_kalman and kalman:
                        X, Y, Z = kalman.update(id, X, Y, Z)

                    landmarks_dict[id] = (X, Y, Z)

                    if show_depth:
                        cv2.putText(color_image,
                                    f"{id}: ({X:.2f}, {Y:.2f}, {Z:.2f})",
                                    (px, py - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4, (0, 255, 255), 1, cv2.LINE_AA)

                # Calculate joint angles
                angles = angle_calc.calculate(landmarks_dict)

                # Log to CSV
                logger.log(angles)

                if show_angles:
                    for joint, angle in angles.items():
                        if angle is not None:
                            # Map joint names to landmark IDs for placement
                            joint_to_id = {
                                'left_elbow': 13, 'right_elbow': 14,
                                'left_shoulder': 11, 'right_shoulder': 12,
                                'left_hip': 23, 'right_hip': 24,
                                'left_knee': 25, 'right_knee': 26
                            }
                            lm_id = joint_to_id.get(joint)
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
        print("\n[INFO] Interrupted by user. Shutting down.")

    finally:
        logger.close()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_system()
