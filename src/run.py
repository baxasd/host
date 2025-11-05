import cv2
from src.camera.realsense_camera import RealSenseCamera
from src.pose.pose_estimator import PoseEstimator
from src.filters.kalman_smoother import KalmanSmoother
from src.utils.helpers import get_mean_depth, deproject

def run_system(use_kalman=True, show_depth=False, model=1):
    """ Main function to run the system """

    print("[INFO] Initializing camera...")

    # Initializing objects
    cam = RealSenseCamera(verbose=True)
    pose_est = PoseEstimator(model)
    kalman = KalmanSmoother() if use_kalman else None

    print(f"[INFO] Kalman filter {'ENABLED' if use_kalman else 'DISABLED'}")

    # Main Loop
    try:
        while True:
            color_image, depth_frame = cam.get_frames()
            if color_image is None:
                continue

            h, w, _ = color_image.shape
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            results = pose_est.estimate(color_image)

            # Draw landmarks
            annotated_image = pose_est.draw_landmarks(color_image, results)

            # Extract 3D coordinates if needed
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

                    if show_depth:
                        cv2.putText(color_image,
                                    f"{id}: ({X:.2f}, {Y:.2f}, {Z:.2f})",
                                    (px, py - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4, (0, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow("3D Pose Skeleton", annotated_image)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Shutting down.")

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_system()
