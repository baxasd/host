import cv2
from camera.realsense_camera import RealSenseCamera
from pose.pose_estimator import PoseEstimator
from filters.kalman_smoother import KalmanSmoother
from utils.helpers import get_mean_depth, deproject

def run_system(use_kalman=True, show_depth=False):

    # Just simple info
    print("[INFO] Initializing camera...")

    # Initialize Objects
    cam = RealSenseCamera(verbose=True)
    pose_est = PoseEstimator()
    kalman = KalmanSmoother() if use_kalman else None

    print(f"[INFO] Kalman filter {'ENABLED' if use_kalman else 'DISABLED'}")
    
    # Main Loop Logic with exception handling.
    try:
        while True:
            try:
                color_image, depth_frame = cam.get_frames()
                if color_image is None:
                    continue

                # Convert BGR to RGB for MediaPipe
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                detection_result = pose_est.estimate(rgb_image)

                # Draw pose landmarks with connections
                annotated_image = PoseEstimator.draw_landmarks_on_image(rgb_image, detection_result)

                # Optionally, extract 3D coordinates using depth + Kalman
                if detection_result.pose_landmarks:
                    h, w, _ = color_image.shape
                    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

                    for person_landmarks in detection_result.pose_landmarks:
                        for id, lm in enumerate(person_landmarks):
                            px, py = int(lm.x * w), int(lm.y * h)
                            if not (0 <= px < w and 0 <= py < h):
                                continue

                            depth = get_mean_depth(depth_frame, px, py, w, h)
                            if depth is None:
                                continue

                            X, Y, Z = deproject(depth_intrin, px, py, depth)
                            if use_kalman and kalman:
                                X, Y, Z = kalman.update(id, X, Y, Z)

                            if show_depth and id == 0:
                                print(f"Landmark {id}: X={X:.3f} Y={Y:.3f} Z={Z:.3f}")

                cv2.imshow("3D Pose Skeleton", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print("Error during frame processing:", e)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Shutting down.")

    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()
