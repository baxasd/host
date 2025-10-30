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
                # Get color and depth frames from Camera Object.
                color_image, depth_frame = cam.get_frames()
                if color_image is None:
                    continue

                # Gets the height and width of color frame
                h, w, _ = color_image.shape
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

                # Gets pose and joint estimation layer from Mediapipe. Applies to color frame.
                results = pose_est.estimate(color_image)

                # Extracts depth from camera when results contains landmarks data
                if results.pose_landmarks:
                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        px, py = int(lm.x * w), int(lm.y * h)
                        if not (0 <= px < w and 0 <= py < h):
                            continue

                        depth = get_mean_depth(depth_frame, px, py, w, h)
                        if depth is None:
                            continue

                        X, Y, Z = deproject(depth_intrin, px, py, depth)

                        # Applies calman smoothing when program ran with --use-kalman arguments
                        if use_kalman and kalman:
                            X, Y, Z = kalman.update(id, X, Y, Z)

                        # Shows 3D Coordinates of specified joint ID in console when program ran with --show-depth argument.
                        if show_depth and id == 0:  # Nose
                            print(f"Landmark {id}: X={X:.3f} Y={Y:.3f} Z={Z:.3f}")

                        cv2.circle(color_image, (px, py), 4, (0,255,0), -1)
                        
                    # Draws pose on screen
                    pose_est.draw(color_image, results)
                
                # Out of if statement, works even just image output when no landmarks detected.
                cv2.imshow("3D Pose Skeleton", color_image)

            except Exception as e:
                print("Error during frame processing:", e)

            # Interrup with ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Shutting down.")

    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()
