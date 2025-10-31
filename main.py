import cv2
import time
import numpy as np
from camera.realsense_camera import RealSenseCamera
from pose.pose_estimator import PoseEstimator
from filters.kalman_smoother import KalmanSmoother
from utils.helpers import get_mean_depth, deproject

def run_system(use_kalman=True, show_depth=False):
    print("[INFO] Initializing camera with GPU acceleration...")
    cam = RealSenseCamera(verbose=True)
    
    # Use GPU-accelerated pose estimator
    pose_est = PoseEstimator(model_path="./models/pose_landmarker_heavy.task")
    kalman = KalmanSmoother() if use_kalman else None
    print(f"[INFO] Kalman filter {'ENABLED' if use_kalman else 'DISABLED'}")
    print("[INFO] GPU acceleration ENABLED")

    frame_count = 0
    fps_counter = 0
    fps_timer = time.time()
    fps = 0  # Initialize fps variable

    try:
        while True:
            start_time = time.time()
            
            color_image, depth_frame = cam.get_frames()
            if color_image is None:
                continue

            # Process every frame with GPU acceleration
            result = pose_est.detect_pose(color_image, frame_count)
            frame_count += 1

            if result is not None:
                annotated_image = PoseEstimator.draw_landmarks_on_image(color_image, result)

                # Extract 3D coordinates
                if result.pose_landmarks:
                    h, w, _ = color_image.shape
                    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

                    for person_landmarks in result.pose_landmarks:
                        for landmark_id, landmark in enumerate(person_landmarks):
                            px, py = int(landmark.x * w), int(landmark.y * h)
                            if not (0 <= px < w and 0 <= py < h):
                                continue

                            depth = get_mean_depth(depth_frame, px, py, w, h)
                            if depth is None:
                                continue

                            X, Y, Z = deproject(depth_intrin, px, py, depth)
                            if use_kalman and kalman:
                                X, Y, Z = kalman.update(landmark_id, X, Y, Z)

                            if show_depth and landmark_id == 0:
                                print(f"Landmark {landmark_id}: X={X:.3f} Y={Y:.3f} Z={Z:.3f}")

                # Display the annotated image
                display_image = annotated_image
            else:
                display_image = color_image

            # Calculate and display FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_timer >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_timer = current_time
                print(f"[PERF] FPS: {fps}")

            # Add FPS text to the display image
            cv2.putText(display_image, f"FPS: {fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_image, "GPU: ENABLED", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("3D Pose Skeleton (GPU)", display_image)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Shutting down.")

    finally:
        cam.stop()
        cv2.destroyAllWindows()