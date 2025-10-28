import cv2
from camera.realsense_camera import RealSenseCamera
from pose.pose_estimator import PoseEstimator
from filters.kalman_smoother import KalmanSmoother
from utils.helpers import get_mean_depth, deproject

def main():
    cam = RealSenseCamera()
    pose_est = PoseEstimator()
    kalman = KalmanSmoother()

    try:
        while True:
            color_image, depth_frame = cam.get_frames()
            if color_image is None:
                continue

            h, w, _ = color_image.shape
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            results = pose_est.estimate(color_image)
            if results.pose_landmarks:
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    if not (0 <= px < w and 0 <= py < h):
                        continue

                    depth = get_mean_depth(depth_frame, px, py, w, h)
                    if depth is None:
                        continue

                    X, Y, Z = deproject(depth_intrin, px, py, depth)
                    X_s, Y_s, Z_s = kalman.update(id, X, Y, Z)

                    if id == 0:  # Nose landmark
                        print(f"ID {id}: X={X_s:.2f} Y={Y_s:.2f} Z={Z_s:.2f}")

                    cv2.circle(color_image, (px, py), 4, (0,255,0), -1)

                pose_est.draw(color_image, results)

            cv2.imshow("3D Smoothed Pose Skeleton", color_image)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
