import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from filterpy.kalman import KalmanFilter

# --- Initialize Mediapipe ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True)

# --- Initialize RealSense ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

align = rs.align(rs.stream.color)

# --- Kalman filters per joint ---
joint_ids = [i for i in range(33)]  # Mediapipe Pose has 33 landmarks
kalman_filters = {}

for jid in joint_ids:
    kf = KalmanFilter(dim_x=6, dim_z=3)
    dt = 1/30
    kf.F = np.array([[1,0,0,dt,0,0],
                     [0,1,0,0,dt,0],
                     [0,0,1,0,0,dt],
                     [0,0,0,1,0,0],
                     [0,0,0,0,1,0],
                     [0,0,0,0,0,1]])
    kf.H = np.array([[1,0,0,0,0,0],
                     [0,1,0,0,0,0],
                     [0,0,1,0,0,0]])
    kf.P *= 0.1
    kf.R *= 0.01
    kf.Q *= 0.01
    kf.x[:3] = 0
    kf.x[3:] = 0
    kalman_filters[jid] = kf

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        h, w, _ = color_image.shape
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)

        if results.pose_landmarks:
            coords_3d = {}  # Store smoothed 3D coordinates

            for id, lm in enumerate(results.pose_landmarks.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                if px < 0 or py < 0 or px >= w or py >= h:
                    continue

                # Average depth around 3x3 patch
                depth_values = []
                for dx in range(-1,2):
                    for dy in range(-1,2):
                        x, y = px + dx, py + dy
                        if 0 <= x < w and 0 <= y < h:
                            d = depth_frame.get_distance(x, y)
                            if d > 0:
                                depth_values.append(d)
                if not depth_values:
                    continue
                z = np.mean(depth_values)

                # Deproject to real-world 3D
                X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [px, py], z)

                # --- Kalman filter update ---
                kf = kalman_filters[id]

                # Initialize first state for Kalman
                if np.all(kf.x[:3] == 0):
                    kf.x[:3] = np.array([[X], [Y], [Z]])

                kf.predict()
                kf.update(np.array([X, Y, Z]))
                X_s, Y_s, Z_s = kf.x[:3]

                coords_3d[id] = (X_s, Y_s, Z_s)

                                # Optional: print some landmarks
                if id in [mp_pose.PoseLandmark.NOSE.value]:
                    print(f"ID {id}: X={X:.2f} Y={Y:.2f} Z={Z:.2f} meters")

                # Draw landmark
                cv2.circle(color_image, (px, py), 5, (0, 255, 0), -1)

            # Draw skeleton connections
            mp.solutions.drawing_utils.draw_landmarks(
                color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("3D Smoothed Pose Skeleton", color_image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
