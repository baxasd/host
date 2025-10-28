import cv2
import mediapipe as mp

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(model_complexity=1, smooth_landmarks=True)
        self.drawer = mp.solutions.drawing_utils

    
    def estimate(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb)
    
    def draw(self, image, results):
        if results.pose_landmarks:
            self.drawer.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

    