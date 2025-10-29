import cv2
import mediapipe as mp

class PoseEstimator:
    def __init__(self):
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(model_complexity=1, smooth_landmarks=True)
            self.drawer = mp.solutions.drawing_utils
        except Exception as e:
            print("Error initializing PoseEstimator:", e)
            raise e

    
    def estimate(self, image):

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return self.pose.process(rgb)
        except Exception as e:
            print("Error in pose estimation:", e)
            return None
    
    def draw(self, image, results):
        try:
            if results.pose_landmarks:
                self.drawer.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        except Exception as e:
            print("Error drawing pose landmarks:", e)

    