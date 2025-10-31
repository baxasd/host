import cv2
import numpy as np
from mediapipe.tasks import python as mp
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat, solutions
from mediapipe.framework.formats import landmark_pb2

class PoseEstimator:

    """ Class for Pose Estimation. 
    Google's Mediapipe Python algorithm to detect body landmarks """

    def __init__(self, model_path="./models/pose_landmarker_full.task"):
        """ Initializes Mediapipe """

        try:

            base_options = mp.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_segmentation_masks=False
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
            print("[INFO] PoseLandmarker initialized successfully")

        except Exception as e:
            print("Error initializing PoseEstimator:", e)
            raise e

    
    def estimate(self, image):
        """Performs pose estimation on a BGR image"""
        try:

            mp_image = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            result = self.detector.detect(mp_image)
            return result
        
        except Exception as e:

            print("Error in pose estimation:", e)
            return None

    def draw_landmarks_on_image(rgb_image, detection_result):
        """Draws pose landmarks with connections on the image."""
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        for pose_landmarks in pose_landmarks_list:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        return annotated_image
