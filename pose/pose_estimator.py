import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

class PoseEstimator:
    """Pose Estimation with explicit GPU acceleration."""

    def __init__(self, model_path="./models/pose_landmarker_heavy.task"):
        try:
            # Explicitly enable GPU
            base_options = python.BaseOptions(
                model_asset_path=model_path,
                delegate=python.BaseOptions.Delegate.GPU  # Force GPU
            )
            
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=2,  # Increase if you need multiple people
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False
            )
            
            self.detector = vision.PoseLandmarker.create_from_options(options)
            print("[INFO] PoseLandmarker initialized with GPU acceleration.")
            
        except Exception as e:
            print(f"GPU initialization failed: {e}")
            print("Falling back to CPU...")
            # Fallback to CPU
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=2,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
            print("[INFO] PoseLandmarker initialized with CPU fallback.")

    def detect_pose(self, bgr_image, timestamp_ms=0):
        """GPU-accelerated pose detection."""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Perform detection
            result = self.detector.detect_for_video(mp_image, timestamp_ms)
            return result
            
        except Exception as e:
            print(f"Error in detect_pose: {e}")
            return None

    @staticmethod
    def draw_landmarks_on_image(rgb_image, detection_result):
        """Draw landmarks with connections."""
        annotated_image = np.copy(rgb_image)
        
        if not detection_result or not detection_result.pose_landmarks:
            return annotated_image

        pose_landmarks_list = detection_result.pose_landmarks
        
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            
            # Convert to protobuf format for drawing
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in pose_landmarks
            ])
            
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )
            
        return annotated_image

    def close(self):
        """Close the detector."""
        if hasattr(self, 'detector'):
            self.detector.close()