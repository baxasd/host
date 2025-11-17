import numpy as np
import math

class AngleCalculator:
    """Calculates joint angles in 3D with plane correction for better accuracy."""

    def __init__(self):
        pass

    @staticmethod
    def vector(a, b):
        """Vector from point a to b."""
        return np.array([b[i] - a[i] for i in range(3)])

    @staticmethod
    def angle_between(v1, v2):
        """Return angle in degrees between two vectors."""
        dot = np.dot(v1, v2)
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        if mag1 * mag2 == 0:
            return 0.0
        cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    @staticmethod
    def project_to_plane(v, normal):
        """Project vector v onto plane defined by normal."""
        normal = normal / np.linalg.norm(normal)
        return v - np.dot(v, normal) * normal

    def calculate(self, landmarks):
        """
        landmarks: dict {id: (X, Y, Z)}
        Returns a dictionary of joint angles: elbows, shoulders, hips, knees
        """
        angles = {}

        # MediaPipe landmark IDs
        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
        LEFT_ELBOW, RIGHT_ELBOW = 13, 14
        LEFT_WRIST, RIGHT_WRIST = 15, 16
        LEFT_HIP, RIGHT_HIP = 23, 24
        LEFT_KNEE, RIGHT_KNEE = 25, 26
        LEFT_ANKLE, RIGHT_ANKLE = 27, 28

        def safe_angle(a_id, b_id, c_id):
            """Calculate angle at joint b, projecting onto the plane formed by a-b-c."""
            try:
                a = landmarks.get(a_id)
                b = landmarks.get(b_id)
                c = landmarks.get(c_id)
                if a is None or b is None or c is None:
                    return None

                v1 = self.vector(b, a)
                v2 = self.vector(b, c)

                # Plane normal
                plane_normal = np.cross(v1, v2)
                if np.linalg.norm(plane_normal) == 0:
                    return self.angle_between(v1, v2)  # straight line case

                # Project vectors onto the plane
                v1_proj = self.project_to_plane(v1, plane_normal)
                v2_proj = self.project_to_plane(v2, plane_normal)

                return self.angle_between(v1_proj, v2_proj)

            except KeyError:
                return None

        # Elbows
        angles['left_elbow'] = safe_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        angles['right_elbow'] = safe_angle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)

        # Shoulders
        angles['left_shoulder'] = safe_angle(LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP)
        angles['right_shoulder'] = safe_angle(RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP)

        # Knees
        angles['left_knee'] = safe_angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
        angles['right_knee'] = safe_angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)

        # Hips
        angles['left_hip'] = safe_angle(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
        angles['right_hip'] = safe_angle(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)

        return angles
