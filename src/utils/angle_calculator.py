"""Utilities for computing joint angles from 3D landmarks.

This module focuses on robust angle calculations in 3D by projecting
vectors onto the local plane of the joint triplet (a - b - c). This
helps reduce noise introduced by depth measurement errors.
"""

import numpy as np


class AngleCalculator:
    """Calculates joint angles in 3D with plane correction for better accuracy."""

    def __init__(self):
        # No state required for this stateless utility class
        pass

    @staticmethod
    def vector(a, b):
        """Return vector from point `a` to point `b` (3D).

        Expects `a` and `b` to be sequences of length 3. Using a small
        helper keeps the geometric code concise elsewhere.
        """
        return np.array([b[i] - a[i] for i in range(3)])

    @staticmethod
    def angle_between(v1, v2):
        """Return angle in degrees between two vectors.

        Uses a numerically safe arccos by clipping the cosine argument.
        Returns 0.0 for degenerate vectors.
        """
        dot = np.dot(v1, v2)
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        if mag1 * mag2 == 0:
            return 0.0
        cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    @staticmethod
    def project_to_plane(v, normal):
        """Project vector `v` onto the plane defined by `normal`.

        The normal is normalized inside the function. The projection is
        performed using standard vector projection formula.
        """
        normal = normal / np.linalg.norm(normal)
        return v - np.dot(v, normal) * normal

    def calculate(self, landmarks):
        """
        Compute a set of joint angles from 3D landmarks.

        Parameters
        - landmarks: dict mapping MediaPipe landmark id -> (X, Y, Z)

        Returns
        - dict of joint name -> angle in degrees or None if not computable
        """
        angles = {}

        # Use centralized landmark IDs from utils.landmarks for consistency
        from src.utils.landmarks import (
            LEFT_SHOULDER, RIGHT_SHOULDER,
            LEFT_ELBOW, RIGHT_ELBOW,
            LEFT_WRIST, RIGHT_WRIST,
            LEFT_HIP, RIGHT_HIP,
            LEFT_KNEE, RIGHT_KNEE,
            LEFT_ANKLE, RIGHT_ANKLE,
        )

        def safe_angle(a_id, b_id, c_id):
            """Calculate the angle at joint `b_id` using points a-b-c.

            Projects the two limb vectors onto the plane defined by the
            triangle (a-b-c) before computing the angle to improve
            robustness to out-of-plane noise.
            """
            try:
                a = landmarks.get(a_id)
                b = landmarks.get(b_id)
                c = landmarks.get(c_id)
                if a is None or b is None or c is None:
                    return None

                v1 = self.vector(b, a)
                v2 = self.vector(b, c)

                # Plane normal formed by the two vectors
                plane_normal = np.cross(v1, v2)
                if np.linalg.norm(plane_normal) == 0:
                    # Degenerate case (collinear points) — fall back to basic angle
                    return self.angle_between(v1, v2)

                # Project vectors onto the plane and compute angle between them
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
