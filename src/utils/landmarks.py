"""Centralized MediaPipe landmark ID constants.

Use these constants across the codebase to avoid duplication and
drift between modules.
"""

# Upper body
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

# Lower body
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Map joint names to landmark IDs for overlay/lookup in UI code
JOINT_TO_LANDMARK_ID = {
    'left_elbow': LEFT_ELBOW,
    'right_elbow': RIGHT_ELBOW,
    'left_shoulder': LEFT_SHOULDER,
    'right_shoulder': RIGHT_SHOULDER,
    'left_hip': LEFT_HIP,
    'right_hip': RIGHT_HIP,
    'left_knee': LEFT_KNEE,
    'right_knee': RIGHT_KNEE,
}
