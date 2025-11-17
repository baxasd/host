"""Command-line entry point for the OST application.

Parses flags and forwards them to `run_system`.
"""

import argparse
from src.run import run_system


def main():
    """CLI arguments to run the application"""
    parser = argparse.ArgumentParser(
        description="3D Pose Tracking with Intel RealSense + MediaPipe"
    )
    # Argument to enable Kalman Smoothing filter
    parser.add_argument(
        "--use-kalman",
        action="store_true",
        help="Enable Kalman filter smoothing for joint coordinates."
    )
    # Argument to show the coordinates in image
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Display per-joint depth values in image"
    )

    # Argument to show the coordinates in image
    parser.add_argument(
        "--show-angles",
        action="store_true",
        help="Display per-joint angle values in image"
    )

    # Argument specifies model complexity
    parser.add_argument(
        "--model",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Pose model complexity: 0 (lite), 1 (full), 2 (heavy)",
    )

    args = parser.parse_args()

    # Call the main run loop with parsed options. Keep the call compact
    # so the CLI file remains a thin wrapper around the system core.
    run_system(use_kalman=args.use_kalman, show_depth=args.show_depth, show_angles=args.show_angles, model=args.model)


if __name__ == "__main__":
    main()
