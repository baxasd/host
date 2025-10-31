import argparse
from src.main import run_system

def main():
    parser = argparse.ArgumentParser(
        description="3D Pose Tracking with Intel RealSense + MediaPipe"
    )
    parser.add_argument(
        "--use-kalman",
        action="store_true",
        help="Enable Kalman filter smoothing for joint coordinates."
    )
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Display per-joint depth values in console (optional)."
    )
    args = parser.parse_args()

    run_system(use_kalman=args.use_kalman, show_depth=args.show_depth)

if __name__ == "__main__":
    main()
