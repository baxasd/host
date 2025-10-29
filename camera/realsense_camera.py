import pyrealsense2 as rs
import numpy as np
import time
import sys

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30, verbose=False):
        self.verbose = verbose
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self.profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            if self.verbose:
                print(f"[INFO] RealSense camera started ({width}x{height}@{fps})")
        except Exception as e:
            print("Error initializing RealSense camera:", e)
            sys.exit(1)

    def get_frames(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                return None, None
            color_image = np.asanyarray(color_frame.get_data())
            time.sleep(0.01)  # prevent blocking
            return color_image, depth_frame
        except Exception as e:
            print("Error getting frames from RealSense camera:", e)
            return None, None

    def stop(self):
        try:
            self.pipeline.stop()
            if self.verbose:
                print("[INFO] RealSense pipeline stopped.")
        except Exception as e:
            print("Error stopping RealSense pipeline:", e)
