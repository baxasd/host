from setuptools import setup, find_packages

setup(
    name="pose-realsense",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "mediapipe",
        "numpy",
        "pyrealsense2",
        "filterpy"
    ],
    entry_points={
        "console_scripts": [
            "pose-realsense=pose_realsense.__main__:main"
        ]
    },
)
