from setuptools import setup, find_packages

setup(
    name="ost-realsense",
    version="0.1",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "opencv-python",
        "mediapipe",
        "numpy",
        "pyrealsense2",
        "filterpy"
    ],
    entry_points={
        "console_scripts": [
            "ost-realsense=src.cli_entry:main"
        ]
    },
)
