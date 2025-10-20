# 🦴 HOST – Human Osteo-Skeletal Tracking

**HOST (Human Osteo-Skeletal Tracking)** is an experimental project that combines **Intel RealSense** depth sensing with **MediaPipe Pose** to detect, track, and record human skeletal movement in real time.  
It provides a foundation for motion-analysis, biomechanics research, and later machine-learning integration.

---

## 🎯 Project Goals
- Capture synchronized **color + depth** streams from an Intel RealSense camera.  
- Apply **MediaPipe Pose** to extract 33 human landmarks (x, y, z, visibility).  
- Store the processed data in **CSV** format for further analysis.  
- Build an extendable base for real-time skeleton analytics and visualization.

---

## 🧩 Tech Stack
| Component | Purpose |
|------------|----------|
| **Python 3.10+** | Main programming language |
| **MediaPipe** | Pose estimation / skeleton tracking |
| **OpenCV** | Frame handling and visualization |
| **pyrealsense2** | Access Intel RealSense camera streams |
| **NumPy & CSV** | Numerical data and export |

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/HOST.git
cd HOST
