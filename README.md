# 🚗 DriveGuard AI: Intelligent Driver Safety & Monitoring System
![Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
DriveGuard AI is an advanced, real-time hybrid monitoring system that utilizes Computer Vision and
Deep Learning to track facial landmarks, Eye Aspect Ratio (EAR), and yawning frequency to detect
micro-sleeps and cognitive distraction instantly. Unlike purely cloud-dependent systems, it processes
video locally on edge devices for zero-latency alerts while simultaneously connecting to a live web
dashboard for remote fleet monitoring.
---
## ✨ Key Features
- **Real-Time Inference:** Zero-latency detection of drowsiness and fatigue using optimized
MobileNetV2 models.
- **Hybrid Validation Logic:** Combines Neural Network predictions with geometric calculations
(EAR/MAR) to eliminate false positives.
- **Smart Auto-Tracking:** Dynamically zooms and focuses on the driver's face based on physical
distance estimation.
- **AI Mitigation System:** Automatically intervenes during severe drowsiness by playing
wakefulness audio/media.
- **Dual-Interface System:** Features a Dark Mode Desktop App (Driver View) and a
WebSocket-powered Web Dashboard (Fleet Manager View).
---

## ️ Project Architecture
1. **Computer Vision Layer:** MediaPipe Face Mesh & OpenCV.
2. **Deep Learning Layer:** PyTorch (MobileNetV2 Transfer Learning).
3. **Backend & Streaming:** FastAPI & WebSockets.
4. **Desktop GUI:** CustomTkinter.
---

## 🚀 How to Run
### 1. Environment Setup

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/DriveGuard-AI.git
cd DriveGuard-AI
pip install -r requirements.txt
```
### 2. Audio Assets for AI Mitigation (Important ⚠️)
The AI Mitigation system requires external audio files (e.g., wakefulness tracks, Quran) to trigger
during emergencies.
- Download the required audio assets from **[(https://drive.google.com/drive/folders/1QSAVcn-BSBvrmMoBSnGCsg_Tfz8j85KH?usp=sharing)]**.
- Extract the downloaded files and place them inside the `assets/audio/` folder before running the
application.
### 3. Running the Local Desktop App (Driver View)
Run the batch file or execute the Python script to launch the local monitoring interface:
```bash
python ui2.py
```
*(Or double-click `ui2.2.bat` on Windows)*
### 4. Running the Web Server (Fleet Manager View)
Start the FastAPI server for the live dashboard:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```
Open a web browser and navigate to `http://localhost:8000`.
---
## 👥 Team Members
- **Nour Mohamed** - Team Lead & AI Architect
- **Fatma Hesham** - Data & Dataset Engineer
- **Moataz Amr** - Computer Vision Engineer
- **Abdelrahman Shawky** - Deep Learning Engineer
- **Mohamed Yahya** - Real-Time Systems Engineer

DEPI (Digital Egypt Pioneers Initiative) - Final Project Submission
