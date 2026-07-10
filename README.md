# DriveGuard-AI
🚗 DriveGuard AI: Intelligent Driver Safety & Monitoring System

📌 Project Overview

Driver fatigue and distraction are leading causes of severe road accidents globally. DriveGuard AI is an advanced, real-time hybrid monitoring system that utilizes Computer Vision and Deep Learning to track facial landmarks, Eye Aspect Ratio (EAR), and yawning frequency to detect micro-sleeps and cognitive distraction instantly.

Unlike purely cloud-dependent systems, it processes video locally on edge devices for zero-latency alerts while simultaneously connecting to a live web dashboard for remote fleet monitoring.

✨ Key Features

👀 Real-Time Detection: Accurately identifies driver drowsiness (closed eyes) and fatigue (yawning) within milliseconds.

⚡ Zero-Latency Alerts: Provides immediate in-cabin audio-visual warnings to the driver to prevent accidents.

🧠 Hybrid Validation Pipeline: Minimizes false positives by combining MobileNetV2 Neural Network predictions with geometric calculations (EAR/MAR).

🌐 Remote Fleet Management: Enables logistics managers to monitor vehicle safety states remotely via a WebSocket-powered web dashboard.

🎯 Smart Auto-Tracking: Maintains facial focus dynamically based on exact distance estimation from the camera.

🎵 AI Mitigation: Automatically launches custom audio content (e.g., energetic music, wakefulness frequencies) if the driver falls asleep for an extended period.

🛠️ Tech Stack

AI & Deep Learning: PyTorch, Torchvision, MobileNetV2

Computer Vision: OpenCV, MediaPipe Face Mesh

Backend & Streaming: FastAPI, Uvicorn, WebSockets

Frontend / GUI: CustomTkinter (Desktop), HTML5 Canvas & Web Audio API (Web)

Deployment: Docker, Hugging Face Spaces, Git LFS

📁 Project Structure

DriveGuard_Project/
├── models/
│   ├── eye_model_best.pth      # Pretrained weights for eye detection
│   └── yawn_model_best.pth     # Pretrained weights for yawn detection
├── templates/
│   └── index.html              # Frontend HTML for web streaming
├── core.py                     # Core AI logic, loading models, inference, EAR/MAR
├── server.py                   # FastAPI backend & WebSocket handler
├── ui2.py                      # Local desktop GUI (CustomTkinter)
├── config.py                   # Global configuration thresholds
├── Dockerfile                  # Cloud deployment configuration
├── requirements.txt            # Python dependencies
└── ui2.2.bat                   # Batch execution script for Windows


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
- Download the required audio assets from **[](https://drive.google.com/drive/folders/1QSAVcn-BSBvrmMoBSnGCsg_Tfz8j85KH?usp=sharing)]**.
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
