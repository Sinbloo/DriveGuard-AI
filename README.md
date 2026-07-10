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


🚀 How to Run

1. Environment Setup

Clone the repository and install the required dependencies:

git clone https://github.com/YourUsername/DriveGuard-AI.git
cd DriveGuard-AI
pip install -r requirements.txt


2. Running the Local Desktop App (Driver View)

To launch the CustomTkinter dark-mode dashboard with local camera access:

# On Windows, you can double-click ui2.2.bat, or run:
python ui2.py


3. Running the Web Application (Fleet Manager View)

To start the FastAPI WebSocket server for live remote monitoring:

uvicorn server:app --host 0.0.0.0 --port 8000


Open a web browser and navigate to http://localhost:8000.

4. Cloud Deployment (Docker / Hugging Face Spaces)

The project is containerized and ready for cloud deployment. To build and run using Docker:

docker build -t driveguard-ai .
docker run -p 7860:7860 driveguard-ai


👥 Meet The Team (DEPI Project)

-Nour Gomaa

Team Lead & AI Architect

Designed system architecture, data strategy, reviewed models, managed GitHub, and executed final integration.

-Fatma Hesham

Data & Dataset Engineer

Handled dataset organization, cleaning, class balancing, augmentation, and PyTorch DataLoader pipelines.

-Moataz Amr

Computer Vision Engineer

Developed the Vision Detection Layer using MediaPipe/OpenCV to extract facial landmarks and ROIs.

-Abdelrahman Shawky

Deep Learning Engineer

Built, fine-tuned, and optimized the MobileNetV2 classification models, managing hyperparameter experiments.

-Mohamed Yahya

Real-Time Systems Eng.

Developed the CustomTkinter GUI, FastAPI backend, audio alert logic, and managed cloud deployment.

DEPI (Digital Egypt Pioneers Initiative) - Final Project Submission
