import cv2
import numpy as np
import base64
import mediapipe as mp
import math
import json
import time
import os 
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn

# =========================================================
# استدعاء دوال من core.py
# =========================================================
from core import (
    load_eye_model, load_yawn_model, extract_region_from_landmarks,
    get_combined_eye_prediction, get_mouth_prediction, draw_bbox,
    DecisionEngine, LEFT_EYE_IDXS, RIGHT_EYE_IDXS, MOUTH_IDXS, calculate_ear
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

eye_model = load_eye_model()
yawn_model = load_yawn_model()
engine = DecisionEngine()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

@app.get("/", response_class=HTMLResponse)
async def get_webpage(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    drowsy_count = 0
    is_drowsy = False
    drowsy_start_time = 0
    music_triggered = False

    try:
        while True:
            data = await websocket.receive_text()
            
            if data == "RESET":
                drowsy_count = 0
                is_drowsy = False
                drowsy_start_time = 0
                music_triggered = False
                engine.eye_closed_frames = 0
                print("System Reset: Counters are back to zero.")
                continue 

            header, encoded = data.split(",", 1)
            nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            fh, fw, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            eye_label, eye_conf = "open", 1.0
            yawn_label, yawn_conf = "no_yawn", 1.0
            distance_cm = 0
            mouth_ratio = 0.0

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                ear = calculate_ear(face_landmarks, frame.shape)

                lx, ly = int(face_landmarks.landmark[468].x * fw), int(face_landmarks.landmark[468].y * fh)
                rx, ry = int(face_landmarks.landmark[473].x * fw), int(face_landmarks.landmark[473].y * fh)
                pixel_distance = math.hypot(rx - lx, ry - ly)
                if pixel_distance > 0:
                    distance_cm = (6.3 * 600) / pixel_distance

                left_eye_crop, left_eye_bbox = extract_region_from_landmarks(frame, face_landmarks, LEFT_EYE_IDXS, 0.25)
                right_eye_crop, right_eye_bbox = extract_region_from_landmarks(frame, face_landmarks, RIGHT_EYE_IDXS, 0.25)
                mouth_crop, mouth_bbox = extract_region_from_landmarks(frame, face_landmarks, MOUTH_IDXS, 0.30)

                if left_eye_crop is not None and right_eye_crop is not None:
                    eye_label, eye_conf = get_combined_eye_prediction(eye_model, left_eye_crop, right_eye_crop)
                
                if ear > 0.28 and eye_label == "close":
                    eye_label = "open"

                if mouth_crop is not None:
                    yawn_label, yawn_conf = get_mouth_prediction(yawn_model, mouth_crop)
                    if mouth_bbox is not None:
                        x, y, w, h = mouth_bbox
                        mouth_ratio = h / w  
                        if yawn_label == "Yawn" and mouth_ratio < 0.40: 
                            yawn_label = "no_yawn"
                                
                        draw_bbox(frame, mouth_bbox, color=(255, 255, 0))
                        cv2.putText(frame, f"MAR: {mouth_ratio:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if left_eye_crop is not None: draw_bbox(frame, left_eye_bbox, color=(255, 0, 0))
                if right_eye_crop is not None: draw_bbox(frame, right_eye_bbox, color=(255, 0, 0))

            status, message = engine.update(eye_label, eye_conf, yawn_label, yawn_conf)

            trigger_music_now = False
            sleep_duration = 0.0 

            if status == "DROWSY":
                if not is_drowsy:
                    is_drowsy = True
                    drowsy_start_time = time.time()
                    drowsy_count += 1 
                
                sleep_duration = time.time() - drowsy_start_time
                
                if not music_triggered:
                    if sleep_duration >= 10.0 or drowsy_count >= 2:
                        trigger_music_now = True
                        music_triggered = True
                        print("AI Action: Triggering Emergency Music!")
            else:
                is_drowsy = False
                sleep_duration = 0.0

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            out_b64 = base64.b64encode(buffer).decode('utf-8')

            yawn_display = "Yes!" if yawn_label == "Yawn" else "No"
            
            response_data = {
                "image": "data:image/jpeg;base64," + out_b64,
                "status": status,
                "distance": int(distance_cm),
                "closed_frames": engine.eye_closed_frames,
                "yawning": yawn_display,
                "sleep_seconds": round(sleep_duration, 1),
                "trigger_music": trigger_music_now 
            }
            await websocket.send_text(json.dumps(response_data))

    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
  
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)



    #   uvicorn server:app --reload
    