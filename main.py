import time
import cv2
import mediapipe as mp

from config import ALERT_SOUND_COOLDOWN_SECONDS
from core import (
    load_eye_model,
    load_yawn_model,
    extract_region_from_landmarks,
    get_combined_eye_prediction,
    get_mouth_prediction,
    draw_bbox,
    draw_status,
    play_alert_sound,
    DecisionEngine,
    LEFT_EYE_IDXS,
    RIGHT_EYE_IDXS,
    MOUTH_IDXS,
    calculate_ear,
)

def main():
    eye_model = load_eye_model()
    yawn_model = load_yawn_model()
    engine = DecisionEngine()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    last_alert_time = 0.0
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            status = "NO FACE"
            message = "Face not detected"

            eye_label = "open"
            eye_conf = 1.0
            yawn_label = "no_yawn"
            yawn_conf = 1.0
            ear = 0.0

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                ear = calculate_ear(face_landmarks, frame.shape)

                left_eye_crop, left_eye_bbox = extract_region_from_landmarks(
                    frame, face_landmarks, LEFT_EYE_IDXS, margin=0.25
                )
                right_eye_crop, right_eye_bbox = extract_region_from_landmarks(
                    frame, face_landmarks, RIGHT_EYE_IDXS, margin=0.25
                )
                mouth_crop, mouth_bbox = extract_region_from_landmarks(
                    frame, face_landmarks, MOUTH_IDXS, margin=0.30
                )

                if left_eye_crop is not None and right_eye_crop is not None:
                    eye_label, eye_conf = get_combined_eye_prediction(
                        eye_model, left_eye_crop, right_eye_crop
                    )
                if ear > 0.28 and eye_label == "close":
                        eye_label = "open"  # نعدلها بالعافية لمفتوحة
                        eye_conf = 1.0

                if mouth_crop is not None:
                    yawn_label, yawn_conf = get_mouth_prediction(yawn_model, mouth_crop)

                status, message = engine.update(eye_label, eye_conf, yawn_label, yawn_conf)

                if left_eye_crop is not None:
                    draw_bbox(frame, left_eye_bbox, color=(255, 0, 0), label="Left Eye")
                if right_eye_crop is not None:
                    draw_bbox(frame, right_eye_bbox, color=(255, 0, 0), label="Right Eye")
                if mouth_crop is not None:
                    draw_bbox(frame, mouth_bbox, color=(255, 255, 0), label="Mouth")

                if status in ["YAWN", "DROWSY"]:
                    now = time.time()
                    if now - last_alert_time >= ALERT_SOUND_COOLDOWN_SECONDS:
                        play_alert_sound()
                        last_alert_time = now

                cv2.putText(frame, f"Eye: {eye_label} ({eye_conf:.2f})", (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Yawn: {yawn_label} ({yawn_conf:.2f})", (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"EAR: {ear:.3f}", (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
                
            frame = draw_status(frame, status, message, engine.eye_closed_frames)

            cv2.imshow("Driver Drowsiness Detection", frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()