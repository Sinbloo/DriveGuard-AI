import time
import cv2
import numpy as np
import torch
import mediapipe as mp

from PIL import Image
from torchvision import models, transforms

from config import (
    EYE_MODEL_PATH,
    YAWN_MODEL_PATH,
    EYE_CLASSES,
    YAWN_CLASSES,
    INPUT_SIZE,
    CONFIDENCE_THRESHOLD,
    EYE_CLOSED_FRAMES_FOR_DROWSY,
    YAWN_COOLDOWN_FRAMES,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Sound (Windows) =====
try:
    import winsound
    WINDOWS_SOUND = True
except Exception:
    WINDOWS_SOUND = False


def play_alert_sound():
    if WINDOWS_SOUND:
        try:
            winsound.Beep(1200, 400)
        except Exception:
            pass
    else:
        print("\a", end="", flush=True)


# ===== Landmark indices =====
LEFT_EYE_IDXS = [33, 133, 160, 159, 158, 157, 173, 246]
RIGHT_EYE_IDXS = [362, 263, 387, 386, 385, 384, 398, 390, 373, 374]
MOUTH_IDXS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# ===== Preprocess =====
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def preprocess_crop(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    tensor = transform(pil_img).unsqueeze(0)
    return tensor


def extract_region_from_landmarks(image, face_landmarks, landmark_indices, margin=0.20):
    h, w = image.shape[:2]
    xs, ys = [], []

    for idx in landmark_indices:
        lm = face_landmarks.landmark[idx]
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))

    if not xs or not ys:
        return None, None

    x1, x2 = max(min(xs), 0), min(max(xs), w - 1)
    y1, y2 = max(min(ys), 0), min(max(ys), h - 1)

    if x2 <= x1 or y2 <= y1:
        return None, None

    bw = x2 - x1
    bh = y2 - y1

    x1 = max(int(x1 - bw * margin), 0)
    y1 = max(int(y1 - bh * margin), 0)
    x2 = min(int(x2 + bw * margin), w - 1)
    y2 = min(int(y2 + bh * margin), h - 1)

    crop = image[y1:y2, x1:x2].copy()
    bbox = (x1, y1, x2, y2)
    return crop, bbox


def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2, label=None):
    if bbox is None:
        return frame

    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if label:
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


# ===== Model helpers =====
def clean_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v
    return cleaned


def build_mobilenetv2(num_classes=2):
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model


def load_model(weights_path, num_classes=2):
    checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=True)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
        model.to(DEVICE)
        model.eval()
        return model

    model = build_mobilenetv2(num_classes=num_classes)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
        state_dict = clean_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    model.to(DEVICE)
    model.eval()
    return model


def predict(model, image_tensor, class_names):
    if image_tensor is None:
        return None, 0.0, None

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
        label = class_names[pred_idx]

    return label, confidence, probs.detach().cpu().numpy()


def load_eye_model():
    return load_model(EYE_MODEL_PATH, num_classes=len(EYE_CLASSES))


def load_yawn_model():
    return load_model(YAWN_MODEL_PATH, num_classes=len(YAWN_CLASSES))


def get_combined_eye_prediction(model, left_crop, right_crop):
    probs_list = []

    for crop in [left_crop, right_crop]:
        tensor = preprocess_crop(crop)
        if tensor is None:
            continue
        _, _, probs = predict(model, tensor, EYE_CLASSES)
        if probs is not None:
            probs_list.append(probs)

    if not probs_list:
        return None, 0.0

    mean_probs = np.mean(np.array(probs_list), axis=0)
    pred_idx = int(np.argmax(mean_probs))
    pred_label = EYE_CLASSES[pred_idx]
    pred_conf = float(mean_probs[pred_idx])
    return pred_label, pred_conf


def get_mouth_prediction(model, mouth_crop):
    tensor = preprocess_crop(mouth_crop)
    if tensor is None:
        return None, 0.0

    label, conf, _ = predict(model, tensor, YAWN_CLASSES)

    if label == "Yawn" and conf < 0.60:
        return "no_yawn", 1.0 - conf 
    
    return label, conf

LEFT_EYE_EAR_IDXS  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDXS = [362, 385, 387, 263, 373, 380]

def calculate_ear(face_landmarks, image_shape):
    h, w = image_shape[:2]

    def _ear(indices):
        pts = []
        for i in indices:
            lm = face_landmarks.landmark[i]
            pts.append(np.array([lm.x * w, lm.y * h]))
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.0

    return (_ear(LEFT_EYE_EAR_IDXS) + _ear(RIGHT_EYE_EAR_IDXS)) / 2.0

class DecisionEngine:
    def __init__(
        self,
        eye_closed_frames_for_drowsy=EYE_CLOSED_FRAMES_FOR_DROWSY,
        yawn_cooldown_frames=YAWN_COOLDOWN_FRAMES,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    ):
        self.eye_closed_frames_for_drowsy = eye_closed_frames_for_drowsy
        self.yawn_cooldown_frames = yawn_cooldown_frames
        self.confidence_threshold = confidence_threshold

        self.eye_closed_frames = 0
        self.yawn_cooldown = 0

    def update(self, eye_label, eye_conf, yawn_label, yawn_conf):
        if self.yawn_cooldown > 0:
            self.yawn_cooldown -= 1

        if eye_label == "close" and eye_conf >= self.confidence_threshold:
            self.eye_closed_frames += 1
        else:
            self.eye_closed_frames = 0

        # Check drowsy FIRST — it's the more severe condition
        if self.eye_closed_frames >= self.eye_closed_frames_for_drowsy:
            return "DROWSY", "Drowsiness detected!"

        if (
            yawn_label == "Yawn"
            and yawn_conf >= self.confidence_threshold
            and self.yawn_cooldown == 0
        ):
            self.yawn_cooldown = self.yawn_cooldown_frames
            return "YAWN", "Yawn detected!"

        return "NORMAL", "Driving safely"


def draw_status(frame, status, message, eye_closed_frames):
    color_map = {
        "NORMAL": (0, 255, 0),
        "YAWN": (0, 165, 255),
        "DROWSY": (0, 0, 255),
        "NO FACE": (255, 255, 0),
    }
    color = color_map.get(status, (255, 255, 255))

    cv2.putText(frame, f"Status: {status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{message}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Closed-eye frames: {eye_closed_frames}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return frame