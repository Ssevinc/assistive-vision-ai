import cv2
import time
import easyocr
import numpy as np
import subprocess
from ultralytics import YOLO

# ---- CONFIG ----
KEYWORDS = {"wc", "exit", "toilet", "market", "hospital"}   # OCR keywords
YOLO_COCO_TARGET_CLASSES = {"traffic light", "bench"}       # from pretrained YOLO
YOLO_CONF_THRESH = 0.7
OCR_CONF_THRESH = 0.6
FRAME_SKIP = 5
COOLDOWN_SEC = 20.0

IGNORED_CUSTOM_CLASSES = {"yellow light"}  # 👈 skip this class in custom model

# ---- OCR ----
reader = easyocr.Reader(['en'], gpu=False)

# ---- YOLO MODELS ----
coco_model = YOLO("yolo-Weights/yolov8n.pt")                  # pretrained COCO
custom_model = YOLO("/Users/s.sevinc/visual-assistant/models/best.pt")      # your trained model
coco_class_names = coco_model.names
custom_class_names = custom_model.names

# ---- macOS TTS ----
def say(msg: str, voice="Samantha"):
    subprocess.run(["/usr/bin/say", "-v", voice, "-o", "/tmp/tts.aiff", msg])
    subprocess.run(["afplay", "/tmp/tts.aiff"])

# ---- STATE ----
last_seen = {}

# ---- CAMERA ----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

frame_id = 0
fps_hist = []

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    t_start = time.time()

    frame_small = cv2.resize(frame, (frame.shape[1], frame.shape[0]))
    found_now = set()

    # --- OCR every FRAME_SKIP frames ---
    if frame_id % FRAME_SKIP == 0:
        results = reader.readtext(frame_small)
        for bbox, text, conf in results:
            text_norm = text.strip().lower()
            if conf < OCR_CONF_THRESH:
                continue
            match = None
            if text_norm in KEYWORDS:
                match = text_norm
            else:
                for kw in KEYWORDS:
                    if kw in text_norm:
                        match = kw
                        break
            if match:
                pts = [(int(x), int(y)) for x, y in bbox]
                cv2.polylines(frame_small, [np.array(pts)], True, (0,255,0), 2)
                cv2.putText(frame_small, text, (pts[0][0], pts[0][1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                now = time.time()
                if match not in last_seen or now - last_seen[match] > COOLDOWN_SEC:
                    last_seen[match] = now
                    found_now.add(match)

    # --- YOLO COCO (only specific classes) ---
    coco_results = coco_model(frame_small, stream=True)
    for r in coco_results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < YOLO_CONF_THRESH:
                continue
            cls_id = int(box.cls[0])
            label = coco_class_names[cls_id]
            if label not in YOLO_COCO_TARGET_CLASSES:
                continue

            # draw
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame_small, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(frame_small, label, (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            now = time.time()
            if label not in last_seen or now - last_seen[label] > COOLDOWN_SEC:
                last_seen[label] = now
                found_now.add(label)

    # --- YOLO CUSTOM (skip unwanted classes) ---
    custom_results = custom_model(frame_small, stream=True)
    for r in custom_results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < YOLO_CONF_THRESH:
                continue
            cls_id = int(box.cls[0])
            label = custom_class_names[cls_id]

            if label in IGNORED_CUSTOM_CLASSES:
                continue  # 👈 skip yellow light

            # draw
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame_small, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(frame_small, label, (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

            now = time.time()
            if label not in last_seen or now - last_seen[label] > COOLDOWN_SEC:
                last_seen[label] = now
                found_now.add(label)

    # --- FPS overlay ---
    fps = 1.0/max(1e-6, (time.time()-t_start))
    fps_hist.append(fps)
    if len(fps_hist) > 30:
        fps_hist.pop(0)
    fps_text = f"FPS: {sum(fps_hist)/len(fps_hist):.1f}"
    cv2.putText(frame_small, fps_text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
    cv2.putText(frame_small, fps_text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

    # --- TTS ---
    if found_now:
        msg = " | ".join(sorted(found_now))
        cv2.putText(frame_small, f"FOUND: {msg}", (10, 54),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 3)
        cv2.putText(frame_small, f"FOUND: {msg}", (10, 54),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 2)
        say(msg)

    # --- Show ---
    cv2.imshow("Webcam OCR+YOLO (multi-model, filtered)", frame_small)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
