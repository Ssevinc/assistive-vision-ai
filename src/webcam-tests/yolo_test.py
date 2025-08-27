from ultralytics import YOLO
import cv2
import math
import time
import subprocess

# ---- CONFIG ----
MODEL_PATH = "/Users/s.sevinc/visual-assistant/models/best.pt"
COOLDOWN_SEC = 20.0  # only repeat after 5 seconds

# ---- TTS (macOS) ----
def say(msg: str, voice="Samantha"):
    subprocess.run(["/usr/bin/say", "-v", voice, "-o", "/tmp/tts.aiff", msg])
    subprocess.run(["afplay", "/tmp/tts.aiff"])

# ---- Load YOLO ----
model = YOLO(MODEL_PATH)

# ---- Class names ----
classNames = model.names 

# ---- Track last time each class was spoken ----
last_seen = {}

# ---- Start webcam ----
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("Press 'q' to quit.")
while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)

    # loop through detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = float(box.conf[0])
            if confidence < 0.5:  # skip low-confidence detections
                continue

            # class name
            cls = int(box.cls[0])
            label = classNames[cls]

            # put text on frame
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # say class name if cooldown passed
            now = time.time()
            if label not in last_seen or now - last_seen[label] > COOLDOWN_SEC:
                last_seen[label] = now
                say(label)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
