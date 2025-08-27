import cv2
import time
import easyocr
import numpy as np
import os

# ---- CONFIG ----
KEYWORDS = {"wc", "exit", "toilet", "market", "hospital", "seyyide"}
FRAME_SKIP = 5
COOLDOWN_SEC = 3.0
CONF_THRESH = 0.6

# ---- OCR ----
# gpu=False because you're on macOS without CUDA (MPS not fully supported by EasyOCR yet)
reader = easyocr.Reader(['en','tr'], gpu=False)

# ---- macOS TTS ----
import subprocess

def say(msg: str, voice: str = "Samantha"):
    # Generate temporary audio file with macOS TTS
    subprocess.run(["/usr/bin/say", "-v", voice, "-o", "temp.aiff", msg])
    # Play it
    subprocess.run(["afplay", "temp.aiff"])

# ---- STATE ----
last_seen = {kw: 0.0 for kw in KEYWORDS}

# ---- CAMERA LOOP ----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

frame_id = 0
fps_hist = []

print("Press 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_id += 1
    t_start = time.time()

    # optional resize
    h, w = frame.shape[:2]
    scale = 1.0   # keep full size for clarity
    frame_small = cv2.resize(frame, (int(w*scale), int(h*scale)))

    found_now = set()
    if frame_id % FRAME_SKIP == 0:
        results = reader.readtext(frame_small)  # [(bbox, text, conf), ...]

        for bbox, text, conf in results:
            text_norm = text.strip().lower()
            if conf < CONF_THRESH:
                continue

            # check keywords
            match = None
            if text_norm in KEYWORDS:
                match = text_norm
            else:
                for kw in KEYWORDS:
                    if kw in text_norm:
                        match = kw
                        break

            # draw if keyword matched
            if match:
                pts = [(int(x), int(y)) for x, y in bbox]
                cv2.polylines(frame_small, [np.array(pts)], True, (0, 255, 0), 2)
                cv2.putText(frame_small, text, (pts[0][0], pts[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                now = time.time()
                if now - last_seen[match] >= COOLDOWN_SEC:
                    last_seen[match] = now
                    found_now.add(match)

    # show fps
    fps = 1.0 / max(1e-6, (time.time() - t_start))
    fps_hist.append(fps)
    if len(fps_hist) > 30:
        fps_hist.pop(0)
    fps_text = f"FPS: {sum(fps_hist) / len(fps_hist):.1f}"
    cv2.putText(frame_small, fps_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame_small, fps_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)

    if found_now:
        msg = " | ".join(sorted(found_now))
        cv2.putText(frame_small, f"FOUND: {msg}", (10, 54),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_small, f"FOUND: {msg}", (10, 54),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        say(msg)  # <-- macOS speech here

    cv2.imshow("Webcam EasyOCR (keywords only)", frame_small)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

