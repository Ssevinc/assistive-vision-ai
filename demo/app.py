from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import base64
from io import BytesIO
from PIL import Image
import pytesseract
import traceback

app = Flask(__name__)

MODELS = {
    "model_a": tf.lite.Interpreter(model_path="demo/models/best_float32.tflite"),
    "model_b": tf.lite.Interpreter(model_path="demo/models/yolov8n_float32.tflite")
}

# Allocate tensors once (faster runtime)
for interpreter in MODELS.values():
    interpreter.allocate_tensors()



# -----------------------------------------------------
#  HOME ROUTE (serves camera page)
# -----------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------------------------------
#  MAIN DETECTION ROUTE
# -----------------------------------------------------
@app.route("/detect", methods=["POST"])
def detect():
    try:
        # Get base64 image string
        data = request.json.get("image", None)
        if not data or "," not in data:
            return jsonify({"error": "Invalid image data"}), 400

        # Decode image
        img_data = base64.b64decode(data.split(",")[1])
        img = np.array(Image.open(BytesIO(img_data)))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Debug info
        print("📷 Frame received, bytes:", len(img_data))

        results = {}

        # -------------------------------------------------
        #  Run both TFLite models
        # -------------------------------------------------
        for name, interpreter in MODELS.items():
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Resize + normalize to match input shape
            h, w = input_details[0]["shape"][1:3]
            resized = cv2.resize(img_bgr, (w, h))
            input_data = np.expand_dims(resized / 255.0, axis=0).astype(np.float32)

            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]["index"])
            results[name] = np.array(output_data).tolist()

        # -------------------------------------------------
        #  OCR (text detection)
        # -------------------------------------------------
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        results["ocr_text"] = text.strip()

        # -------------------------------------------------
        #  Return combined results
        # -------------------------------------------------
        return jsonify(results)

    except Exception as e:
        print("Exception in /detect:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055)
