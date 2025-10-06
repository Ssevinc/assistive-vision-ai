from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path= "/Users/s.sevinc/visual-assistant/demo/models/best_float32.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_inference(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (input_details[0]['shape'], input_details[0]['shape'][1]))
    input_date = np.expand_dims(img_resized/255.0, axis=0).astype(np.float32)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data.tolist()

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/detect', methods = ['POST'])
def detect():
    data = request.json['image']
    img_data = base64.b64decode(data.split(','[1]))
    img = np.array(Image.open(BytesIO(img_data)))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    detections = run_inference(img_bgr)
    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

