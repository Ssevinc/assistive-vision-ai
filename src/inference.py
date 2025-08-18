from ultralytics import YOLO
import sys

def run_inference(source="/Users/s.sevinc/visual-assistant/newyork.mp4", model_path="/Users/s.sevinc/visual-assistant/models/best.pt", conf=0.5):
    """
    Run inference using a YOLO model.

    Args:
        source (str): Source for inference, can be a video file path or camera index.
        model_path (str): Path to the YOLO model weights.
        conf (float): Confidence threshold for detections.

    Returns:
        None
    """
    try:
        # Load the YOLO model
        model = YOLO(model_path)
        
        # Run inference
        results = model.predict(source=source, conf=conf,save= True,show= True)
        
        # Print results
        print(results)
        
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    run_inference()