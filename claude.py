import cv2
import torch
import numpy as np
from pyneuphonic import Neuphonic, TTSConfig
from pyneuphonic.player import AudioPlayer

def initialize_model():
    # Load YOLO model from torchvision
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    return model

def setup_neuphonic(api_key):
    client = Neuphonic(api_key=api_key)
    sse = client.tts.SSEClient()
    tts_config = TTSConfig(
        speed=1.3,
        voice='8e9c4bc8-3979-48ab-8626-df53befc2090',
        model="neu_hq"
    )
    return sse, tts_config

def process_frame(frame, model, confidence_threshold=0.5):
    # Convert frame to RGB (YOLO expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform inference
    results = model(frame_rgb)
    
    # Get detections
    detections = []
    for pred in results.xyxy[0]:  # xyxy format: x1, y1, x2, y2, confidence, class
        if pred[4] >= confidence_threshold:
            x1, y1, x2, y2 = map(int, pred[:4])
            conf = float(pred[4])
            class_id = int(pred[5])
            label = model.names[class_id]
            detections.append((x1, y1, x2, y2, label, conf))
    
    return detections

def draw_detections(frame, detections):
    for x1, y1, x2, y2, label, conf in detections:
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label and confidence
        text = f"{label} ({conf:.2f})"
        cv2.putText(frame, text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main():
    # Initialize
    api_key = "143e2f6bd9a8c80ef8ed940ab10cbae9630c8927dce524d65de85dcd1613f8a0.ec7715cc-37e6-4590-8265-e6528ec1125e"  # Replace with your API key
    model = initialize_model()
    sse, tts_config = setup_neuphonic(api_key)
    
    # Open video capture
    video_path = "Screen Recording 2024-12-06 at 19.34.41.mov"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame. Exiting.")
            break
            
        # Resize frame for faster processing
        frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
        
        # Process frame
        detections = process_frame(frame, model)
        
        # Draw detections
        frame = draw_detections(frame, detections)
        
        # Audio feedback for detections
        for detection in detections:
            label, conf = detection[4], detection[5]
            text = f"{label}"
            with AudioPlayer() as player:
                response = sse.send(text, tts_config=tts_config)
                player.play(response)
        
        # Display the frame
        cv2.imshow("Real-time Object Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()