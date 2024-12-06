import cv2
import torch
import numpy as np
from pyneuphonic import Neuphonic, TTSConfig
from pyneuphonic.player import AudioPlayer
from queue import Queue
from threading import Thread
import time

class DetectionSystem:
    def __init__(self, api_key):
        self.model = self._initialize_model()
        self.sse, self.tts_config = self._setup_neuphonic(api_key)
        self.audio_queue = Queue()
        self.last_speech_time = {}  # Dictionary to track last speech time for each label
        self.min_speech_interval = 3  # Minimum seconds between repeated announcements
        self.is_speaking = False
        
        # Start audio processing thread
        self.audio_thread = Thread(target=self._process_audio_queue, daemon=True)
        self.audio_thread.start()

    def _initialize_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        return model

    def _setup_neuphonic(self, api_key):
        client = Neuphonic(api_key=api_key)
        sse = client.tts.SSEClient()
        # Adjusted speech settings for clearer pronunciation
        tts_config = TTSConfig(
            speed=1.2,  # Slowed down the speech
            voice='8e9c4bc8-3979-48ab-8626-df53befc2090',
            model="neu_hq"
        )
        return sse, tts_config

    def process_frame(self, frame, confidence_threshold=0.5):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            results = self.model(frame_rgb)
        
        detections = []
        for pred in results.xyxy[0]:
            if pred[4] >= confidence_threshold:
                x1, y1, x2, y2 = map(int, pred[:4])
                conf = float(pred[4])
                class_id = int(pred[5])
                label = self.model.names[class_id]
                detections.append((x1, y1, x2, y2, label, conf))
        
        return detections

    def _process_audio_queue(self):
        with AudioPlayer() as player:
            while True:
                if not self.audio_queue.empty() and not self.is_speaking:
                    text = self.audio_queue.get()
                    self.is_speaking = True
                    response = self.sse.send(text, tts_config=self.tts_config)
                    player.play(response)
                    time.sleep(0.5)  # Added small pause after each announcement
                    self.is_speaking = False
                    self.audio_queue.task_done()
                time.sleep(0.1)

    def should_announce(self, label):
        if self.is_speaking:
            return False
            
        current_time = time.time()
        if label not in self.last_speech_time:
            self.last_speech_time[label] = current_time
            return True
        
        if current_time - self.last_speech_time[label] >= self.min_speech_interval:
            self.last_speech_time[label] = current_time
            return True
        return False

def draw_detections(frame, detections):
    for x1, y1, x2, y2, label, conf in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} ({conf:.2f})"
        cv2.putText(frame, text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def main():
    api_key = "143e2f6bd9a8c80ef8ed940ab10cbae9630c8927dce524d65de85dcd1613f8a0.ec7715cc-37e6-4590-8265-e6528ec1125e"
    system = DetectionSystem(api_key)
    
    video_path = "NeuphonicHackathon/Screen Recording 2024-12-06 at 19.34.41.mov"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    process_every_n_frames = 3
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame. Exiting.")
            break
        
        frame_count += 1
        if frame_count % process_every_n_frames != 0:
            continue
            
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        detections = system.process_frame(frame)
        frame = draw_detections(frame, detections)
        
        # Only queue new audio if not currently speaking
        if not system.is_speaking:
            for detection in detections:
                label = detection[4]
                if system.should_announce(label):
                    # Modified announcement text for clarity
                    system.audio_queue.put(f"I can see {label}.")
        
        cv2.imshow("Real-time Object Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()