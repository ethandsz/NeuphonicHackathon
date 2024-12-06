import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import urllib.request
import json
import numpy as np
from pyneuphonic import Neuphonic, TTSConfig, Agent
from pyneuphonic.player import AudioPlayer

api_key = "143e2f6bd9a8c80ef8ed940ab10cbae9630c8927dce524d65de85dcd1613f8a0.ec7715cc-37e6-4590-8265-e6528ec1125e" # GET THIS FROM beta.neuphonic.com!!!!!!!!!

client = Neuphonic(api_key=api_key)
voices = client.voices.get()  # get's all available voices

client = Neuphonic(api_key=api_key)
sse = client.tts.SSEClient()
tts_config = TTSConfig(speed=1.3,  voice='8e9c4bc8-3979-48ab-8626-df53befc2090', model="neu_hq")



# Load pre-trained AlexNet model
alexnet = models.alexnet(pretrained=True)
alexnet.eval()  # Set the model to evaluation mode

# Download ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels_path = "imagenet_labels.json"
urllib.request.urlretrieve(LABELS_URL, labels_path)

with open(labels_path, "r") as f:
    labels = json.load(f)

# Define preprocessing for frames
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert OpenCV image (NumPy array) to PIL image
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Helper: Convert raw model output to probabilities
def get_prediction_with_confidence(output):
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probabilities, 1)
    return confidence.item(), predicted_class.item()

# Sliding window function
def sliding_window(frame, step_size, window_size):
    for y in range(0, frame.shape[0] - window_size[1], step_size):
        for x in range(0, frame.shape[1] - window_size[0], step_size):
            yield (x, y, frame[y:y + window_size[1], x:x + window_size[0]])

# Video input (path to video file)
video_path = "Screen Recording 2024-12-06 at 19.34.41.mov"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

# Sliding window parameters
window_size = (64, 64)  # Window size for sliding
step_size = 32  # Step size for sliding

# Real-time video processing
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx = 0.1, fy = 0.1)
    print(frame.shape)
    if not ret:
        print("End of video or failed to read frame. Exiting.")
        break

    detections = []

    # Apply sliding window
    for (x, y, window) in sliding_window(frame, step_size, window_size):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            continue  # Skip if the window is incomplete (edge case)

        # Preprocess the window for AlexNet
        input_tensor = transform(window).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = alexnet(input_tensor)

        # Get predicted class and confidence
        confidence, predicted_class = get_prediction_with_confidence(output)
        if confidence > 0.5:  # Confidence threshold
            detections.append((x, y, labels[predicted_class], confidence))

    # Draw detections on the frame
    for (x, y, label, confidence) in detections:
        cv2.rectangle(frame, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
        text = f"{label} ({confidence * 100:.1f}%)"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        with AudioPlayer() as player:
            response = sse.send(f'{text}', tts_config=tts_config)
            player.play(response)

    # Display the frame
    cv2.imshow("Sliding Window Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
