import os
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir())

import cv2
from fer.fer import FER #type: ignore
import os

# Get the folder of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = "face_converted.jpg"



# Load the image
image = cv2.imread(image_path)

if image is None:
    print(f"Image not found! Check the path: {image_path}")
else:
    # Create FER detector
    detector = FER()

    # Detect emotions
    emotions = detector.detect_emotions(image)

    # Print results
    print(emotions)

