import cv2
import numpy as np
import webbrowser
import time
import os
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
import spotipy  # type: ignore
from spotipy.oauth2 import SpotifyOAuth  # type: ignore

# ---------------- Silence TensorFlow Warnings ----------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ---------------- Spotify setup ----------------
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="b87f1c7848554a2cad9f133ea31b7b68",
    client_secret="ead04af1027b460890f0b9caee1bd61e",
    redirect_uri="http://127.0.0.1:8080/callback",
    scope="user-library-read playlist-read-private"
))

# Map detected emotions to Spotify genres or moods
emotion_to_genre = {
    "Happy": "pop",
    "Sad": "acoustic",
    "Angry": "metal",
    "Fear": "ambient",
    "Disgust": "punk",
    "Surprise": "dance",
    "Neutral": "chill"
}

# ---------------- Load model ----------------
model = load_model("emotion_model_fixed.h5")

# Emotion labels (must match model’s output order)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ---------------- Load face detector ----------------
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---------------- Start webcam ----------------
cap = cv2.VideoCapture(1)  # Use 0 for built-in webcam
if not cap.isOpened():
    print("❌ Cannot open webcam.")
    exit()

print("✅ Webcam opened! Press 'q' to quit.")

current_emotion = None
last_opened_time = 0
playlist_cooldown = 15  # seconds between playlist openings

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame. Exiting...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            cv2.putText(frame, 'No Face Detected', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum(roi_gray) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = model.predict(roi, verbose=0)[0]
                    emotion = emotion_labels[prediction.argmax()]
                    label_position = (x, y - 10)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, emotion, label_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Open playlist only if emotion changed & cooldown passed
                    if emotion != current_emotion and (time.time() - last_opened_time > playlist_cooldown):
                        current_emotion = emotion
                        last_opened_time = time.time()
                        print(f"🎵 Detected Emotion: {emotion}")

                        genre = emotion_to_genre.get(emotion, "pop")
                        results = sp.search(q=f"{genre} playlist", type="playlist", limit=1)

                        if results['playlists']['items']:
                            playlist_url = results['playlists']['items'][0]['external_urls']['spotify']
                            print(f"▶️ Opening Spotify playlist for {emotion}: {playlist_url}")
                            webbrowser.open(playlist_url)
                        else:
                            print("⚠️ No Spotify playlist found for this emotion.")
                else:
                    cv2.putText(frame, 'No Face Found', (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('🎵 Emotion-Based Song Recommender', frame)

        # Quit cleanly
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("❌ 'q' pressed. Exiting...")
            break

except KeyboardInterrupt:
    print("\n🛑 Interrupted manually. Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam released, all windows closed.")
