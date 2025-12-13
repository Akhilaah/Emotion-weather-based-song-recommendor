import os
import tempfile
import subprocess
from fastapi import FastAPI, UploadFile, File, Query #type: ignore
from fastapi.middleware.cors import CORSMiddleware #type: ignore
from pydantic import BaseModel #type: ignore
import cv2
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import torch
import torchaudio #type: ignore
import requests
import librosa #type: ignore
import io
import base64
import tensorflow as tf
import tensorflow_hub as hub #type: ignore
import numpy as np
import joblib

text_model = joblib.load("text_emotion_model.pkl")

print("Loading YAMNet model...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")


# Load official class names
class_map_path = tf.keras.utils.get_file(
    "yamnet_class_map.csv",
    "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
)
import csv
class_names = []
with open(class_map_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        class_names.append(row[2])   # <-- THIS is the correct column
# Map YAMNet classes → emotions
YAMNET_TO_EMOTION = {
    "laugh": "happy",
    "giggle": "happy",
    "cry": "sad",
    "sob": "sad",
    "whimper": "sad",
    "angry": "angry",
    "argument": "angry",
    "scream": "fear",
    "shout": "fear",
    "speech": "neutral",
    "conversation": "neutral"
}
def classify_emotion_from_yamnet(wav_path):
    # Load WAV file
   # Load WAV with TensorFlow only (no tfio)
    audio_binary = tf.io.read_file(wav_path)
    waveform, sr = tf.audio.decode_wav(audio_binary, desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1)


    # Run YAMNet
    scores, embeddings, spectrogram = yamnet_model(waveform)
    mean_scores = tf.reduce_mean(scores, axis=0)
    top_class = int(tf.argmax(mean_scores))
    predicted_class = class_names[top_class]

    # Map YAMNet label → emotion
    emotion = "neutral"
    for key in YAMNET_TO_EMOTION:
        if key.lower() in predicted_class.lower():
            emotion = YAMNET_TO_EMOTION[key]
            break

    return {
        "yamnet_class": predicted_class,
        "emotion": emotion
    }





# ---------------- Environment & Thread Settings ----------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "none"

# ---------------- FastAPI ----------------
app = FastAPI(title="Mood + Weather Song Recommender")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change to ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Spotify ----------------
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="b87f1c7848554a2cad9f133ea31b7b68",
    client_secret="ead04af1027b460890f0b9caee1bd61e",
    redirect_uri="http://127.0.0.1:8080/callback",
    scope="user-library-read playlist-read-private playlist-read-collaborative"
))

def get_playlist_url(results):
    if results and results.get('playlists') and results['playlists'].get('items'):
        for item in results['playlists']['items']:
            if item and "external_urls" in item and "spotify" in item["external_urls"]:
                return item["external_urls"]["spotify"]
    return None

# ---------------- Emotion & Weather Mappings ----------------
emotion_to_genre = {
    "happy": "pop",
    "calm": "acoustic",
    "sad": "lo-fi",
    "angry": "metal",
    "fear": "ambient",
    "disgust": "punk",
    "surprise": "dance",
    "neutral": "chill"
}

weather_to_genre = {
    "Clear": "pop",
    "Clouds": "indie",
    "Rain": "acoustic",
    "Thunderstorm": "rock",
    "Haze": "Ambient",
    "Drizzle": "chill",
    "Snow": "piano",
    "Mist": "ambient",
    "Fog": "lo-fi"
}



# ---------------- Face Detector ----------------
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---------------- Pydantic Models ----------------
class TextInput(BaseModel):
    text: str

class WeatherInput(BaseModel):
    weather: str
    emotion: str

# ---------------- WEATHER ENDPOINT ----------------
OPENWEATHER_API_KEY = "ceac93a247353ad23cf698fe6c29531d"

@app.get("/weather")
def get_weather(city: str = Query(..., description="City name")):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if data.get("cod") != 200:
        return {"error": "City not found"}

    weather_main = data["weather"][0]["main"]
    temp = data["main"]["temp"]
    genre = weather_to_genre.get(weather_main, "chill")

    try:
        results = sp.search(q=f"{genre} playlist", type="playlist", limit=5)
        playlist_url = get_playlist_url(results) or "No playlist found"
    except Exception as e:
        print("Spotify error:", e)
        playlist_url = "Error fetching playlist"

    return {
        "city": city,
        "weather": weather_main,
        "temperature": temp,
        "recommended_genre": genre,
        "playlist_url": playlist_url
    }

# -------- LOAD TRAINED TEXT MODEL --------


text_model = joblib.load("text_emotion_model.pkl")
text_vectorizer = joblib.load("text_vectorizer.pkl")

@app.post("/predict/text")
async def predict_text(input: TextInput):
    try:
        text = input.text

        # Transform text using the SAME vectorizer used during training
        x_vec = text_vectorizer.transform([text])

        # Predict the label
        prediction = text_model.predict(x_vec)[0]

        # Convert label → genre
        genre = emotion_to_genre.get(prediction.lower(), "chill")

        # Generate playlist
        results = sp.search(q=f"{genre} playlist", type="playlist", limit=5)
        playlist_url = get_playlist_url(results) or "No playlist found"

        return {
            "emotion": prediction,
            "genre": genre,
            "playlist_url": playlist_url
        }

    except Exception as e:
        return {"error": f"text prediction failed: {str(e)}"}

# ---------------- IMAGE EMOTION ENDPOINT ----------------
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    from deepface import DeepFace
    import numpy as np
    import cv2

    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # DeepFace emotion analysis
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

        emotion = result[0]["dominant_emotion"].capitalize()

        # Map emotion → genre
        genre = emotion_to_genre.get(emotion.lower(), "pop")

        # Spotify playlist
        results = sp.search(q=f"{genre} playlist", type="playlist", limit=5)
        playlist_url = get_playlist_url(results)

        return {
            "emotion": emotion,
            "playlist_url": playlist_url
        }

    except Exception as e:
        return {"error": str(e)}


    return {"emotion": emotion, "playlist_url": playlist_url}
       
@app.post("/analyze-audio")
async def analyze_audio(audio: UploadFile = File(...)):
    try:
        # Save incoming .webm file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio.write(await audio.read())
            webm_path = temp_audio.name

        # Convert WEBM → WAV for librosa/YAMNet
        wav_path = webm_path.replace(".webm", ".wav")
        command = [
            "ffmpeg", "-i", webm_path,
            "-ar", "16000", "-ac", "1",
            wav_path, "-y"
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # YAMNet classification
        result = classify_emotion_from_yamnet(wav_path)
        yam_class = result["yamnet_class"]
        predicted_emotion = result["emotion"]


        # Map emotion to genre
        genre = emotion_to_genre.get(predicted_emotion.lower(), "chill")

        # Get Spotify playlist
        results = sp.search(q=f"{genre} playlist", type="playlist", limit=5)
        playlist_url = get_playlist_url(results)

        # Cleanup
        os.remove(webm_path)
        os.remove(wav_path)

        return {
            "yamnet_class": yam_class,
            "emotion": predicted_emotion,
            "genre": genre,
            "playlistUrl": playlist_url
        }

    except Exception as e:
        return {"error": f"Audio processing failed: {str(e)}"}



# ---------------- WEATHER + EMOTION RECOMMENDATION ----------------
@app.post("/predict/weather")
async def recommend_by_weather(input: WeatherInput):
    weather = input.weather.lower()
    emotion = input.emotion.lower()
   

    weather_emotion_genres = {
        ("sunny", "happy"): "pop",
        ("sunny", "sad"): "indie",
        ("sunny", "neutral"): "acoustic",
        ("rain", "sad"): "lofi",
        ("rain", "happy"): "jazz",
        ("rain", "neutral"): "ambient",
        ("cloud", "sad"): "soul",
        ("cloud", "happy"): "rock",
        ("snow", "sad"): "piano",
        ("snow", "happy"): "electronic",
    }

    genre = weather_emotion_genres.get((weather, emotion), "chill")
    try:
        results = sp.search(q=f"{genre} playlist", type="playlist", limit=5)
        playlist_url = get_playlist_url(results) or "No playlist found"
    except Exception as e:
        print("Spotify error:", e)
        playlist_url = "Error fetching playlist"

    return {"weather": weather, "emotion": emotion, "playlist_url": playlist_url}
# ---------------- AUTO WEATHER + EMOTION COMBINED ----------------
@app.post("/weather/auto/emotion")
async def auto_weather_emotion(data: dict):
    lat = data["lat"]
    lon = data["lon"]
    emotion = data["emotion"].lower()
    

    # Get weather from OpenWeather
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    weather_data = response.json()
    weather_main = weather_data["weather"][0]["main"].lower()

    # Combined mapping (your logic)
    weather_emotion_genres = {
        ("sunny", "happy"): "pop",
        ("sunny", "sad"): "indie",
        ("sunny", "neutral"): "acoustic",
        ("rain", "sad"): "lofi",
        ("rain", "happy"): "jazz",
        ("rain", "neutral"): "ambient",
        ("clouds", "sad"): "soul",
        ("clouds", "happy"): "rock",
        ("snow", "sad"): "piano",
        ("snow", "happy"): "electronic",
    }

    # Choose playlist based on BOTH
    genre = weather_emotion_genres.get((weather_main, emotion), "chill")

    results = sp.search(q=f"{genre} playlist", type="playlist", limit=5)
    playlist_url = get_playlist_url(results)

    return {
        "weather": weather_main,
        "emotion": emotion,
        "genre": genre,
        "playlist_url": playlist_url
    }
