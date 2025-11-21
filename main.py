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

# ---------------- Load SpeechBrain model ----------------






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

# ---------------- TEXT EMOTION ENDPOINT ----------------
@app.post("/predict/text")
async def predict_text(input: TextInput):
    text = input.text.lower()
    emotion_keywords = {
        "happy": ["happy", "joyful", "cheerful", "delighted", "content", "glad", "pleased","great","excited"],
        "sad": ["sad", "unhappy", "depressed", "miserable", "down", "gloomy", "heartbroken", "melancholy"],
        "angry": ["angry", "mad", "furious", "irritated", "annoyed", "upset", "enraged"],
        "fear": ["fear", "afraid", "scared", "terrified", "nervous", "anxious", "worried"],
        "disgust": ["disgust", "disgusted", "revolted", "repulsed", "nauseated"],
        "surprise": ["surprise", "amazed", "astonished", "startled", "shocked"],
        "neutral": ["neutral", "fine", "okay", "normal", "nothing special"]
    }

    detected_emotion = "neutral"
    for emotion, keywords in emotion_keywords.items():
        if any(word in text for word in keywords):
            detected_emotion = emotion
            break

    genre = emotion_to_genre.get(detected_emotion, "pop")

    try:
        results = sp.search(q=f"{genre} playlist", type="playlist", limit=5)
        playlist_url = get_playlist_url(results) or "No playlist found"
    except Exception as e:
        print("Spotify error:", e)
        playlist_url = "Error fetching playlist"

    return {"emotion": detected_emotion.capitalize(), "playlist_url": playlist_url}

# ---------------- IMAGE EMOTION ENDPOINT ----------------
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    from tensorflow.keras.models import load_model #type: ignore
    from tensorflow.keras.preprocessing.image import img_to_array #type: ignore

    model = load_model("emotion_model_fixed.h5")
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        prediction = model.predict(roi, verbose=0)[0]
        emotion = emotion_labels[prediction.argmax()]
    else:
        emotion = "Neutral"

    genre = emotion_to_genre.get(emotion.lower(), "pop")
    try:
        results = sp.search(q=f"{genre} playlist", type="playlist", limit=5)
        playlist_url = get_playlist_url(results) or "No playlist found"
    except Exception as e:
        print("Spotify error:", e)
        playlist_url = "Error fetching playlist"

    return {"emotion": emotion, "playlist_url": playlist_url}


"""# ---------------- AUDIO EMOTION ENDPOINT ----------------
from pyAudioAnalysis import audioTrainTest as aT  # type: ignore
from pydub import AudioSegment  # type: ignore

@app.post("/predict/audio")
async def predict_audio(audio: UploadFile = File(...)):
    try:
        # Save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio.write(await audio.read())
            webm_path = temp_audio.name

        # Convert to WAV
        wav_path = webm_path.replace(".webm", ".wav")
        AudioSegment.from_file(webm_path).export(wav_path, format="wav")

        # Load emotion model (correct 4-class model)
    
        model_path = "/Users/akhilaa/mood_song_recommender/face_env/lib/python3.12/site-packages/pyAudioAnalysis/models/svm_rbf_4class"
        loaded = aT.load_model(model_path)

        classifier = loaded[0]
        mean = loaded[1]
        std = loaded[2]
        labels = loaded[3]

        # Predict emotion
        result, probabilities = aT.file_classification(wav_path, model_path, "svm")
        predicted_emotion = labels[int(result)]

        # Map emotion → playlist genre
        emotion_map = {
            "happiness": "pop",
            "anger": "metal",
            "sadness": "lofi",
            "fear": "ambient"
        }

        genre = emotion_map.get(predicted_emotion.lower(), "pop")

        # Spotify playlist search
        results = sp.search(q=f"{genre} playlist", type="playlist", limit=5)
        playlist_url = get_playlist_url(results)

        os.remove(webm_path)
        os.remove(wav_path)

        return {
            "emotion": predicted_emotion,
            "recommended_genre": genre,
            "playlist_url": playlist_url
        }

    except Exception as e:
        return {"error": f"Audio processing error: {str(e)}"}
        """
"""# ---------------- AUDIO EMOTION ENDPOINT ----------------
from pyAudioAnalysis import audioTrainTest as aT
from pydub import AudioSegment

@app.post("/predict/audio")
async def predict_audio(audio: UploadFile = File(...)):
    try:
        # Save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio.write(await audio.read())
            webm_path = temp_audio.name

        # Convert WEBM → WAV
        wav_path = webm_path.replace(".webm", ".wav")
        AudioSegment.from_file(webm_path).export(wav_path, format="wav")

        # Load 4-class model
        model_path = "/Users/akhilaa/mood_song_recommender/face_env/lib/python3.12/site-packages/pyAudioAnalysis/models/svm_rbf_4class"

        classifier, mean, std = aT.load_model(model_path)

        # ---- GET class names (IMPORTANT FIX) ----
        class_names = classifier.class_names if hasattr(classifier, "class_names") else classifier.classes_

        # ---- Predict emotion (ONLY TWO VALUES RETURNED) ----
        result, probabilities = aT.file_classification(wav_path, model_path, "svm")
        predicted_emotion = class_names[int(result)]

        # Emotion → Genre mapping
        emotion_map = {
            "happiness": "pop",
            "anger": "metal",
            "sadness": "lofi",
            "fear": "ambient"
        }
        genre = emotion_map.get(predicted_emotion.lower(), "pop")

        # Spotify playlist
        results = sp.search(q=f"{genre} playlist", type="playlist", limit=5)
        playlist_url = get_playlist_url(results)

        # Cleanup
        os.remove(webm_path)
        os.remove(wav_path)

        return {
            "emotion": predicted_emotion,
            "recommended_genre": genre,
            "playlist_url": playlist_url,
            "debug": {
                "class_names": class_names,
                "result": int(result),
                "probabilities": probabilities.tolist(),
            }
        }

    except Exception as e:
        return {"error": f"Audio processing error: {str(e)}"}
"""

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
