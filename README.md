# Emotion-weather-based-song-recommendor
An intelligent, AI-powered music recommendation system that generates personalized playlists by analyzing user emotion and real-time weather conditions.This project combines computer vision, NLP, and weather intelligence to recommend music that matches both the user’s current mood and their environment. 
🚀 Overview

This project intelligently recommends songs by combining:

Emotion Detection (text, image, or real-time camera)

Weather Recognition (uses live weather API)

Music Mood Mapping (emotion + weather → playlist)

It enhances user experience by personalizing music based on mood and environment.

🛠️ Tech Stack
Frontend (React.js)

React + Hooks

Axios

Beautiful gradient UI

Backend (Python)

Flask/FastAPI (depending on your setup)

TensorFlow / PyTorch (emotion models)

OpenCV (image & camera emotion detection)

External APIs

OpenWeather API

 Spotify API

 UI Preview & Features
🌤️ Home Dashboard – Weather + Emotion Hub
<img src="/mnt/data/Screenshot 2025-11-21 at 8.21.39 PM.png" width="80%" />
😄 Text Emotion Detection

Detect emotion from typed text

Simple, interactive UI

Instant recommendations

<img src="/mnt/data/Screenshot 2025-11-21 at 8.21.47 PM.png" width="80%" />

📷 Real-time Camera Emotion Detection

Uses webcam feed

Predicts mood from face in real time

Auto-refreshing playlist

🖼️ Image Upload Emotion Detection

Upload any image

Model predicts dominant emotion

Playlist generated automatically

🎶 Weather-based Music Playlist

Enter city / enable location

Fetches temperature + condition

Maps weather → playlist mood

      ┌──────────────┐
      │  User Input   │
      │ (Text/Image/  │
      │  Camera/City) │
      └──────┬───────┘
             │
             ▼
      ┌──────────────┐
      │  React Front  │
      │     End       │
      └──────┬───────┘
             │Axios
             ▼
      ┌──────────────┐
      │  Python API   │
      ├──────────────┤
      │ Emotion Model │
      │ Weather API   │
      └──────┬───────┘
             │
             ▼
      ┌──────────────┐
      │ Playlist Gen  │
      └──────────────┘


Folder Structure
📁 mood_song_recommender
 ├── mood_song_frontend/   # React UI
 ├── app.py / main.py      # Backend
 ├── detect_emotion.py     # Image/Camera emotion model
 ├── spotify_test.py       # Playlist logic
 ├── utils/                # Helper functions
 ├── requirements.txt
 └── README.md


Installation & Setup
 Clone the repository
git clone https://github.com/Akhilaah/Emotion-weather-based-song-recommendor.git

Backend Setup
cd backend
pip install -r requirements.txt
python app.py

rontend Setup
cd mood_song_frontend
npm install
npm start

How It Works
1.User inputs mood via text, image, or camera
2.Weather API fetches real-time weather
3.Emotion + Weather → Category
4.System recommends playlist

Future Improvements
1.Spotify OAuth login
2.Save playlists for users
3.Mood history + analytics
4.Multi-lingual emotion detection
5.Add animations & dark mode
