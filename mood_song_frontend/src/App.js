import { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./App.css";
import ImageInput from "./components/ImageInput";

function App() {
  const [text, setText] = useState("");
  const [city, setCity] = useState("");
  const [lat, setLat] = useState(null);
  const [lon, setLon] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [detectedEmotion, setDetectedEmotion] = useState("");
  const [playlistUrl, setPlaylistUrl] = useState("");
  const [currentWeather, setCurrentWeather] = useState(null);
  const [currentTemp, setCurrentTemp] = useState(null);
  const [weatherLoading, setWeatherLoading] = useState(true);
  // 🎙 AUDIO STATES
  const [recordStatus, setRecordStatus] = useState("idle"); // idle | recording | uploading
  const [audioEmotion, setAudioEmotion] = useState("");
  const [audioPlaylistUrl, setAudioPlaylistUrl] = useState("");
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);


  console.log("User location:", { lat, lon });

  /* ---------------------------------------------------
        ⭐ AUTO GET USER LOCATION (SAFE VERSION)
  ---------------------------------------------------- */
  useEffect(() => {
    if (!("geolocation" in navigator)) {
      alert("Your browser does not support location access.");
      setWeatherLoading(false);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const latitude = pos.coords.latitude;
        const longitude = pos.coords.longitude;

        setLat(latitude);
        setLon(longitude);

        console.log("📍 Location detected:", pos.coords);

        try {
          const res = await axios.get(
            `http://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&appid=ceac93a247353ad23cf698fe6c29531d&units=metric`
          );

          setCurrentWeather(res.data.weather[0].main);
          setCurrentTemp(res.data.main.temp);
        } catch (err) {
          console.error("Weather fetch failed:", err);
        }

        setWeatherLoading(false);
      },
      (err) => {
        console.error("Location error:", err);

        if (err.code === 1) {
          alert("Please allow location access for auto weather-based playlist.");
        } else {
          alert("Unable to access your location.");
        }

        setWeatherLoading(false);
      }
    );
  }, []);

  /* ---------------------------------------------------
        ⭐ Weather + Emotion Combined Request
  ---------------------------------------------------- */
  const getWeatherEmotionPlaylist = async (emotion) => {
    if (!lat || !lon) {
      console.log("⚠ Waiting for location...");
      return;
    }

    try {
      const res = await axios.post("http://127.0.0.1:8000/weather/auto/emotion", {
        lat,
        lon,
        emotion,
      });

      console.log("🌤 + 😄 Combined Response:", res.data);

      setDetectedEmotion(`Emotion: ${emotion} | Weather: ${res.data.weather}`);
      setPlaylistUrl(res.data.playlist_url);
    } catch (err) {
      console.error(err);
      alert("Weather+Emotion playlist fetch failed.");
    }
  };

  const handleImageResult = (emotion, url) => {
    setDetectedEmotion(emotion);
    setPlaylistUrl(url);
  };

  // ---------------- TEXT EMOTION ----------------
  const handleTextSubmit = async () => {
    if (!text) return;

    try {
      const res = await axios.post("http://127.0.0.1:8000/predict/text", { text });
      const emotion = res.data.emotion;
      setDetectedEmotion(emotion);
      getWeatherEmotionPlaylist(emotion);
    } catch {
      alert("Text prediction failed");
    }
  };
  
  
  const startRecording = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    const options = { mimeType: "audio/webm; codecs=opus" };
    const recorder = new MediaRecorder(stream, options);

    mediaRecorderRef.current = recorder;
    chunksRef.current = [];

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    recorder.onstop = async () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm; codecs=opus" });
      console.log("🎤 Blob size:", blob.size);

      if (blob.size === 0) {
        alert("Audio capture failed — blob is empty!");
        setRecordStatus("idle");
        return;
      }

      await uploadAudio(blob);
    };

    recorder.start(200); // <-- IMPORTANT
    console.log("🎙️ Recording started...");
    setRecordStatus("recording");

  } catch (err) {
    console.error(err);
    alert("Please allow microphone access.");
  }
};


const stopRecording = () => {
  if (mediaRecorderRef.current) {
    console.log("🛑 Recording stopped...");
    mediaRecorderRef.current.stop();
    setRecordStatus("uploading");
  }
};


const uploadAudio = async (blob) => {
  try {
    const formData = new FormData();
    formData.append("audio", blob, "recording.webm");

    console.log("⬆️ Uploading blob size:", blob.size);

    const res = await axios.post("http://127.0.0.1:8000/analyze-audio", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });

    console.log("📥 Backend Response:", res.data);

    setAudioEmotion(res.data.emotion);
    setAudioPlaylistUrl(res.data.playlistUrl);
    setRecordStatus("idle");

  } catch (err) {
    console.error("Audio upload error:", err);
    setRecordStatus("idle");
  }
};

  // ---------------- IMAGE UPLOAD EMOTION ----------------
  const handleImageSubmit = async () => {
    if (!selectedImage) return;

    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      const res = await axios.post("http://127.0.0.1:8000/predict/image", formData);
      const emotion = res.data.emotion;
      setDetectedEmotion(emotion);
      getWeatherEmotionPlaylist(emotion);
    } catch {
      alert("Image prediction failed");
    }
  };

  // ---------------- WEATHER ONLY ----------------
  const handleWeatherSubmit = async () => {
    if (!city) return;
    try {
      const res = await axios.get(`http://127.0.0.1:8000/weather?city=${city}`);
      setDetectedEmotion(`Weather: ${res.data.weather}`);
      setPlaylistUrl(res.data.playlist_url);
    } catch {
      alert("Weather fetch failed");
    }
  };

  return (
    <div className="App">
      <header>
        <h1>🎵 Mood & Weather Song Recommender</h1>
        <p>Detect emotions from text, image, or real-time camera</p>
      </header>

      {/* ⭐⭐⭐ WEATHER WIDGET ⭐⭐⭐ */}
      <div className="card fade-in" style={{ textAlign: "center" }}>
        <h2>🌦 Current Weather</h2>

        {weatherLoading ? (
          <p>Fetching your weather...</p>
        ) : currentWeather ? (
          <>
            <p style={{ fontSize: "1.3rem", margin: "5px 0" }}>
              <strong>{currentWeather}</strong>
            </p>
            <p style={{ fontSize: "1.1rem", opacity: 0.8 }}>
              Temperature: <strong>{currentTemp}°C</strong>
            </p>
          </>
        ) : (
          <p>Unable to fetch weather</p>
        )}
      </div>

      {/* TEXT */}
      <div className="card">
        <h2>Text Emotion</h2>
        <input
          type="text"
          placeholder="Type your mood..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button onClick={handleTextSubmit}>
          <span></span> Detect Emotion
        </button>
      </div>

      {/* CAMERA */}
      <div className="card">
        <h2>Real-Time Camera Emotion</h2>
        <ImageInput onResult={handleImageResult} />
      </div>

      
      {/* AUDIO EMOTION */}
<div className="card">
  <h2>🎙 Audio Emotion</h2>

  {recordStatus === "idle" && (
    <button onClick={startRecording}>
      🎤 Start Recording
    </button>
  )}

  {recordStatus === "recording" && (
    <button onClick={stopRecording} style={{ background: "red" }}>
      ⏹ Stop Recording
    </button>
  )}

  {recordStatus === "uploading" && <p>Analyzing your audio...</p>}

  {audioEmotion && (
    <p>
      <strong>Emotion:</strong> {audioEmotion}
    </p>
  )}

  {audioPlaylistUrl && (
    <a
      href={audioPlaylistUrl}
      target="_blank"
      rel="noreferrer"
      className="lavender-button"
      style={{
        display: "inline-block",
        marginTop: "10px",
        padding: "12px 25px",
        borderRadius: "30px",
        background: "linear-gradient(90deg, #84fab0, #8fd3f4)",
        color: "white",
        textDecoration: "none",
        fontWeight: "500"
      }}
    >
      🎧 Open Audio-Based Playlist
    </a>
  )}
</div>




      {/* IMAGE UPLOAD */}
      <div className="card">
        <h2>Upload Image</h2>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setSelectedImage(e.target.files[0])}
        />
        <button onClick={handleImageSubmit}>
          <span></span> Detect Emotion
        </button>
      </div>

      {/* WEATHER (optional manual entry) */}
      <div className="card">
        <h2>Weather Playlist</h2>
        <input
          type="text"
          placeholder="Enter city..."
          value={city}
          onChange={(e) => setCity(e.target.value)}
        />
        <button onClick={handleWeatherSubmit}>
          <span></span> Get Playlist
        </button>
      </div>

      {/* RESULT */}
      {detectedEmotion && (
        <div className="card fade-in">
          <h2>Result</h2>
          <p>
            <strong>Detected:</strong> {detectedEmotion}
          </p>

          {playlistUrl && (
            <a
              href={playlistUrl}
              target="_blank"
              rel="noreferrer"
              className="lavender-button"
              style={{
                display: "inline-block",
                marginTop: "10px",
                padding: "12px 25px",
                borderRadius: "30px",
                background: "linear-gradient(90deg, #a1c4fd, #c2e9fb)",
                color: "white",
                textDecoration: "none",
                fontWeight: "500"
              }}
            >
              🎧 Open Playlist
            </a>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
