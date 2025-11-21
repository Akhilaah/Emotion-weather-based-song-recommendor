// src/components/TextInput.js
import React, { useState } from "react";
import axios from "axios";

function TextInput() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text) return;
    setLoading(true);

    try {
      // 1️⃣ Detect emotion from text
      const emotionResponse = await axios.post(
        "http://127.0.0.1:8000/predict/text",
        { text }
      );

      const emotion = emotionResponse.data.emotion.toLowerCase();

      // 2️⃣ Get user geolocation
      const pos = await new Promise((resolve, reject) =>
        navigator.geolocation.getCurrentPosition(resolve, reject)
      );

      const lat = pos.coords.latitude;
      const lon = pos.coords.longitude;

      // 3️⃣ Combine weather + emotion (new endpoint)
      const combinedResponse = await axios.post(
        "http://127.0.0.1:8000/weather/auto/emotion",
        {
          lat,
          lon,
          emotion,
        }
      );

      setResult({
        emotion: combinedResponse.data.emotion.toUpperCase(),
        weather: combinedResponse.data.weather.toUpperCase(),
        playlist_url: combinedResponse.data.playlist_url,
      });
    } catch (err) {
      console.error("Error:", err);
      setResult({
        emotion: "ERROR",
        playlist_url: null,
      });
    }

    setLoading(false);
  };

  return (
    <div>
      <form onSubmit={handleSubmit} style={{ display: "flex", alignItems: "center" }}>
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type your mood here..."
          style={{
            width: "70%",
            marginRight: 10,
            padding: "8px",
            borderRadius: "8px",
            border: "none",
          }}
        />
        <button
          type="submit"
          style={{
            backgroundColor: "#00c3ff",
            color: "white",
            border: "none",
            padding: "10px 15px",
            borderRadius: "8px",
            cursor: "pointer",
          }}
        >
          Submit
        </button>
      </form>

      {loading && <p>⏳ Analyzing your mood...</p>}

      {result && (
        <div style={{ marginTop: 10 }}>
          <p>
            💭 Emotion: <strong>{result.emotion}</strong>
          </p>
          {result.weather && (
            <p>
              🌤️ Weather: <strong>{result.weather}</strong>
            </p>
          )}
          {result.playlist_url && (
            <a
              href={result.playlist_url}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                color: "#ffd700",
                fontWeight: "bold",
                textDecoration: "none",
              }}
            >
              🎧 Open Personalized Playlist
            </a>
          )}
        </div>
      )}
    </div>
  );
}

export default TextInput;
