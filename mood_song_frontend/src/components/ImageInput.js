import React, { useState, useRef } from "react";
import Webcam from "react-webcam";
import axios from "axios";

function ImageInput({ onResult }) {
  const [showWebcam, setShowWebcam] = useState(false);
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const webcamRef = useRef(null);

  // 📸 Capture webcam image
  const capture = () => {
    const imgSrc = webcamRef.current.getScreenshot();
    setImage(imgSrc);
  };

  // 🌤️ WEATHER + EMOTION COMBINED FETCH
  const fetchWeatherEmotionPlaylist = async (emotion) => {
    try {
      // Get location
      const pos = await new Promise((resolve, reject) =>
        navigator.geolocation.getCurrentPosition(resolve, reject)
      );

      const lat = pos.coords.latitude;
      const lon = pos.coords.longitude;

      // Call backend for combined weather+emotion
      const res = await axios.post("http://127.0.0.1:8000/weather/auto/emotion", {
        lat,
        lon,
        emotion,
      });

      return {
        emotion,
        playlist_url: res.data.playlist_url,
      };
    } catch (err) {
      console.error("Weather + Emotion error:", err);
      return {
        emotion,
        playlist_url: null,
      };
    }
  };

  // 🚀 Send captured image to backend
  const sendImage = async () => {
    if (!image) return;

    setLoading(true);
    try {
      const blob = await (await fetch(image)).blob();
      const formData = new FormData();
      formData.append("file", blob, "captured.png");

      // Step 1 — Detect emotion from backend
      const res = await axios.post(
        "http://127.0.0.1:8000/predict/image",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      const detectedEmotion = res.data.emotion;

      // Step 2 — Combine with weather automatically
      const finalResult = await fetchWeatherEmotionPlaylist(detectedEmotion);

      // Send final result back to App.js
      onResult?.(finalResult.emotion, finalResult.playlist_url);

    } catch (err) {
      console.error("Camera upload failed:", err);
      onResult?.("Error detecting emotion", null);
    }

    setLoading(false);
  };

  return (
    <div className="image-input-container fade-in">

      {/* ---------- OPEN CAMERA BUTTON ---------- */}
      {!showWebcam && (
        <button className="lavender-button big" onClick={() => setShowWebcam(true)}>
           Open Camera
        </button>
      )}

      {/* ---------- WEBCAM VIEW ---------- */}
      {showWebcam && (
        <div className="fade-in" style={{ marginTop: "15px" }}>
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/png"
            className="webcam-box"
          />

          <button className="lavender-button big" onClick={capture}>
            📷 Capture Photo
          </button>
        </div>
      )}

      {/* ---------- PREVIEW + SUBMIT ---------- */}
      {image && (
        <div className="fade-in" style={{ marginTop: "15px" }}>
          <img src={image} alt="Captured" className="image-preview" />

          <button className="lavender-button big" onClick={sendImage}>
            ✨ Detect Emotion
          </button>
        </div>
      )}

      {/* ---------- SPINNER LOADING ---------- */}
      {loading && (
        <div className="spinner-container fade-in">
          <div className="spinner"></div>
          <p>Analyzing image…</p>
        </div>
      )}
    </div>
  );
}

export default ImageInput;
