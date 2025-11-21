import React, { useState } from "react";
import axios from "axios";
import Webcam from "react-webcam";

const MoodInput = () => {
  const [text, setText] = useState("");
  const [emotionResult, setEmotionResult] = useState("");
  const [playlistUrl, setPlaylistUrl] = useState("");
  const [showWebcam, setShowWebcam] = useState(false);
  const [audioFile, setAudioFile] = useState(null);

  const handleTextSubmit = async () => {
    const response = await axios.post("http://127.0.0.1:8000/predict/text", { text: inputText });
    setEmotionResult(response.data.emotion);
    setPlaylistUrl(response.data.playlist_url);
  };

  const handleImageCapture = async (imageSrc) => {
    const blob = await fetch(imageSrc).then((res) => res.blob());
    const file = new File([blob], "capture.jpg", { type: "image/jpeg" });

    const formData = new FormData();
    formData.append("file", file);

    const response = await axios.post("http://127.0.0.1:8000/predict/image", formData);
    setEmotionResult(response.data.emotion);
    setPlaylistUrl(response.data.playlist_url);
  };

  const handleAudioSubmit = async () => {
    if (!audioFile) return alert("Please select an audio file first!");
    const formData = new FormData();
    formData.append("file", audioFile);

    const response = await axios.post("http://127.0.0.1:8000/predict/audio", formData);
    setEmotionResult(response.data.emotion);
    setPlaylistUrl(response.data.playlist_url);
  };

  return (
    <div style={{ padding: "20px", textAlign: "center" }}>
      <h1>🎵 Mood-Based Song Recommender</h1>

      <div style={{ marginTop: "20px" }}>
        <h3>Enter your mood as text</h3>
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type how you feel..."
          style={{ padding: "8px", width: "60%" }}
        />
        <button onClick={handleTextSubmit}>Submit</button>
      </div>

      <div style={{ marginTop: "40px" }}>
        <h3>Or capture your image</h3>
        {showWebcam ? (
          <Webcam
            audio={false}
            screenshotFormat="image/jpeg"
            width={320}
            height={240}
          >
            {({ getScreenshot }) => (
              <button
                onClick={() => {
                  const imageSrc = getScreenshot();
                  handleImageCapture(imageSrc);
                  setShowWebcam(false);
                }}
              >
                Capture
              </button>
            )}
          </Webcam>
        ) : (
          <button onClick={() => setShowWebcam(true)}>Open Camera</button>
        )}
      </div>

      <div style={{ marginTop: "40px" }}>
        <h3>Or upload an audio file</h3>
        <input type="file" accept="audio/*" onChange={(e) => setAudioFile(e.target.files[0])} />
        <button onClick={handleAudioSubmit}>Analyze Audio</button>
      </div>

      {emotionResult && (
        <div style={{ marginTop: "40px" }}>
          <h3>Detected Emotion: {emotionResult}</h3>
          {playlistUrl && (
            <p>
              🎧 Recommended Playlist:{" "}
              <a href={playlistUrl} target="_blank" rel="noopener noreferrer">
                Open on Spotify
              </a>
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default MoodInput;
