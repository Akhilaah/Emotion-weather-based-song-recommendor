/*let mediaRecorder;
let audioChunks = [];

const startRecording = () => {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
            const formData = new FormData();
            formData.append("audio", audioBlob, "temp_audio.webm");

            // Play recorded audio
            const audioURL = URL.createObjectURL(audioBlob);
            const audioElem = document.getElementById("audioPlayback");
            audioElem.src = audioURL;
            audioElem.play();

            // Send to backend
            try {
                const response = await fetch("http://127.0.0.1:8000/predict/audio", {
                    method: "POST",
                    body: formData,
                });
                const data = await response.json();
                console.log("Prediction:", data);
                document.getElementById("detectedEmotion").innerText = `Detected emotion: ${data.emotion}`;

                // Optionally, fetch Spotify playlist
                const playlistResponse = await fetch(`http://127.0.0.1:8000/predict/text`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: data.emotion })
                });
                const playlistData = await playlistResponse.json();
                console.log("Playlist:", playlistData);
                document.getElementById("playlistLink").href = playlistData.playlist_url;
                document.getElementById("playlistLink").innerText = "Open Playlist";
            } catch (err) {
                console.error("Error uploading audio:", err);
            }
        };

        mediaRecorder.start();
    }).catch(err => console.error("Microphone access denied", err));
};

const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
};
*/