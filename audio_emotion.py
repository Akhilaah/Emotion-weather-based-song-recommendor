import numpy as np
import tensorflow as tf
import tensorflow_hub as hub #type: ignore
import librosa #type: ignore

# Load YAMNet model
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Load class map (provided by TF Hub)
class_map_path = tf.keras.utils.get_file(
    "yamnet_class_map.csv",
    "https://storage.googleapis.com/audioset/yamnet_class_map.csv"
)
class_names = [line.strip().split(",")[0] 
               for line in open(class_map_path).readlines()][1:]

# Map YAMNet classes → emotions
YAMNET_TO_EMOTION = {
    "laugh": "happy",
    "giggle": "happy",
    "cry": "sad",
    "sob": "sad",
    "whimper": "sad",
    "angry": "angry",
    "argument": "angry",
    "screaming": "fear",
    "shout": "fear",
    "speech": "neutral",
    "conversation": "neutral"
}

def classify_emotion_from_yamnet(wav_path):
    # ---- Load audio using librosa ----
    y, sr = librosa.load(wav_path, sr=16000)  # YAMNet requires 16 kHz
    waveform = y.astype(np.float32)

    # ---- Run YAMNet ----
    scores, embeddings, spectrogram = yamnet_model(waveform)
    scores_np = scores.numpy()
    mean_scores = np.mean(scores_np, axis=0)  # average over time frames

    # ---- Get top predicted class ----
    top_index = np.argmax(mean_scores)
    predicted_class = class_names[top_index]

    # ---- Map class → emotion ----
    emotion = "neutral"
    for keyword, emo in YAMNET_TO_EMOTION.items():
        if keyword in predicted_class.lower():
            emotion = emo
            break

    return {
        "yamnet_class": predicted_class,
        "emotion": emotion
    }
