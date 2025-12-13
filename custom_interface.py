import torch
from speechbrain.inference.interfaces import Pretrained #type: ignore

class CustomEmotionRecognizer(Pretrained):
    MODULES_NEEDED = ["model"]
    HPARAMS_NEEDED = ["audio_pipeline"]

    def classify_file(self, wav_path):
        out_prob = self.transcribe_file(wav_path)
        scores = out_prob.squeeze().tolist()

        emotions = ["angry", "happy", "sad", "neutral"]

        # Highest logit = predicted emotion
        predicted_idx = torch.argmax(out_prob).item()

        return {
            "emotion": emotions[predicted_idx],
            "scores": dict(zip(emotions, scores)),
        }



