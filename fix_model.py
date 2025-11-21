from tensorflow.keras.models import load_model #type: ignore
import os

# Load your old model (make sure it's in the same folder)
model_path = "emotion_model.hdf5"
fixed_model_path = "emotion_model_fixed.h5"

if not os.path.exists(model_path):
    print("❌ Error: emotion_model.hdf5 not found in the current directory.")
else:
    print(f"🔹 Loading model from: {model_path}")
    model = load_model(model_path, compile=False)

    # Save it in new format
    model.save(fixed_model_path)
    print(f"✅ Model successfully re-saved as: {fixed_model_path}")
 