import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Agri AI - Tomato Leaf Disease Prediction API
# Model: EfficientNetB0 | Input: 224x224 | Classes: 4

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Model",
    "AGRI_AI_Model.keras"
)

CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Spider Mites",
    "Target Spot"
]

IMAGE_SIZE = (224, 224)

app = FastAPI(title="Agri AI - Tomato Leaf Disease API")

# Allow requests from your Vercel frontend (any origin for now — tighten later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None  # loaded lazily on first request to keep cold starts/deploys faster


def get_model():
    global model
    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    return model


def prepare_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


@app.get("/")
def root():
    return {"status": "ok", "message": "Agri AI API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        image_array = prepare_image(image_bytes)
    except Exception as e:
        return {"error": f"Could not process image: {e}"}

    clf = get_model()
    predictions = clf.predict(image_array, verbose=0)
    probabilities = predictions[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index] * 100)

    all_probs = {
        CLASS_NAMES[i]: round(float(probabilities[i] * 100), 2)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": all_probs
    }
