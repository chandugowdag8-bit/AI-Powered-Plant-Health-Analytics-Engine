
import io
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# Project paths
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "plant_disease_model.keras"
LABELS_PATH = BASE / "class_names.json"

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = float(
    os.getenv("CONFIDENCE_THRESHOLD", "0.60")
)


# FastAPI application
app = FastAPI(
    title="AI Plant Health Analytics API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local development only
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global model variables
model = None
class_names = []


# Load trained AI model
@app.on_event("startup")
def load_model():
    global model, class_names

    if MODEL_PATH.exists() and LABELS_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)

        class_names = json.loads(
            LABELS_PATH.read_text(encoding="utf-8")
        )

        print("AI MODEL LOADED SUCCESSFULLY")
        print("Total classes:", len(class_names))
    else:
        print("WARNING: AI model or class labels not found.")


# Home endpoint
@app.get("/")
def root():
    return {
        "message": "Plant disease AI API",
        "model_loaded": model is not None
    }


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": len(class_names)
    }


# Plant disease prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Train it first."
        )

    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp"
    }:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG or WEBP image."
        )

    raw = await file.read()

    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 5 MB."
        )

    try:
        image = (
            Image.open(io.BytesIO(raw))
            .convert("RGB")
            .resize(IMG_SIZE)
        )

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    # Image preprocessing
    arr = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(arr, axis=0)

    # AI prediction
    probs = model.predict(batch, verbose=0)[0]

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    predicted_class = class_names[idx]

    # Detect background or missing leaf
    if predicted_class == "Background_without_leaves":
        return {
            "prediction": (
                "No plant leaf detected. "
                "Please upload a clear leaf photo."
            ),
            "top_class": predicted_class,
            "confidence": confidence,
            "status": "no_leaf",
            "threshold": CONFIDENCE_THRESHOLD,
            "model": "MobileNetV2 fine-tuned on PlantVillage"
        }

    # Check prediction confidence
    uncertain = confidence < CONFIDENCE_THRESHOLD

    return {
        "prediction": (
            predicted_class
            if not uncertain
            else "Unable to identify"
        ),
        "top_class": predicted_class,
        "confidence": confidence,
        "status": (
            "uncertain"
            if uncertain
            else "prediction"
        ),
        "threshold": CONFIDENCE_THRESHOLD,
        "model": "MobileNetV2 fine-tuned on PlantVillage"
    }
