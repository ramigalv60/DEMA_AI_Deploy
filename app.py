import os
import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Request
from pydantic import BaseModel
from PIL import Image
from tensorflow.keras.models import load_model

class CFG:
    image_size = (224, 224)  # Example image size, adjust as needed
    seed = 42

tf.random.set_seed(CFG.seed)

app = FastAPI()

# Load the model
model = load_model('models/model.h5')

class PredictionRequest(BaseModel):
    image_path: str

def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, CFG.image_size)
    image = image / 255.0  # Normalize image
    return image

@app.get("/")
def read_root():
    return {"message": "Welcome to the model prediction API!"}

@app.post("/predict")
def predict(request: PredictionRequest):
    image = load_and_preprocess_image(request.image_path)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    binary_prediction = int(prediction[0][0] > 0.5)
    return {"prediction": binary_prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)