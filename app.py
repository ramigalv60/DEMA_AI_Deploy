import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
from tensorflow.keras.models import load_model
from io import BytesIO

class CFG:
    image_size = (224, 224)  # Example image size, adjust as needed
    seed = 42

tf.random.set_seed(CFG.seed)

app = FastAPI()

# Load the model without custom objects
model = load_model('models/model.h5')
#tf.keras.layers.TFSMLayer("FOLDER_NAME", call_endpoint="serving_default")

def load_and_preprocess_image(image_data):
    image = Image.open(BytesIO(image_data))
    image = image.resize(CFG.image_size)
    image = np.array(image) / 255.0  # Manually normalize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.get("/")
def read_root():
    return {"message": "Welcome to the model prediction API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = load_and_preprocess_image(image_data)
    prediction = model.predict(image)
    binary_prediction = int(prediction[0][0] > 0.5)
    return {"prediction": binary_prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)