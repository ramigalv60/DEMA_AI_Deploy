import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
class CFG:
    image_size_melanoma = (224, 224) 
    image_size_lesion = (256, 256)
    seed = 42

tf.random.set_seed(CFG.seed)

# Define the classes based on the model
classes = ['Enfeksiyonel', 'Ekzama', 'Akne', 'Pigment', 'Benign', 'Malign']

app = FastAPI()

# Load the model without custom objects
#model_melanoma = layers.TFSMLayer('modelos/saved_model_melanoma', call_endpoint="serving_default")
#model_lesiones = layers.TFSMLayer('modelos/saved_model_lesiones', call_endpoint="serving_default")

def load_and_preprocess_image_melanoma(image_data):
    image = Image.open(BytesIO(image_data))

    image = image.convert('RGB')
    image = image.resize(CFG.image_size_melanoma)
    image_array = img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def load_and_preprocess_image_lesiones(image_data):
    image = Image.open(BytesIO(image_data))

    image = image.convert('RGB')
    image = image.resize(CFG.image_size_lesion)
    image_array = img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.get("/")
def read_root():
    return {"message": "Welcome to the model prediction API!"}

@app.post("/predict/melanoma")
async def predict(file: UploadFile = File(...)):
    try:
        model = load_model('models/my_model.keras')
        image_data = await file.read()
        image = load_and_preprocess_image_melanoma(image_data)
        prediction = model.predict(image)
        prediction = prediction.tolist()
        return {"prediction": prediction}
        #return {"prediction": classes[np.argmax(prediction)]}
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/predict/lesiones")
async def predict(file: UploadFile = File(...)):
    try:
        model = load_model('models/bestmodel.keras')
        image_data = await file.read()
        image = load_and_preprocess_image_lesiones(image_data)
        prediction = model.predict(image)
        prediction = prediction.tolist()
        return {"prediction": classes[np.argmax(prediction)]}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)