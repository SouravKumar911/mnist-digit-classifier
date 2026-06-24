from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import tensorflow as tf
import numpy as np
from PIL import Image

# Create FastAPI app
app = FastAPI()

# Static folder
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

# Templates folder
templates = Jinja2Templates(
    directory="templates"
)

# Load CNN model
model = tf.keras.models.load_model(
    "mnist_cnn.h5"
)

from fastapi.responses import FileResponse

@app.get("/")
async def home():
    return FileResponse("templates/index.html")

# Prediction API
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Read image
    image = Image.open(file.file)

    # Convert to grayscale
    image = image.convert("L")

    # Resize to MNIST size
    image = image.resize((28, 28))

    # Convert to numpy array
    image = np.array(image)

    # Normalize
    image = image / 255.0

    # Reshape for CNN
    image = image.reshape(
        1,
        28,
        28,
        1
    )

    # Prediction
    prediction = model.predict(
        image,
        verbose=0
    )

    digit = int(
        np.argmax(prediction)
    )

    confidence = float(
        np.max(prediction) * 100
    )

    return {
        "digit": digit,
        "confidence": round(confidence, 2)
    }