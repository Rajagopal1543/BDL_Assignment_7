from fastapi import FastAPI, UploadFile, File
from prometheus_fastapi_instrumentator import Instrumentator
from PIL import Image
import io
import keras.models
from keras.models import load_model
import numpy as np
from scipy import ndimage

app = FastAPI()



def predict_digit(model, img):
    prediction = model.predict(img.reshape(1, -1))
    return np.argmax(prediction)

def format_image(image):
    img_gray = image.convert('L')
    img_resized = np.array(img_gray.resize((28, 28))) / 255.0
    cy, cx = ndimage.center_of_mass(img_resized)
    rows, cols = img_resized.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    img_centered = ndimage.shift(img_resized, (shifty, shiftx), cval=0)
    return img_centered.flatten()

@app.post('/predict')
async def predict(upload_file: UploadFile = File(...)):
    contents = await upload_file.read()
    img = Image.open(io.BytesIO(contents))
    img_array = format_image(img)
    path = 'Best_Model.h5'
    model = load_model(path)
    digit = predict_digit(model, img_array)
    return {"digit": str(digit)}

# Instrument the FastAPI application with Prometheus
Instrumentator().instrument(app).expose(app)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("task1:app", host="127.0.0.1", port=8000, reload=True)
