# Import library FastAPI
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import pickle

# Load model dari file
with open("models/linear_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Inisialisasi FastAPI
app = FastAPI()


# Model data untuk request API
class InputData(BaseModel):
    x: float


# Endpoint untuk prediksi
@app.post("/predict")
def predict(data: InputData):
    x_input = np.array([[data.x]])  # Konversi ke array numpy
    y_output = loaded_model.predict(x_input)  # Lakukan prediksi
    return {"prediction": y_output[0][0]}
