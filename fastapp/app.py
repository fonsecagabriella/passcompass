# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class InputData(BaseModel):
    features: list[float]  # e.g., [5.1, 3.5, 1.4, 0.2]

@app.post("/predict")
def predict(input_data: InputData):
    X = np.array(input_data.features).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": prediction.tolist()}
