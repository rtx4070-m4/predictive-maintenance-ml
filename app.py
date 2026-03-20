
from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI()

class InputData(BaseModel):
    sequence: list

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def get_prediction(data: InputData):
    prob = predict(data.sequence)
    return {"failure_probability": prob}
