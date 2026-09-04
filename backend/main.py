from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
import os
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "model"

model = joblib.load(MODEL_DIR / 'cbc_diagnosis_model.joblib')
le = joblib.load(MODEL_DIR / 'label_encoder.joblib')

@app.get('/')
def home():
    return {'message': 'Hello, this is my CBC diagnosis API'}

class CBCInput(BaseModel):
    WBC: float
    LYMp: float
    NEUTp: float
    LYMn: float
    NEUTn: float
    RBC: float
    HGB: float
    HCT: float
    MCV: float
    MCH: float
    MCHC: float
    PLT: float
    PDW: float
    PCT: float

@app.post("/predict")
def predict(data: CBCInput):
    input_df = pd.DataFrame([data.dict()])
    pred_encoded = model.predict(input_df)
    pred_label = le.inverse_transform(pred_encoded)
    return {"diagnosis": pred_label[0]}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)