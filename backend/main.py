from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI()

# Build a path relative to THIS FILE's location, not the terminal's cwd
BASE_DIR = Path(__file__).resolve().parent      # .../cbc_project/backend
MODEL_DIR = BASE_DIR.parent / "model"             # .../cbc_project/model

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

# 2. The actual prediction endpoint
@app.post("/predict")
def predict(data: CBCInput):
    input_df = pd.DataFrame([data.dict()])
    pred_encoded = model.predict(input_df)
    pred_label = le.inverse_transform(pred_encoded)
    return {"diagnosis": pred_label[0]}