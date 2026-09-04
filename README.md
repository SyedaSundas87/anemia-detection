# CBC Diagnosis Predictor — Anemia & Blood Disorder Detection

An end-to-end machine learning project that predicts a blood diagnosis (Healthy, various anemia types, Leukemia, Thrombocytopenia, etc.) from a patient's Complete Blood Count (CBC) values. Built as a learning project covering the full ML lifecycle — data cleaning, model training, a FastAPI backend, and a simple web frontend.

## Live Demo

- Frontend: _add your deployed link here_
- API docs: _add your deployed backend URL + `/docs` here_

## Overview

Given 14 CBC lab values, the model classifies a patient into one of 9 diagnosis categories:

- Healthy
- Iron deficiency anemia
- Normocytic hypochromic anemia
- Normocytic normochromic anemia
- Other microcytic anemia
- Macrocytic anemia
- Thrombocytopenia
- Leukemia
- Leukemia with thrombocytopenia

## Dataset

- **Source:** `diagnosed_cbc_data_v4.csv`
- **Size:** 1,281 rows (1,232 after removing duplicates), 14 numeric features, 1 target column
- **Features:** WBC, LYMp, NEUTp, LYMn, NEUTn, RBC, HGB, HCT, MCV, MCH, MCHC, PLT, PDW, PCT

## Model

- **Algorithm:** Random Forest Classifier (compared against Logistic Regression, KNN, SVM, and XGBoost)
- **Tuning:** `RandomizedSearchCV` with 5-fold stratified cross-validation, optimizing macro-F1
- **Final performance:** ~100% test accuracy, 0.957 macro-F1
- **Why Random Forest:** tree-based models captured the non-linear, threshold-based relationships in CBC data (e.g. low HGB + low MCV → iron deficiency) far better than linear models

## Project Structure

```
cbc_project/
├── model/
│   ├── cbc_diagnosis_model.joblib   # trained Random Forest model
│   └── label_encoder.joblib          # maps predictions back to diagnosis names
├── backend/
│   └── main.py                       # FastAPI app
├── frontend/
│   └── index.html                    # simple form-based UI
├── notebooks/
│   └── (training notebook)
├── requirements.txt
└── README.md
```

## Tech Stack

- **ML:** Python, pandas, scikit-learn
- **Backend:** FastAPI, Uvicorn
- **Frontend:** HTML, vanilla JavaScript (fetch API)

## Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/SyedaSundas87/anemia-detection.git
cd anemia-detection
```

### 2. Set up a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the backend
```bash
uvicorn backend.main:app --reload
```
API will be live at `http://127.0.0.1:8000` — interactive docs at `http://127.0.0.1:8000/docs`.

### 5. Run the frontend
Open `frontend/index.html` in a browser (or use VS Code's Live Server extension).

## API Usage

**POST** `/predict`

Request body:
```json
{
  "WBC": 6.5, "LYMp": 32.0, "NEUTp": 58.0, "LYMn": 2.1, "NEUTn": 3.8,
  "RBC": 4.8, "HGB": 13.5, "HCT": 40.0, "MCV": 85.0, "MCH": 28.5,
  "MCHC": 33.5, "PLT": 250.0, "PDW": 13.0, "PCT": 0.22
}
```

Response:
```json
{ "diagnosis": "Healthy" }
```

## Future Improvements

- Add prediction confidence scores (`predict_proba`)
- Handle rare classes better (e.g. SMOTE oversampling — some diagnoses have <15 training samples)
- Add SHAP-based explainability for individual predictions
- Input range validation (reject biologically impossible values)

## Disclaimer

This is an educational project, not a medical diagnostic tool. Predictions should not be used for real clinical decisions.
