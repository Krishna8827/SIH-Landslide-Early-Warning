from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MODEL = joblib.load(ROOT / "models" / "landslide_risk_ensemble.joblib")

app = FastAPI(
    title="SIH Landslide Risk API",
    description="Prototype landslide-risk inference API for the supplied SIH dataset.",
    version="1.0.0"
)

class RiskInput(BaseModel):
    elevation_m: float
    slope_deg: float
    aspect_deg: float
    curvature: float
    rainfall_24h_mm: float
    rainfall_72h_mm: float
    rainfall_7d_mm: float
    land_cover: str
    ndvi: float
    soil_type: str
    distance_to_river_m: float
    distance_to_road_m: float

def risk_level(score: float) -> str:
    if score < 0.25:
        return "LOW"
    if score < 0.50:
        return "MODERATE"
    if score < 0.75:
        return "HIGH"
    return "CRITICAL"

@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost-lightgbm-ensemble"}

@app.post("/predict")
def predict(payload: RiskInput):
    row = pd.DataFrame([payload.model_dump()])
    score = float(MODEL.predict_proba(row)[0, 1])
    return {
        "risk_probability": round(score, 6),
        "risk_level": risk_level(score)
    }
