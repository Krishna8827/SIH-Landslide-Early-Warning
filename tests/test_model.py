from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "landslide_risk_ensemble.joblib"

def test_model_loads_and_predicts():
    model = joblib.load(MODEL)
    sample = pd.DataFrame([{
        "elevation_m": 1450.0,
        "slope_deg": 38.0,
        "aspect_deg": 120.0,
        "curvature": 0.15,
        "rainfall_24h_mm": 95.0,
        "rainfall_72h_mm": 210.0,
        "rainfall_7d_mm": 340.0,
        "land_cover": "Dense Forest",
        "ndvi": 0.55,
        "soil_type": "Loam",
        "distance_to_river_m": 800.0,
        "distance_to_road_m": 250.0,
    }])
    score = float(model.predict_proba(sample)[0, 1])
    assert 0.0 <= score <= 1.0
