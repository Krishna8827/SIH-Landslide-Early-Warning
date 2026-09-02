from pathlib import Path
import argparse, json, joblib, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "landslide_risk_ensemble.joblib"

FEATURES = [
    "elevation_m", "slope_deg", "aspect_deg", "curvature",
    "rainfall_24h_mm", "rainfall_72h_mm", "rainfall_7d_mm",
    "land_cover", "ndvi", "soil_type",
    "distance_to_river_m", "distance_to_road_m"
]

model = joblib.load(MODEL_PATH)

def risk_level(score: float) -> str:
    if score < 0.25:
        return "LOW"
    if score < 0.50:
        return "MODERATE"
    if score < 0.75:
        return "HIGH"
    return "CRITICAL"

def predict(features: dict) -> dict:
    missing = [f for f in FEATURES if f not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    row = pd.DataFrame([{f: features[f] for f in FEATURES}])
    score = float(model.predict_proba(row)[0, 1])
    return {
        "risk_probability": round(score, 6),
        "risk_level": risk_level(score)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="JSON object with the 12 model features")
    args = parser.parse_args()
    print(json.dumps(predict(json.loads(args.json)), indent=2))
