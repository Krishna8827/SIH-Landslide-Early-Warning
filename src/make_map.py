from pathlib import Path
import joblib, pandas as pd, folium
from folium.plugins import HeatMap

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "landslide_training_data.csv")
model = joblib.load(ROOT / "models" / "landslide_risk_ensemble.joblib")

FEATURES = [
    "elevation_m", "slope_deg", "aspect_deg", "curvature",
    "rainfall_24h_mm", "rainfall_72h_mm", "rainfall_7d_mm",
    "land_cover", "ndvi", "soil_type",
    "distance_to_river_m", "distance_to_road_m"
]

prob = model.predict_proba(df[FEATURES])[:, 1]

m = folium.Map(location=[26.3, 92.5], zoom_start=6, tiles="OpenStreetMap")
HeatMap(
    [[float(a), float(b), float(c)] for a, b, c in zip(df.latitude, df.longitude, prob)],
    radius=13, blur=15, max_zoom=9
).add_to(m)
m.save(ROOT / "outputs" / "risk_heatmap_demo.html")
print("Saved outputs/risk_heatmap_demo.html")
