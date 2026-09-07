from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

POS_FILE = ROOT / "data" / "processed" / "roads_rivers" / "ner_events_with_all_positive_features.csv"
NEG_FILE = ROOT / "data" / "processed" / "background_features" / "ner_background_points_with_all_features.csv"

OUT_DIR = ROOT / "data" / "processed" / "training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "v2_training_matrix.csv"
SUMMARY_JSON = OUT_DIR / "v2_training_summary.json"
MANIFEST_JSON = OUT_DIR / "v2_feature_manifest.json"
MISSINGNESS_CSV = OUT_DIR / "v2_feature_missingness_by_class.csv"

FEATURES = [
    "elevation_m",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "curvature_1_per_m",
    "rainfall_24h_mm",
    "rainfall_72h_mm",
    "rainfall_7d_mm",
    "land_cover",
    "ndvi_model_value",
    "ndvi_missing",
    "soil_moisture_surface_m3_m3",
    "soil_moisture_7_28cm_m3_m3",
    "soil_moisture_28_100cm_m3_m3",
    "soil_moisture_0_100cm_m3_m3",
    "soil_moisture_surface_3d_mean_m3_m3",
    "soil_moisture_surface_7d_mean_m3_m3",
    "soil_moisture_surface_anomaly_vs_7d",
    "distance_to_road_m",
    "distance_to_river_m",
]

NUMERIC_FEATURES = [f for f in FEATURES if f != "land_cover"]
CATEGORICAL_FEATURES = ["land_cover"]

META = [
    "sample_id",
    "pair_id",
    "label",
    "sample_type",
    "state",
    "latitude",
    "longitude",
    "event_date",
    "spatial_block_0p5deg",
]

SOURCE_NUMERIC = [
    "elevation_m",
    "slope_deg",
    "curvature_1_per_m",
    "rainfall_24h_mm",
    "rainfall_72h_mm",
    "rainfall_7d_mm",
    "ndvi_model_value",
    "soil_moisture_surface_m3_m3",
    "soil_moisture_7_28cm_m3_m3",
    "soil_moisture_28_100cm_m3_m3",
    "soil_moisture_0_100cm_m3_m3",
    "soil_moisture_surface_3d_mean_m3_m3",
    "soil_moisture_surface_7d_mean_m3_m3",
    "soil_moisture_surface_anomaly_vs_7d",
    "distance_to_road_m",
    "distance_to_river_m",
]

def require(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing columns: {missing}")

def spatial_block(lat, lon, size=0.5):
    if pd.isna(lat) or pd.isna(lon):
        return np.nan
    a = np.floor(float(lat) / size) * size
    b = np.floor(float(lon) / size) * size
    return f"{a:.1f}_{b:.1f}"

def build(df, positive):
    out = pd.DataFrame()

    if positive:
        out["sample_id"] = df["event_id"].astype(str)
        out["pair_id"] = df["event_id"].astype(str)
        out["label"] = 1
        out["sample_type"] = "real_landslide_event"
    else:
        out["sample_id"] = df["background_id"].astype(str)
        out["pair_id"] = df["matched_positive_event_id"].astype(str)
        out["label"] = 0
        out["sample_type"] = "matched_pseudo_absence"

    out["state"] = df["state"]
    out["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    out["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    aspect = pd.to_numeric(df["aspect_deg"], errors="coerce")
    radians = np.deg2rad(aspect)
    out["aspect_sin"] = np.sin(radians)
    out["aspect_cos"] = np.cos(radians)

    for c in SOURCE_NUMERIC:
        out[c] = pd.to_numeric(df[c], errors="coerce")

    out["land_cover"] = df["land_cover"].astype("string")
    out["ndvi_missing"] = out["ndvi_model_value"].isna().astype(int)

    out["spatial_block_0p5deg"] = [
        spatial_block(lat, lon)
        for lat, lon in zip(out["latitude"], out["longitude"])
    ]

    return out[META + FEATURES]

def make_missingness(df):
    rows = []
    for feature in FEATURES:
        for label, name in [(1, "positive"), (0, "background")]:
            part = df[df["label"] == label]
            missing = int(part[feature].isna().sum())
            rows.append({
                "feature": feature,
                "class": name,
                "rows": int(len(part)),
                "missing": missing,
                "missing_pct": round(100 * missing / len(part), 3),
            })
    return pd.DataFrame(rows)

def main():
    print("=" * 80)
    print("STAGE 8 — BUILD FINAL V2 TRAINING MATRIX")
    print("=" * 80)

    pos = pd.read_csv(POS_FILE)
    neg = pd.read_csv(NEG_FILE)

    require(
        pos,
        ["event_id", "state", "latitude", "longitude", "event_date", "aspect_deg", "land_cover"] + SOURCE_NUMERIC,
        "positive dataset",
    )
    require(
        neg,
        ["background_id", "matched_positive_event_id", "state", "latitude", "longitude", "event_date", "aspect_deg", "land_cover"] + SOURCE_NUMERIC,
        "background dataset",
    )

    p = build(pos, True)
    n = build(neg, False)
    combined = pd.concat([p, n], ignore_index=True)

    if len(combined) != len(pos) + len(neg):
        raise RuntimeError("Unexpected row-count change.")
    if combined["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample_id values found.")

    combined.to_csv(OUT_CSV, index=False)
    make_missingness(combined).to_csv(MISSINGNESS_CSV, index=False)

    summary = {
        "rows_total": int(len(combined)),
        "positive_rows": int((combined["label"] == 1).sum()),
        "background_rows": int((combined["label"] == 0).sum()),
        "model_features": len(FEATURES),
        "spatial_blocks_0p5deg": int(combined["spatial_block_0p5deg"].nunique(dropna=True)),
        "positive_ndvi_available": int(combined.loc[combined["label"] == 1, "ndvi_model_value"].notna().sum()),
        "background_ndvi_available": int(combined.loc[combined["label"] == 0, "ndvi_model_value"].notna().sum()),
        "positive_road_available": int(combined.loc[combined["label"] == 1, "distance_to_road_m"].notna().sum()),
        "background_road_available": int(combined.loc[combined["label"] == 0, "distance_to_road_m"].notna().sum()),
        "positive_river_available": int(combined.loc[combined["label"] == 1, "distance_to_river_m"].notna().sum()),
        "background_river_available": int(combined.loc[combined["label"] == 0, "distance_to_river_m"].notna().sum()),
        "output_csv": str(OUT_CSV),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = {
        "target": "label",
        "positive_class": 1,
        "negative_class": 0,
        "negative_class_note": "Background samples are matched pseudo-absences, not confirmed stable locations.",
        "model_features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metadata_not_for_model": META,
        "validation_group": "spatial_block_0p5deg",
        "notes": [
            "Latitude, longitude, state, event date and IDs are metadata only and must not be model features.",
            "Aspect is represented as sin/cos instead of raw degrees.",
            "NDVI missingness is represented explicitly with ndvi_missing.",
            "Road/river distances are current OSM susceptibility proxies, not event-date historical reconstructions.",
            "WorldCover 2021 is a static land-cover proxy.",
            "Historical daily rainfall/soil-moisture features are for prototype calibration and do not by themselves prove a fixed operational lead time.",
        ],
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Positive rows:   {summary['positive_rows']}")
    print(f"Background rows: {summary['background_rows']}")
    print(f"Total rows:      {summary['rows_total']}")
    print(f"Model features:  {summary['model_features']}")
    print(f"0.5° spatial blocks: {summary['spatial_blocks_0p5deg']}")
    print()
    print(f"NDVI available  positive={summary['positive_ndvi_available']}/{len(pos)}  background={summary['background_ndvi_available']}/{len(neg)}")
    print(f"Road available  positive={summary['positive_road_available']}/{len(pos)}  background={summary['background_road_available']}/{len(neg)}")
    print(f"River available positive={summary['positive_river_available']}/{len(pos)}  background={summary['background_river_available']}/{len(neg)}")

    print("\n" + "=" * 80)
    print("STAGE 8 COMPLETE")
    print("=" * 80)
    print(f"Training matrix:\n{OUT_CSV}")
    print(f"\nFeature manifest:\n{MANIFEST_JSON}")
    print(f"\nMissingness report:\n{MISSINGNESS_CSV}")

if __name__ == "__main__":
    main()
