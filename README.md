# AI-Powered Landslide Risk Prediction for North-East India

SIH-ready machine-learning prototype that converts engineered terrain and
weather features into a landslide risk probability and dashboard risk class.

## Final model

The repository compares:
- XGBoost
- LightGBM
- Random Forest

The final predictor is a **soft-voting ensemble: 30% XGBoost + 70% LightGBM**.

### Prototype geographic cross-validation results

| Metric | Result |
|---|---:|
| ROC-AUC | 0.9766 |
| PR-AUC | 0.9249 |
| F1 | 0.8721 |
| Precision | 0.8331 |
| Recall | 0.9150 |
| Selected threshold | 0.18 |

Validation uses 5-fold `StratifiedGroupKFold` with approximately 0.5-degree
geographic blocks to reduce leakage between nearby points.

## Inputs

The model consumes 12 features:

`elevation_m`, `slope_deg`, `aspect_deg`, `curvature`,
`rainfall_24h_mm`, `rainfall_72h_mm`, `rainfall_7d_mm`,
`land_cover`, `ndvi`, `soil_type`,
`distance_to_river_m`, `distance_to_road_m`.

Potential target leakage / post-event columns are intentionally excluded:
`event_id`, `state`, `latitude`, `longitude`, `monsoon_phase`,
`severity_class`, `is_fatal`, `reliability_score`, `source`.

## Dashboard risk bands

- **LOW:** 0.00–0.25
- **MODERATE:** 0.25–0.50
- **HIGH:** 0.50–0.75
- **CRITICAL:** 0.75–1.00

## Repository structure

```text
data/       supplied training dataset
models/     trained ensemble
src/        training, inference, API, map generation
reports/    metrics, model comparison, plots, feature importance
outputs/    demo predictions and interactive heatmap
tests/      smoke test
```

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
pytest -q
```

### Run one prediction

```bash
python src/predict.py --json "{\"elevation_m\":1450,\"slope_deg\":38,\"aspect_deg\":120,\"curvature\":0.15,\"rainfall_24h_mm\":95,\"rainfall_72h_mm\":210,\"rainfall_7d_mm\":340,\"land_cover\":\"Dense Forest\",\"ndvi\":0.55,\"soil_type\":\"Loam\",\"distance_to_river_m\":800,\"distance_to_road_m\":250}"
```

### Start the API

```bash
uvicorn src.api:app --reload
```

Open `/docs` on the local FastAPI server for interactive API testing.

### Retrain

```bash
python src/train.py
```

### Rebuild the demo risk heatmap

```bash
python src/make_map.py
```

Then open `outputs/risk_heatmap_demo.html`.

## Dataset warning

This is essential when presenting the SIH result:

- Dataset rows: **4800**
- Positive rows: **1200**
- Negative rows: **3600**
- Rows marked `real_event`: **381**
- Synthetic augmented positives: **819**
- Synthetic pseudo-absence negatives: **3600**

Therefore the reported numbers are **prototype validation metrics on the supplied
dataset**, not validated operational accuracy for a public early-warning system.

The current code receives already-engineered environmental features. Live IMD
weather ingestion, DEM/raster extraction, satellite feeds, IoT soil sensors,
CAP/SMS alerts, and a production GIS backend are integration layers rather than
features implemented by this training dataset.

## Suggested SIH explanation

> We use static terrain susceptibility variables together with dynamic rainfall
> triggers. XGBoost and LightGBM learn nonlinear interactions among slope,
> elevation, terrain morphology, rainfall accumulation, vegetation, soil, and
> proximity features. Predictions are emitted as probabilities so the GIS
> frontend can render continuous risk rather than only a binary decision.
> Geographic group validation is used to reduce optimistic leakage from nearby
> locations.

## Safety / deployment

Do not use this prototype as the sole basis for real emergency decisions.
Operational deployment requires independent real-world validation, real
non-landslide controls, calibrated thresholds, live sensor/weather validation,
and disaster-management authority review.
