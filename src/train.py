from pathlib import Path
import json, warnings, joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "landslide_training_data.csv"
MODEL = ROOT / "models" / "landslide_risk_ensemble.joblib"
METRICS = ROOT / "reports" / "metrics_runtime.json"

FEATURES = [
    "elevation_m", "slope_deg", "aspect_deg", "curvature",
    "rainfall_24h_mm", "rainfall_72h_mm", "rainfall_7d_mm",
    "land_cover", "ndvi", "soil_type",
    "distance_to_river_m", "distance_to_road_m"
]
CATEGORICAL = ["land_cover", "soil_type"]

df = pd.read_csv(DATA)
X = df[FEATURES]
y = df["landslide"].astype(int)

groups = (
    np.floor(df["latitude"] * 2).astype(int).astype(str)
    + "_"
    + np.floor(df["longitude"] * 2).astype(int).astype(str)
)

pre = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
    remainder="passthrough"
)

spw = float((y == 0).sum() / (y == 1).sum())

xgb = XGBClassifier(
    n_estimators=500, learning_rate=0.04, max_depth=5, min_child_weight=2,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, reg_alpha=0.05,
    scale_pos_weight=spw, eval_metric="logloss", random_state=42, n_jobs=-1
)
lgbm = LGBMClassifier(
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, reg_alpha=0.05,
    class_weight={0: 1.0, 1: spw}, random_state=42, n_jobs=-1, verbosity=-1
)

voter = VotingClassifier(
    estimators=[("xgb", xgb), ("lgbm", lgbm)],
    voting="soft", weights=[0.30, 0.70], n_jobs=1
)
pipeline = Pipeline([("preprocess", pre), ("model", voter)])

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    prob = cross_val_predict(
        pipeline, X, y, groups=groups, cv=cv,
        method="predict_proba", n_jobs=1
    )[:, 1]

thresholds = np.linspace(0.05, 0.95, 181)
best = None
for threshold in thresholds:
    pred = (prob >= threshold).astype(int)
    row = {
        "threshold": float(threshold),
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0))
    }
    if best is None or row["f1"] > best["f1"]:
        best = row

metrics = {
    "roc_auc": float(roc_auc_score(y, prob)),
    "pr_auc": float(average_precision_score(y, prob)),
    **best
}

pipeline.fit(X, y)
MODEL.parent.mkdir(exist_ok=True)
joblib.dump(pipeline, MODEL)
with open(METRICS, "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
print(f"Saved model: {MODEL}")
