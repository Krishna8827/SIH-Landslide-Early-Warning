from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import json
import os
import sqlite3
import time

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
DATA_DIR = ROOT / "data" / "processed" / "inference"
MODEL_PATH = ROOT / "models" / "v2" / "v2_ensemble_bundle.joblib"
RUNTIME_DIR = Path(os.getenv("LANDSLIDE_RUNTIME_DIR", str(ROOT / "data" / "runtime")))
UPLOAD_DIR = RUNTIME_DIR / "uploads"
DB_PATH = RUNTIME_DIR / "stage11.db"

RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS: Optional[pd.DataFrame] = None
PREDICTIONS_PATH: Optional[Path] = None
SUMMARY: dict = {}
MODEL_BUNDLE = None
STARTED_AT = datetime.now(timezone.utc).isoformat()

ALLOWED_REPORT_TYPES = {
    "crack", "slope movement", "minor landslide", "major landslide",
    "blocked road", "flooding", "other"
}
ALLOWED_REPORT_STATUS = {"NEW", "VERIFIED", "IN_PROGRESS", "RESOLVED"}
ALLOWED_ROAD_STATUS = {"OPEN", "CAUTION", "BLOCKED", "UNKNOWN"}
ALLOWED_MEDIA_TYPES = {
    "image/jpeg", "image/png", "image/webp",
    "video/mp4", "video/webm"
}

TRANSLATIONS = {
    "en": {
        "title": "Landslide Risk Alert",
        "body": (
            "Elevated landslide risk detected near {state}. "
            "Prototype risk score: {score:.2f}. "
            "Follow district authority instructions and avoid vulnerable slopes."
        ),
    },
    "hi": {
        "title": "भूस्खलन जोखिम चेतावनी",
        "body": (
            "{state} के पास भूस्खलन का बढ़ा हुआ जोखिम पाया गया है। "
            "प्रोटोटाइप जोखिम स्कोर: {score:.2f}। "
            "जिला प्रशासन के निर्देशों का पालन करें और संवेदनशील ढलानों से बचें।"
        ),
    },
}

def newest(pattern: str) -> Optional[Path]:
    files = sorted(DATA_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def load_all() -> None:
    global PREDICTIONS, PREDICTIONS_PATH, SUMMARY, MODEL_BUNDLE

    csv_path = newest("ner_risk_predictions_*.csv")
    if not csv_path:
        raise FileNotFoundError(
            f"No Stage 10 prediction CSV found in {DATA_DIR}. "
            "Complete Stage 10 before launching the website."
        )

    df = pd.read_csv(csv_path)
    required = {
        "grid_id", "state", "latitude", "longitude",
        "risk_probability", "risk_level", "prototype_alert"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Stage 10 CSV missing required columns: {missing}")

    if df["prototype_alert"].dtype == object:
        df["prototype_alert"] = (
            df["prototype_alert"].astype(str).str.lower().isin(["true", "1", "yes"])
        )

    PREDICTIONS = df
    PREDICTIONS_PATH = csv_path

    summary_path = newest("ner_risk_summary_*.json")
    if summary_path:
        try:
            SUMMARY = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            SUMMARY = {}

    MODEL_BUNDLE = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS field_reports(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            state TEXT,
            district TEXT,
            report_type TEXT NOT NULL,
            description TEXT,
            media_path TEXT,
            status TEXT NOT NULL DEFAULT 'NEW'
        );

        CREATE TABLE IF NOT EXISTS road_status(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            road_name TEXT NOT NULL,
            state TEXT,
            latitude REAL,
            longitude REAL,
            status TEXT NOT NULL DEFAULT 'UNKNOWN',
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS alerts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            grid_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            state TEXT,
            risk_probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ACTIVE'
        );
        """)

def rows(sql: str, args=()):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, args).fetchall()]

def seed_alerts() -> None:
    if PREDICTIONS is None:
        return
    now = datetime.now(timezone.utc).isoformat()
    active = PREDICTIONS[PREDICTIONS["prototype_alert"] == True]
    with sqlite3.connect(DB_PATH) as conn:
        for r in active.itertuples(index=False):
            conn.execute("""
                INSERT OR IGNORE INTO alerts
                (grid_id, created_at, updated_at, state, risk_probability, risk_level, status)
                VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE')
            """, (
                str(r.grid_id), now, now, str(r.state),
                float(r.risk_probability), str(r.risk_level)
            ))

def risk_level(prob: float) -> str:
    p = float(prob)
    if p < 0.25:
        return "LOW"
    if p < 0.50:
        return "MODERATE"
    if p < 0.75:
        return "HIGH"
    return "CRITICAL"

def alert_threshold() -> float:
    if isinstance(MODEL_BUNDLE, dict):
        try:
            return float(MODEL_BUNDLE.get("prototype_oof_f1_threshold", 0.30))
        except Exception:
            pass
    return 0.30

class PredictPayload(BaseModel):
    elevation_m: Optional[float] = None
    slope_deg: Optional[float] = None
    aspect_sin: Optional[float] = None
    aspect_cos: Optional[float] = None
    curvature_1_per_m: Optional[float] = None
    rainfall_24h_mm: Optional[float] = None
    rainfall_72h_mm: Optional[float] = None
    rainfall_7d_mm: Optional[float] = None
    land_cover: Optional[str] = None
    ndvi_model_value: Optional[float] = None
    ndvi_missing: Optional[int] = None
    soil_moisture_surface_m3_m3: Optional[float] = None
    soil_moisture_7_28cm_m3_m3: Optional[float] = None
    soil_moisture_28_100cm_m3_m3: Optional[float] = None
    soil_moisture_0_100cm_m3_m3: Optional[float] = None
    soil_moisture_surface_3d_mean_m3_m3: Optional[float] = None
    soil_moisture_surface_7d_mean_m3_m3: Optional[float] = None
    soil_moisture_surface_anomaly_vs_7d: Optional[float] = None
    distance_to_road_m: Optional[float] = None
    distance_to_river_m: Optional[float] = None

class StatusPatch(BaseModel):
    status: str

class RoadCreate(BaseModel):
    road_name: str
    state: Optional[str] = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    status: str = "UNKNOWN"
    notes: Optional[str] = ""

class RoadPatch(BaseModel):
    status: str
    notes: Optional[str] = None

app = FastAPI(
    title="NER Landslide Intelligence Platform",
    version="2.0.0",
    description="Hackathon-ready AI + GIS landslide risk decision-support platform."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    load_all()
    init_db()
    seed_alerts()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_BUNDLE is not None,
        "prediction_rows": 0 if PREDICTIONS is None else int(len(PREDICTIONS)),
        "prediction_file": None if PREDICTIONS_PATH is None else PREDICTIONS_PATH.name,
        "started_at": STARTED_AT,
    }

@app.get("/api/system/status")
def system_status():
    return {
        "application": "LIVE",
        "model": "LIVE" if MODEL_BUNDLE is not None else "UNAVAILABLE",
        "risk_map": "CACHED" if PREDICTIONS is not None else "UNAVAILABLE",
        "rainfall": "CACHED",
        "soil_moisture": "CACHED",
        "satellite": "CACHED",
        "terrain": "CACHED",
        "osm": "CACHED",
        "database": "LIVE",
        "last_inference": SUMMARY.get("inference_date"),
        "prediction_file": None if PREDICTIONS_PATH is None else PREDICTIONS_PATH.name,
        "note": "CACHED means Stage 10 processed data, not a live external feed."
    }

@app.get("/api/states")
def states():
    return sorted(PREDICTIONS["state"].dropna().astype(str).unique().tolist())

@app.get("/api/summary")
def summary():
    df = PREDICTIONS
    by_state = (
        df.groupby("state")
        .agg(
            monitored=("grid_id", "count"),
            mean_risk=("risk_probability", "mean"),
            max_risk=("risk_probability", "max"),
            alerts=("prototype_alert", "sum"),
        )
        .round(4)
        .reset_index()
        .sort_values("mean_risk", ascending=False)
    )
    return {
        "total_monitored_zones": int(len(df)),
        "low_zones": int((df["risk_level"] == "LOW").sum()),
        "moderate_zones": int((df["risk_level"] == "MODERATE").sum()),
        "high_risk_zones": int((df["risk_level"] == "HIGH").sum()),
        "critical_zones": int((df["risk_level"] == "CRITICAL").sum()),
        "prototype_alerts": int(df["prototype_alert"].sum()),
        "maximum_risk_probability": float(df["risk_probability"].max()),
        "mean_risk_probability": float(df["risk_probability"].mean()),
        "most_affected_state_by_mean_risk": None if by_state.empty else str(by_state.iloc[0]["state"]),
        "inference_date": SUMMARY.get("inference_date"),
        "alert_threshold": alert_threshold(),
        "model_validation": {
            "ensemble_roc_auc_geographic_cv": 0.8060,
            "ensemble_pr_auc_geographic_cv": 0.8049,
            "oof_threshold_f1": 0.7545,
            "oof_threshold_precision": 0.6938,
            "oof_threshold_recall": 0.8268,
            "note": "ROC-AUC is not accuracy; threshold metrics are prototype OOF metrics."
        },
    }

@app.get("/api/analytics")
def analytics():
    df = PREDICTIONS
    state_summary = (
        df.groupby("state")
        .agg(
            monitored_zones=("grid_id", "count"),
            mean_risk=("risk_probability", "mean"),
            max_risk=("risk_probability", "max"),
            prototype_alerts=("prototype_alert", "sum"),
            mean_rainfall_24h=("rainfall_24h_mm", "mean"),
            mean_slope=("slope_deg", "mean"),
        )
        .round(4)
        .reset_index()
        .to_dict(orient="records")
    )
    risk_counts = (
        df["risk_level"]
        .value_counts()
        .reindex(["LOW", "MODERATE", "HIGH", "CRITICAL"], fill_value=0)
        .to_dict()
    )
    return {
        "risk_counts": {k: int(v) for k, v in risk_counts.items()},
        "state_summary": state_summary,
    }

@app.get("/api/risk")
def risk(
    state: Optional[str] = None,
    level: Optional[str] = None,
    alerts_only: bool = False,
    q: Optional[str] = None,
    limit: int = 5000,
):
    df = PREDICTIONS
    if state:
        df = df[df["state"].astype(str).str.lower() == state.lower()]
    if level:
        df = df[df["risk_level"].astype(str).str.upper() == level.upper()]
    if alerts_only:
        df = df[df["prototype_alert"] == True]
    if q:
        qn = q.lower()
        df = df[
            df["state"].astype(str).str.lower().str.contains(qn, na=False)
            | df["grid_id"].astype(str).str.lower().str.contains(qn, na=False)
        ]
    df = df.head(max(1, min(int(limit), 5000)))
    return df.replace({np.nan: None}).to_dict(orient="records")

@app.get("/api/risk/high")
def high_risk(limit: int = 100):
    df = PREDICTIONS.sort_values("risk_probability", ascending=False)
    return df.head(max(1, min(int(limit), 1000))).replace({np.nan: None}).to_dict(orient="records")

@app.get("/api/risk/{grid_id}")
def risk_one(grid_id: str):
    hit = PREDICTIONS[PREDICTIONS["grid_id"].astype(str) == grid_id]
    if hit.empty:
        raise HTTPException(404, "Grid point not found")
    return hit.iloc[0].replace({np.nan: None}).to_dict()

@app.post("/api/predict")
def predict(payload: PredictPayload):
    if MODEL_BUNDLE is None:
        raise HTTPException(503, "Model bundle unavailable")

    features = MODEL_BUNDLE["features"]
    models = MODEL_BUNDLE["models"]
    row = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()

    if row.get("ndvi_missing") is None:
        row["ndvi_missing"] = int(row.get("ndvi_model_value") is None)

    X = pd.DataFrame([{f: row.get(f) for f in features}])
    probs = {name: float(model.predict_proba(X)[:, 1][0]) for name, model in models.items()}

    weights = MODEL_BUNDLE.get("ensemble_weights") or {
        name: 1.0 / len(probs) for name in probs
    }
    total = sum(float(weights.get(k, 0)) for k in probs)
    if total <= 0:
        weights = {k: 1.0 for k in probs}
        total = float(len(probs))

    ensemble = sum(probs[k] * float(weights.get(k, 0)) / total for k in probs)
    threshold = alert_threshold()

    return {
        "model_probabilities": probs,
        "risk_probability": ensemble,
        "risk_level": risk_level(ensemble),
        "prototype_alert_threshold": threshold,
        "prototype_alert": ensemble >= threshold,
        "disclaimer": "Prototype AI-generated risk assessment; not an official government warning."
    }

@app.get("/api/alerts")
def alerts(status: Optional[str] = None):
    if status:
        return rows(
            "SELECT * FROM alerts WHERE status=? ORDER BY risk_probability DESC",
            (status.upper(),)
        )
    return rows("SELECT * FROM alerts ORDER BY risk_probability DESC")

@app.post("/api/alerts/{alert_id}/acknowledge")
def acknowledge(alert_id: int):
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "UPDATE alerts SET status='ACKNOWLEDGED', updated_at=? WHERE id=?",
            (now, alert_id)
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "Alert not found")
    return {"ok": True, "alert_id": alert_id, "status": "ACKNOWLEDGED"}

@app.get("/api/reports")
def reports():
    return rows("SELECT * FROM field_reports ORDER BY id DESC")

@app.post("/api/reports")
async def create_report(
    latitude: float = Form(...),
    longitude: float = Form(...),
    report_type: str = Form(...),
    state: str = Form(""),
    district: str = Form(""),
    description: str = Form(""),
    media: Optional[UploadFile] = File(None),
):
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise HTTPException(400, "Invalid coordinates")

    report_type = report_type.strip().lower()
    if report_type not in ALLOWED_REPORT_TYPES:
        raise HTTPException(400, "Invalid report type")

    media_path = None
    if media and media.filename:
        if media.content_type not in ALLOWED_MEDIA_TYPES:
            raise HTTPException(400, "Unsupported media type")

        payload = await media.read()
        if len(payload) > 20 * 1024 * 1024:
            raise HTTPException(400, "Media exceeds 20 MB prototype limit")

        safe_suffix = Path(media.filename).suffix.lower()
        target = UPLOAD_DIR / f"{int(time.time() * 1000)}_{os.getpid()}{safe_suffix}"
        target.write_bytes(payload)
        media_path = str(target.relative_to(RUNTIME_DIR))

    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("""
            INSERT INTO field_reports
            (created_at, updated_at, latitude, longitude, state, district,
             report_type, description, media_path, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'NEW')
        """, (
            now, now, latitude, longitude, state, district,
            report_type, description, media_path
        ))
        report_id = cur.lastrowid

    return {
        "ok": True,
        "report_id": report_id,
        "status": "NEW",
        "note": "Report remains unverified until reviewed."
    }

@app.patch("/api/reports/{report_id}")
def patch_report(report_id: int, payload: StatusPatch):
    status = payload.status.upper()
    if status not in ALLOWED_REPORT_STATUS:
        raise HTTPException(400, "Invalid report status")

    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "UPDATE field_reports SET status=?, updated_at=? WHERE id=?",
            (status, now, report_id)
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "Report not found")
    return {"ok": True, "report_id": report_id, "status": status}

@app.get("/api/roads")
def roads_api():
    return rows("SELECT * FROM road_status ORDER BY id DESC")

@app.post("/api/roads")
def create_road(payload: RoadCreate):
    status = payload.status.upper()
    if status not in ALLOWED_ROAD_STATUS:
        raise HTTPException(400, "Invalid road status")

    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("""
            INSERT INTO road_status
            (created_at, updated_at, road_name, state, latitude, longitude, status, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now, now, payload.road_name, payload.state,
            payload.latitude, payload.longitude, status, payload.notes
        ))
        road_id = cur.lastrowid

    return {"ok": True, "road_id": road_id, "status": status}

@app.patch("/api/roads/{road_id}")
def patch_road(road_id: int, payload: RoadPatch):
    status = payload.status.upper()
    if status not in ALLOWED_ROAD_STATUS:
        raise HTTPException(400, "Invalid road status")

    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("""
            UPDATE road_status
            SET status=?, notes=COALESCE(?, notes), updated_at=?
            WHERE id=?
        """, (status, payload.notes, now, road_id))
        if cur.rowcount == 0:
            raise HTTPException(404, "Road not found")

    return {"ok": True, "road_id": road_id, "status": status}

@app.get("/api/notification-preview/{grid_id}")
def notification_preview(grid_id: str, lang: str = "en"):
    hit = PREDICTIONS[PREDICTIONS["grid_id"].astype(str) == grid_id]
    if hit.empty:
        raise HTTPException(404, "Grid point not found")

    r = hit.iloc[0]
    selected = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return {
        "language": lang if lang in TRANSLATIONS else "en",
        "title": selected["title"],
        "message": selected["body"].format(
            state=r["state"],
            score=float(r["risk_probability"])
        ),
        "delivery": "MOCK",
        "note": "Preview only; no SMS/push was sent."
    }

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(FRONTEND_DIR / "manifest.webmanifest")

@app.get("/sw.js")
def service_worker():
    return FileResponse(FRONTEND_DIR / "sw.js", media_type="application/javascript")
