from pathlib import Path
from datetime import timedelta
import hashlib
import json
import shutil
import time

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" / "background_features"
    / "background_non_osm_features.csv"
)
BACKUP_FILE = (
    ROOT / "data" / "processed" / "background_features"
    / "background_non_osm_features_before_rainfall_repair.csv"
)
CHECKPOINT_FILE = (
    ROOT / "data" / "processed" / "background_features"
    / "background_rainfall_repair_checkpoint.csv"
)
SUMMARY_FILE = (
    ROOT / "data" / "processed" / "background_features"
    / "background_rainfall_repair_summary.json"
)
CACHE_DIR = (
    ROOT / "data" / "raw" / "rainfall"
    / "nasa_power" / "background_cache"
)

CACHE_DIR.mkdir(parents=True, exist_ok=True)

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETER = "PRECTOTCORR"

TIMEOUT_SECONDS = 45
MAX_ATTEMPTS = 5
REQUEST_DELAY_SECONDS = 0.35


def cache_path(lat, lon, event_date):
    raw = f"{lat:.6f}|{lon:.6f}|{event_date}"
    key = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.json"


def fetch_power(lat, lon, event_date):
    event_date = pd.Timestamp(event_date).date()
    start_date = event_date - timedelta(days=6)

    cpath = cache_path(lat, lon, event_date.isoformat())

    if cpath.exists():
        try:
            return json.loads(cpath.read_text(encoding="utf-8"))
        except Exception:
            pass

    params = {
        "parameters": PARAMETER,
        "community": "AG",
        "longitude": f"{lon:.6f}",
        "latitude": f"{lat:.6f}",
        "start": start_date.strftime("%Y%m%d"),
        "end": event_date.strftime("%Y%m%d"),
        "format": "JSON",
    }

    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                POWER_URL,
                params=params,
                timeout=TIMEOUT_SECONDS,
                headers={"User-Agent": "SIH-Landslide-Research/1.0"},
            )

            if response.status_code == 200:
                payload = response.json()

                values = (
                    payload.get("properties", {})
                    .get("parameter", {})
                    .get(PARAMETER, {})
                )

                if not isinstance(values, dict) or not values:
                    raise RuntimeError("NASA POWER response missing PRECTOTCORR values")

                cpath.write_text(
                    json.dumps(payload, indent=2),
                    encoding="utf-8",
                )

                time.sleep(REQUEST_DELAY_SECONDS)
                return payload

            last_error = RuntimeError(
                f"HTTP {response.status_code}: {response.text[:180]}"
            )

        except Exception as exc:
            last_error = exc

        if attempt < MAX_ATTEMPTS:
            wait = min(8.0, 1.5 * attempt)
            print(f"      retrying in {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"NASA POWER failed after {MAX_ATTEMPTS} attempts: {last_error}")


def parse_rainfall(payload, event_date):
    event_date = pd.Timestamp(event_date).date()
    start_date = event_date - timedelta(days=6)

    raw = (
        payload.get("properties", {})
        .get("parameter", {})
        .get(PARAMETER, {})
    )

    daily = []
    dates = []

    for i in range(7):
        day = start_date + timedelta(days=i)
        key = day.strftime("%Y%m%d")
        value = raw.get(key, np.nan)

        try:
            value = float(value)
        except Exception:
            value = np.nan

        # NASA POWER missing-value sentinels are large negative values.
        if pd.isna(value) or value <= -900:
            value = np.nan

        dates.append(day.isoformat())
        daily.append(value)

    # Order is oldest -> event day.
    arr = np.asarray(daily, dtype=float)

    if np.isnan(arr).any():
        raise RuntimeError(
            f"Incomplete 7-day rainfall window: {daily}"
        )

    rainfall_24h = float(arr[-1])
    rainfall_72h = float(arr[-3:].sum())
    rainfall_7d = float(arr.sum())

    daily_json = json.dumps(
        [
            {"date": d, "rainfall_mm": float(v)}
            for d, v in zip(dates, arr)
        ],
        ensure_ascii=False,
    )

    return rainfall_24h, rainfall_72h, rainfall_7d, daily_json


def save_progress(df):
    df.to_csv(INPUT_FILE, index=False)
    df.to_csv(CHECKPOINT_FILE, index=False)


def main():
    print("=" * 84)
    print("STAGE 7B-1R — REPAIR BACKGROUND NASA POWER RAINFALL")
    print("=" * 84)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    if not BACKUP_FILE.exists():
        shutil.copy2(INPUT_FILE, BACKUP_FILE)
        print(f"Backup created:\n{BACKUP_FILE}\n")

    df = pd.read_csv(INPUT_FILE)

    required = ["background_id", "latitude", "longitude", "event_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    for col, default in {
        "rainfall_24h_mm": np.nan,
        "rainfall_72h_mm": np.nan,
        "rainfall_7d_mm": np.nan,
        "rainfall_daily_values_json": None,
        "rainfall_source": None,
        "rainfall_extraction_ok": False,
    }.items():
        if col not in df.columns:
            df[col] = default

    complete_before = (
        df["rainfall_24h_mm"].notna()
        & df["rainfall_72h_mm"].notna()
        & df["rainfall_7d_mm"].notna()
    )

    todo = df.index[~complete_before].tolist()

    print(f"Rows: {len(df)}")
    print(f"Rainfall complete before repair: {int(complete_before.sum())}/{len(df)}")
    print(f"Rows to process: {len(todo)}")

    failures = []

    for n, idx in enumerate(todo, start=1):
        row = df.loc[idx]

        bgid = str(row["background_id"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        event_date = str(row["event_date"])

        print(
            f"[{n}/{len(todo)}] {bgid} "
            f"lat={lat:.5f} lon={lon:.5f} date={event_date}"
        )

        try:
            payload = fetch_power(lat, lon, event_date)
            r24, r72, r7, daily_json = parse_rainfall(payload, event_date)

            df.at[idx, "rainfall_24h_mm"] = r24
            df.at[idx, "rainfall_72h_mm"] = r72
            df.at[idx, "rainfall_7d_mm"] = r7
            df.at[idx, "rainfall_daily_values_json"] = daily_json
            df.at[idx, "rainfall_source"] = "NASA POWER Daily PRECTOTCORR"
            df.at[idx, "rainfall_extraction_ok"] = True

            print(
                f"    24h={r24:.2f} mm | "
                f"72h={r72:.2f} mm | "
                f"7d={r7:.2f} mm"
            )

        except Exception as exc:
            df.at[idx, "rainfall_extraction_ok"] = False
            failures.append(
                {
                    "background_id": bgid,
                    "error": str(exc),
                }
            )
            print(f"    ERROR: {str(exc)[:220]}")

        # Safe resume after interruption.
        save_progress(df)

    complete_after = (
        df["rainfall_24h_mm"].notna()
        & df["rainfall_72h_mm"].notna()
        & df["rainfall_7d_mm"].notna()
    )

    summary = {
        "rows": int(len(df)),
        "complete_before": int(complete_before.sum()),
        "complete_after": int(complete_after.sum()),
        "remaining_missing": int((~complete_after).sum()),
        "failures": failures,
        "updated_file": str(INPUT_FILE),
        "backup_file": str(BACKUP_FILE),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 84)
    print("STAGE 7B-1R COMPLETE")
    print("=" * 84)
    print(f"Rainfall complete: {summary['complete_after']}/{len(df)}")
    print(f"Still missing: {summary['remaining_missing']}")
    print(f"\nUpdated file:\n{INPUT_FILE}")
    print(f"\nSummary:\n{SUMMARY_FILE}")

    if summary["remaining_missing"] == 0:
        print(
            "\nNEXT: rerun 07b3 merge, then Stage 8, then Stage 9C audit "
            "before retraining Stage 9."
        )


if __name__ == "__main__":
    main()
