from pathlib import Path
import json
import time
import hashlib

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_CANDIDATES = [
    ROOT / "data" / "processed" / "terrain" / "ner_events_with_copernicus_dem.csv",
    ROOT / "data" / "raw" / "historical" / "ner_landslide_events_cleaned.csv",
]

RAW_DIR = ROOT / "data" / "raw" / "rainfall" / "nasa_power"
CACHE_DIR = RAW_DIR / "cache"
PROCESSED_DIR = ROOT / "data" / "processed" / "rainfall"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = PROCESSED_DIR / "ner_events_with_dem_rainfall.csv"
SUMMARY_JSON = PROCESSED_DIR / "rainfall_extraction_summary.json"

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETER = "PRECTOTCORR"

# Request the event day plus 6 previous days.
LOOKBACK_DAYS = 6

REQUEST_DELAY_SECONDS = 0.35
MAX_RETRIES = 4
TIMEOUT_SECONDS = 90

INVALID_POWER_VALUES = {-999.0, -9999.0, -99999.0}


# ============================================================
# HELPERS
# ============================================================

def find_input_file() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the Stage 2 output.\n"
        "Expected:\n"
        "data/processed/terrain/ner_events_with_copernicus_dem.csv"
    )


def cache_key(latitude: float, longitude: float, start: str, end: str) -> str:
    raw = f"{latitude:.6f}|{longitude:.6f}|{start}|{end}|{PARAMETER}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def fetch_power_daily(
    latitude: float,
    longitude: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict:
    start = start_date.strftime("%Y%m%d")
    end = end_date.strftime("%Y%m%d")

    key = cache_key(latitude, longitude, start, end)
    cache_path = CACHE_DIR / f"{key}.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    params = {
        "parameters": PARAMETER,
        "community": "AG",
        "longitude": float(longitude),
        "latitude": float(latitude),
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": "UTC",
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                POWER_URL,
                params=params,
                headers={
                    "User-Agent": "SIH-Landslide-Research/1.0",
                    "Accept": "application/json",
                },
                timeout=TIMEOUT_SECONDS,
            )

            if response.status_code == 200:
                payload = response.json()

                if "messages" in payload and payload["messages"]:
                    # POWER can return informational messages; keep response anyway.
                    pass

                cache_path.write_text(
                    json.dumps(payload, indent=2),
                    encoding="utf-8",
                )

                time.sleep(REQUEST_DELAY_SECONDS)
                return payload

            last_error = RuntimeError(
                f"NASA POWER HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        except Exception as error:
            last_error = error

        if attempt < MAX_RETRIES:
            wait = 1.5 * attempt
            print(
                f"    request failed (attempt {attempt}/{MAX_RETRIES}); "
                f"retrying in {wait:.1f}s"
            )
            time.sleep(wait)

    raise RuntimeError(
        f"NASA POWER request failed after {MAX_RETRIES} attempts: {last_error}"
    )


def clean_power_value(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if value in INVALID_POWER_VALUES:
        return np.nan

    if not np.isfinite(value):
        return np.nan

    # Precipitation cannot physically be negative.
    if value < 0:
        return np.nan

    return value


def extract_rainfall_windows(
    payload: dict,
    event_date: pd.Timestamp,
):
    try:
        values = payload["properties"]["parameter"][PARAMETER]
    except (KeyError, TypeError):
        return None

    daily = {}

    for date_text, value in values.items():
        clean_value = clean_power_value(value)
        try:
            date = pd.to_datetime(date_text, format="%Y%m%d")
        except Exception:
            continue

        daily[date.normalize()] = clean_value

    event_day = event_date.normalize()

    required_dates = [
        event_day - pd.Timedelta(days=offset)
        for offset in range(0, LOOKBACK_DAYS + 1)
    ]

    series = pd.Series(
        {
            date: daily.get(date, np.nan)
            for date in required_dates
        },
        dtype="float64",
    )

    rain_24h = series.get(event_day, np.nan)

    rain_72h_dates = [
        event_day - pd.Timedelta(days=offset)
        for offset in range(0, 3)
    ]

    rain_7d_dates = required_dates

    rain_72_values = series.reindex(rain_72h_dates)
    rain_7d_values = series.reindex(rain_7d_dates)

    # Require complete windows; do not silently sum missing days.
    rain_72h = (
        float(rain_72_values.sum())
        if rain_72_values.notna().all()
        else np.nan
    )

    rain_7d = (
        float(rain_7d_values.sum())
        if rain_7d_values.notna().all()
        else np.nan
    )

    return {
        "rainfall_24h_mm": float(rain_24h)
        if pd.notna(rain_24h)
        else np.nan,
        "rainfall_72h_mm": rain_72h,
        "rainfall_7d_mm": rain_7d,
        "rainfall_daily_values_json": json.dumps(
            {
                date.strftime("%Y-%m-%d"): (
                    None if pd.isna(value) else float(value)
                )
                for date, value in series.sort_index().items()
            }
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("STAGE 3 — NASA POWER HISTORICAL RAINFALL EXTRACTION")
    print("=" * 72)

    input_file = find_input_file()
    print(f"\nInput file:\n{input_file}")

    df = pd.read_csv(input_file)

    required_columns = {"latitude", "longitude", "event_date"}
    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Input CSV is missing required columns: {sorted(missing)}"
        )

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["_event_date_parsed"] = pd.to_datetime(
        df["event_date"],
        errors="coerce",
    )

    valid_mask = (
        df["latitude"].notna()
        & df["longitude"].notna()
        & df["_event_date_parsed"].notna()
    )

    print(f"Total rows: {len(df)}")
    print(f"Rows with valid coordinates + event date: {int(valid_mask.sum())}")

    results = []

    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        result = {
            "rainfall_source": "NASA POWER Daily API / PRECTOTCORR",
            "rainfall_extraction_ok": False,
            "rainfall_24h_mm": np.nan,
            "rainfall_72h_mm": np.nan,
            "rainfall_7d_mm": np.nan,
            "rainfall_daily_values_json": None,
        }

        latitude = row["latitude"]
        longitude = row["longitude"]
        event_date = row["_event_date_parsed"]

        if pd.isna(latitude) or pd.isna(longitude) or pd.isna(event_date):
            results.append(result)
            continue

        start_date = event_date - pd.Timedelta(days=LOOKBACK_DAYS)
        end_date = event_date

        event_id = row.get("event_id", row_number)

        print(
            f"[{row_number}/{len(df)}] event={event_id} "
            f"date={event_date.date()} "
            f"lat={latitude:.4f} lon={longitude:.4f}"
        )

        try:
            payload = fetch_power_daily(
                float(latitude),
                float(longitude),
                start_date,
                end_date,
            )

            rainfall = extract_rainfall_windows(
                payload,
                event_date,
            )

            if rainfall is not None:
                result.update(rainfall)

                result["rainfall_extraction_ok"] = bool(
                    pd.notna(result["rainfall_24h_mm"])
                    and pd.notna(result["rainfall_72h_mm"])
                    and pd.notna(result["rainfall_7d_mm"])
                )

        except Exception as error:
            print(f"    ERROR: {error}")

        results.append(result)

        # Checkpoint after every 25 records so a network interruption
        # does not throw away completed work.
        if row_number % 25 == 0:
            checkpoint = pd.concat(
                [
                    df.iloc[:row_number]
                    .drop(columns=["_event_date_parsed"])
                    .reset_index(drop=True),
                    pd.DataFrame(results),
                ],
                axis=1,
            )

            checkpoint.to_csv(
                PROCESSED_DIR / "rainfall_checkpoint.csv",
                index=False,
            )

    rainfall_df = pd.DataFrame(results)

    output = pd.concat(
        [
            df.drop(columns=["_event_date_parsed"]).reset_index(drop=True),
            rainfall_df.reset_index(drop=True),
        ],
        axis=1,
    )

    output.to_csv(OUTPUT_CSV, index=False)

    successful = int(output["rainfall_extraction_ok"].sum())
    failed = len(output) - successful

    summary = {
        "source": "NASA POWER Daily API",
        "parameter": PARAMETER,
        "time_standard": "UTC",
        "rows": int(len(output)),
        "successful_complete_windows": successful,
        "failed_or_incomplete_windows": failed,
        "lookback_days_including_event_day": LOOKBACK_DAYS + 1,
        "output_csv": str(OUTPUT_CSV),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("STAGE 3 COMPLETE")
    print("=" * 72)

    print(f"Events processed: {len(output)}")
    print(f"Complete rainfall windows: {successful}")
    print(f"Failed/incomplete rainfall windows: {failed}")

    valid = output[output["rainfall_extraction_ok"] == True]

    if not valid.empty:
        print("\nRainfall feature summary:")
        print(
            valid[
                [
                    "rainfall_24h_mm",
                    "rainfall_72h_mm",
                    "rainfall_7d_mm",
                ]
            ].describe().round(3)
        )

    print(f"\nProcessed output:\n{OUTPUT_CSV}")
    print(f"\nCache directory:\n{CACHE_DIR}")
    print(
        "\nNOTE: For historical model calibration, the event-day rainfall is included. "
        "For an operational early-warning forecast, future/live inference must use "
        "observed antecedent rainfall plus forecast precipitation available before "
        "the prediction time."
    )


if __name__ == "__main__":
    main()
