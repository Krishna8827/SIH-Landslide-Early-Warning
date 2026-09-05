from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_CANDIDATES = [
    ROOT
    / "data"
    / "processed"
    / "landcover_ndvi"
    / "ner_events_with_dem_rainfall_landcover_ndvi.csv",
    ROOT
    / "data"
    / "processed"
    / "rainfall"
    / "ner_events_with_dem_rainfall.csv",
]

RAW_DIR = ROOT / "data" / "raw" / "soil_moisture" / "era5_land"
CACHE_DIR = RAW_DIR / "cache"
OUT_DIR = ROOT / "data" / "processed" / "soil_moisture"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = (
    OUT_DIR
    / "ner_events_with_dem_rainfall_landcover_ndvi_soil_moisture.csv"
)

SUMMARY_JSON = OUT_DIR / "soil_moisture_summary.json"

API_URL = "https://archive-api.open-meteo.com/v1/archive"

MODEL = "era5_land"

DAILY_VARIABLES = [
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
    "soil_moisture_28_to_100cm_mean",
    "soil_moisture_0_to_100cm_mean",
]

LOOKBACK_DAYS = 6

MAX_RETRIES = 4
TIMEOUT_SECONDS = 90
REQUEST_DELAY_SECONDS = 0.25


# ============================================================
# HELPERS
# ============================================================

def find_input_file() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the Stage 4 output.\n"
        "Expected:\n"
        "data/processed/landcover_ndvi/"
        "ner_events_with_dem_rainfall_landcover_ndvi.csv"
    )


def cache_key(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> str:
    raw = (
        f"{latitude:.6f}|{longitude:.6f}|"
        f"{start_date}|{end_date}|{MODEL}|"
        + ",".join(DAILY_VARIABLES)
    )

    return hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()


def fetch_era5_land(
    latitude: float,
    longitude: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict:
    start_text = start_date.strftime("%Y-%m-%d")
    end_text = end_date.strftime("%Y-%m-%d")

    key = cache_key(
        latitude,
        longitude,
        start_text,
        end_text,
    )

    cache_path = CACHE_DIR / f"{key}.json"

    if cache_path.exists():
        return json.loads(
            cache_path.read_text(encoding="utf-8")
        )

    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "start_date": start_text,
        "end_date": end_text,
        "daily": ",".join(DAILY_VARIABLES),
        "models": MODEL,
        "timezone": "GMT",
        "cell_selection": "land",
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=TIMEOUT_SECONDS,
                headers={
                    "User-Agent": "SIH-Landslide-Research/1.0",
                    "Accept": "application/json",
                },
            )

            if response.status_code == 200:
                payload = response.json()

                if payload.get("error"):
                    raise RuntimeError(
                        payload.get(
                            "reason",
                            "Open-Meteo returned an API error.",
                        )
                    )

                cache_path.write_text(
                    json.dumps(
                        payload,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                time.sleep(REQUEST_DELAY_SECONDS)

                return payload

            last_error = RuntimeError(
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        except Exception as error:
            last_error = error

        if attempt < MAX_RETRIES:
            wait_seconds = 1.5 * attempt

            print(
                f"    request failed "
                f"(attempt {attempt}/{MAX_RETRIES}); "
                f"retrying in {wait_seconds:.1f}s"
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        "ERA5-Land request failed after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )


def clean_value(value):
    if value is None:
        return np.nan

    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(value):
        return np.nan

    # Volumetric soil water content should not be negative.
    if value < 0:
        return np.nan

    return value


def payload_to_daily_frame(payload: dict) -> pd.DataFrame:
    daily = payload.get("daily")

    if not isinstance(daily, dict):
        return pd.DataFrame()

    times = daily.get("time")

    if not isinstance(times, list):
        return pd.DataFrame()

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                times,
                errors="coerce",
                utc=True,
            )
        }
    )

    for variable in DAILY_VARIABLES:
        values = daily.get(variable)

        if not isinstance(values, list):
            values = [None] * len(frame)

        # Defensive alignment.
        if len(values) < len(frame):
            values = values + [None] * (
                len(frame) - len(values)
            )

        if len(values) > len(frame):
            values = values[: len(frame)]

        frame[variable] = [
            clean_value(value)
            for value in values
        ]

    frame = frame.dropna(
        subset=["date"]
    ).copy()

    frame["date"] = (
        frame["date"]
        .dt.tz_convert(None)
        .dt.normalize()
    )

    return frame


def mean_if_complete(series: pd.Series):
    if series.empty:
        return np.nan

    if not series.notna().all():
        return np.nan

    return float(series.mean())


def extract_features(
    payload: dict,
    event_date: pd.Timestamp,
):
    frame = payload_to_daily_frame(payload)

    if frame.empty:
        return None

    event_day = pd.Timestamp(event_date)

    if event_day.tzinfo is not None:
        event_day = event_day.tz_convert(None)

    event_day = event_day.normalize()

    indexed = frame.set_index("date")

    result = {}

    # --------------------------------------------------------
    # Event-day soil moisture by depth
    # --------------------------------------------------------

    day_to_output = {
        "soil_moisture_0_to_7cm_mean":
            "soil_moisture_surface_m3_m3",
        "soil_moisture_7_to_28cm_mean":
            "soil_moisture_7_28cm_m3_m3",
        "soil_moisture_28_to_100cm_mean":
            "soil_moisture_28_100cm_m3_m3",
        "soil_moisture_0_to_100cm_mean":
            "soil_moisture_0_100cm_m3_m3",
    }

    for variable, output_name in day_to_output.items():
        if (
            event_day in indexed.index
            and variable in indexed.columns
        ):
            value = indexed.loc[
                event_day,
                variable,
            ]

            if isinstance(value, pd.Series):
                value = value.iloc[0]

            result[output_name] = (
                float(value)
                if pd.notna(value)
                else np.nan
            )
        else:
            result[output_name] = np.nan

    # --------------------------------------------------------
    # Antecedent surface-soil-moisture means
    # Event day + previous days
    # --------------------------------------------------------

    surface_variable = "soil_moisture_0_to_7cm_mean"

    dates_3d = [
        event_day - pd.Timedelta(days=offset)
        for offset in range(0, 3)
    ]

    dates_7d = [
        event_day - pd.Timedelta(days=offset)
        for offset in range(0, 7)
    ]

    surface_3d = indexed[
        surface_variable
    ].reindex(dates_3d)

    surface_7d = indexed[
        surface_variable
    ].reindex(dates_7d)

    result["soil_moisture_surface_3d_mean_m3_m3"] = (
        mean_if_complete(surface_3d)
    )

    result["soil_moisture_surface_7d_mean_m3_m3"] = (
        mean_if_complete(surface_7d)
    )

    # --------------------------------------------------------
    # Simple wetting trend
    # Event-day surface moisture minus 7-day mean.
    # Positive => wetter than the recent 7-day average.
    # --------------------------------------------------------

    event_surface = result[
        "soil_moisture_surface_m3_m3"
    ]

    mean_7d = result[
        "soil_moisture_surface_7d_mean_m3_m3"
    ]

    if (
        pd.notna(event_surface)
        and pd.notna(mean_7d)
    ):
        result[
            "soil_moisture_surface_anomaly_vs_7d"
        ] = float(
            event_surface - mean_7d
        )
    else:
        result[
            "soil_moisture_surface_anomaly_vs_7d"
        ] = np.nan

    # Preserve the 7-day daily surface values for auditing.
    daily_lookup = (
        indexed[surface_variable]
        .reindex(dates_7d)
    )

    result[
        "soil_moisture_surface_7d_values_json"
    ] = json.dumps(
        {
            date.strftime("%Y-%m-%d"): (
                None
                if pd.isna(value)
                else float(value)
            )
            for date, value in daily_lookup.sort_index().items()
        }
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 76)
    print("STAGE 5 — HISTORICAL ERA5-LAND SOIL MOISTURE")
    print("=" * 76)

    input_file = find_input_file()

    print(
        f"\nInput file:\n{input_file}"
    )

    df = pd.read_csv(input_file)

    required = {
        "latitude",
        "longitude",
        "event_date",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df["_event_date_parsed"] = pd.to_datetime(
        df["event_date"],
        errors="coerce",
        utc=True,
    )

    valid_mask = (
        df["latitude"].notna()
        & df["longitude"].notna()
        & df["_event_date_parsed"].notna()
    )

    print(
        f"Total rows: {len(df)}"
    )

    print(
        "Rows with valid coordinate + event date: "
        f"{int(valid_mask.sum())}"
    )

    # --------------------------------------------------------
    # Preserve NDVI but create a safer model-ready value.
    # Good + marginal reliability remain usable.
    # Cloudy / missing NDVI become NaN for modelling.
    # --------------------------------------------------------

    if {
        "ndvi",
        "ndvi_quality_ok",
    }.issubset(df.columns):

        quality_bool = (
            df["ndvi_quality_ok"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        )

        df["ndvi_model_value"] = (
            pd.to_numeric(
                df["ndvi"],
                errors="coerce",
            )
            .where(quality_bool)
        )

        print(
            "NDVI values retained for modelling: "
            f"{int(df['ndvi_model_value'].notna().sum())}"
            f"/{len(df)}"
        )

    results = []

    for row_number, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):
        result = {
            "soil_moisture_source":
                "ERA5-Land via Open-Meteo Historical Weather API",
            "soil_moisture_model":
                "era5_land",
            "soil_moisture_extraction_ok":
                False,
            "soil_moisture_surface_m3_m3":
                np.nan,
            "soil_moisture_7_28cm_m3_m3":
                np.nan,
            "soil_moisture_28_100cm_m3_m3":
                np.nan,
            "soil_moisture_0_100cm_m3_m3":
                np.nan,
            "soil_moisture_surface_3d_mean_m3_m3":
                np.nan,
            "soil_moisture_surface_7d_mean_m3_m3":
                np.nan,
            "soil_moisture_surface_anomaly_vs_7d":
                np.nan,
            "soil_moisture_surface_7d_values_json":
                None,
        }

        latitude = row["latitude"]
        longitude = row["longitude"]
        event_date = row["_event_date_parsed"]

        if (
            pd.isna(latitude)
            or pd.isna(longitude)
            or pd.isna(event_date)
        ):
            results.append(result)
            continue

        start_date = (
            event_date
            - pd.Timedelta(days=LOOKBACK_DAYS)
        )

        end_date = event_date

        event_id = row.get(
            "event_id",
            row_number,
        )

        print(
            f"[{row_number}/{len(df)}] "
            f"event={event_id} "
            f"date={event_date.date()} "
            f"lat={float(latitude):.4f} "
            f"lon={float(longitude):.4f}"
        )

        try:
            payload = fetch_era5_land(
                float(latitude),
                float(longitude),
                start_date,
                end_date,
            )

            features = extract_features(
                payload,
                event_date,
            )

            if features is not None:
                result.update(features)

                required_values = [
                    result[
                        "soil_moisture_surface_m3_m3"
                    ],
                    result[
                        "soil_moisture_surface_3d_mean_m3_m3"
                    ],
                    result[
                        "soil_moisture_surface_7d_mean_m3_m3"
                    ],
                ]

                result[
                    "soil_moisture_extraction_ok"
                ] = all(
                    pd.notna(value)
                    for value in required_values
                )

        except Exception as error:
            print(
                f"    ERROR: {error}"
            )

        results.append(result)

        # Save progress every 25 events.
        if row_number % 25 == 0:
            checkpoint = pd.concat(
                [
                    df.iloc[:row_number]
                    .drop(
                        columns=[
                            "_event_date_parsed"
                        ]
                    )
                    .reset_index(drop=True),
                    pd.DataFrame(results),
                ],
                axis=1,
            )

            checkpoint.to_csv(
                OUT_DIR
                / "soil_moisture_checkpoint.csv",
                index=False,
            )

    soil_df = pd.DataFrame(results)

    output = pd.concat(
        [
            df.drop(
                columns=[
                    "_event_date_parsed"
                ]
            ).reset_index(drop=True),
            soil_df.reset_index(drop=True),
        ],
        axis=1,
    )

    output.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    successful = int(
        output[
            "soil_moisture_extraction_ok"
        ].sum()
    )

    failed = (
        len(output)
        - successful
    )

    summary = {
        "rows": int(len(output)),
        "source":
            "ERA5-Land via Open-Meteo Historical Weather API",
        "model": MODEL,
        "resolution_note":
            "ERA5-Land is approximately 0.1 degree (~11 km) reanalysis.",
        "complete_soil_moisture_windows":
            successful,
        "failed_or_incomplete_windows":
            failed,
        "output_csv":
            str(OUTPUT_CSV),
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 76
    )

    print(
        "STAGE 5 COMPLETE"
    )

    print(
        "=" * 76
    )

    print(
        f"Events processed: {len(output)}"
    )

    print(
        "Complete soil-moisture windows: "
        f"{successful}"
    )

    print(
        "Failed/incomplete soil-moisture windows: "
        f"{failed}"
    )

    valid_output = output[
        output[
            "soil_moisture_extraction_ok"
        ] == True
    ]

    if not valid_output.empty:
        columns = [
            "soil_moisture_surface_m3_m3",
            "soil_moisture_surface_3d_mean_m3_m3",
            "soil_moisture_surface_7d_mean_m3_m3",
            "soil_moisture_7_28cm_m3_m3",
            "soil_moisture_28_100cm_m3_m3",
            "soil_moisture_0_100cm_m3_m3",
        ]

        available_columns = [
            column
            for column in columns
            if column in valid_output.columns
        ]

        print(
            "\nSoil-moisture feature summary:"
        )

        print(
            valid_output[
                available_columns
            ]
            .describe()
            .round(4)
        )

    print(
        f"\nProcessed output:\n{OUTPUT_CSV}"
    )

    print(
        f"\nRaw response cache:\n{CACHE_DIR}"
    )

    print(
        "\nIMPORTANT: ERA5-Land is a reanalysis/modelled historical "
        "soil-moisture product, not an in-situ sensor measurement. "
        "Its coarse ~11 km grid should be treated as a regional "
        "dynamic wetness feature, especially in mountainous terrain."
    )


if __name__ == "__main__":
    main()
