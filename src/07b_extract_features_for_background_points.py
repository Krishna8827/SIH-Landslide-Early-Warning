from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as warp_transform
from pystac_client import Client
import planetary_computer


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_CANDIDATES = [
    ROOT / "data" / "processed" / "background" / "ner_background_points_matched.csv",
]

OUT_DIR = ROOT / "data" / "processed" / "background_features"
RAW_DEM_DIR = ROOT / "data" / "raw" / "dem" / "copernicus_glo30"
SAT_META_DIR = ROOT / "data" / "raw" / "satellite_metadata"
POWER_CACHE_DIR = ROOT / "data" / "raw" / "rainfall" / "nasa_power" / "cache"
SOIL_CACHE_DIR = ROOT / "data" / "raw" / "soil_moisture" / "era5_land" / "cache"
OSM_CACHE_DIR = ROOT / "data" / "raw" / "osm" / "overpass_cache"

OUT_DIR.mkdir(parents=True, exist_ok=True)
POWER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SOIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUT_DIR / "ner_background_points_with_all_features.csv"
SUMMARY_JSON = OUT_DIR / "background_features_summary.json"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
NER_BBOX = [87.5, 21.5, 97.5, 29.5]

# DEM
DEM_COLLECTION = "cop-dem-glo-30"
WINDOW_RADIUS = 4

# WorldCover / NDVI
WORLDCOVER_COLLECTION = "esa-worldcover"
MODIS_COLLECTION = "modis-13Q1-061"
WORLDCOVER_REFERENCE_YEAR = 2021
MODIS_NDVI_ASSET = "250m_16_days_NDVI"
MODIS_RELIABILITY_ASSET = "250m_16_days_pixel_reliability"
MAX_NDVI_DATE_DISTANCE_DAYS = 20

WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

MODIS_RELIABILITY = {
    0: "Good data",
    1: "Marginal data",
    2: "Snow/Ice",
    3: "Cloudy",
}

# Rainfall
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMETER = "PRECTOTCORR"
RAIN_LOOKBACK_DAYS = 6
POWER_TIMEOUT = 90
POWER_MAX_RETRIES = 4
POWER_REQUEST_DELAY_SECONDS = 0.25
INVALID_POWER_VALUES = {-999.0, -9999.0, -99999.0}

# Soil moisture
SOIL_API_URL = "https://archive-api.open-meteo.com/v1/archive"
SOIL_MODEL = "era5_land"
SOIL_DAILY_VARIABLES = [
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
    "soil_moisture_28_to_100cm_mean",
    "soil_moisture_0_to_100cm_mean",
]
SOIL_LOOKBACK_DAYS = 6
SOIL_TIMEOUT = 90
SOIL_MAX_RETRIES = 4
SOIL_REQUEST_DELAY_SECONDS = 0.20

# OSM / Overpass
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
SEARCH_RADII_M = [5000, 15000, 30000]
ROAD_REGEX = (
    "motorway|motorway_link|trunk|trunk_link|"
    "primary|primary_link|secondary|secondary_link|"
    "tertiary|tertiary_link|unclassified|residential|"
    "service|living_street|road|track"
)
WATERWAY_REGEX = "river|stream|canal|drain"
OVERPASS_MAX_RETRIES_PER_ENDPOINT = 2
OVERPASS_TIMEOUT_SECONDS = 90
OVERPASS_REQUEST_DELAY_SECONDS = 0.25


# ============================================================
# GENERIC HELPERS
# ============================================================

def find_input_file() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find background point file:\n"
        "data/processed/background/ner_background_points_matched.csv"
    )


def point_inside_bbox(lon: float, lat: float, bbox) -> bool:
    west, south, east, north = bbox
    eps = 1e-9
    return (
        west - eps <= lon <= east + eps
        and south - eps <= lat <= north + eps
    )


def transformed_xy(dataset, lon_values, lat_values):
    lon_values = list(map(float, lon_values))
    lat_values = list(map(float, lat_values))

    if dataset.crs is None:
        raise RuntimeError("Raster has no CRS.")

    if dataset.crs.to_string().upper() in {"EPSG:4326", "OGC:CRS84"}:
        return list(zip(lon_values, lat_values))

    xs, ys = warp_transform(
        "EPSG:4326",
        dataset.crs,
        lon_values,
        lat_values,
    )

    return list(zip(xs, ys))


def sample_band(dataset, lon_values, lat_values):
    coords = transformed_xy(dataset, lon_values, lat_values)
    samples = []

    for sample in dataset.sample(coords, indexes=1, masked=True):
        value = sample[0]

        if np.ma.is_masked(value):
            samples.append(np.nan)
        else:
            try:
                value = float(value)
            except Exception:
                value = np.nan

            if dataset.nodata is not None and value == dataset.nodata:
                value = np.nan

            samples.append(value)

    return samples


def open_remote_raster(href):
    signed_href = planetary_computer.sign(href)
    return rasterio.open(signed_href)


# ============================================================
# STAC HELPERS
# ============================================================

def item_time_bounds(item):
    start = item.properties.get("start_datetime")
    end = item.properties.get("end_datetime")

    if start:
        start = pd.to_datetime(start, utc=True, errors="coerce")
    if end:
        end = pd.to_datetime(end, utc=True, errors="coerce")

    if item.datetime is not None:
        dt = pd.Timestamp(item.datetime)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")

        if start is None or pd.isna(start):
            start = dt
        if end is None or pd.isna(end):
            end = dt

    if start is None or pd.isna(start):
        start = pd.NaT
    if end is None or pd.isna(end):
        end = start

    return start, end


def item_midpoint(item):
    start, end = item_time_bounds(item)
    if pd.isna(start):
        return pd.NaT
    if pd.isna(end):
        return start
    return start + (end - start) / 2


# ============================================================
# DEM HELPERS
# ============================================================

def choose_item_for_point(items, lon: float, lat: float):
    matches = [item for item in items if point_inside_bbox(lon, lat, item.bbox)]
    if not matches:
        return None

    matches.sort(
        key=lambda item: abs(
            (item.bbox[2] - item.bbox[0]) *
            (item.bbox[3] - item.bbox[1])
        )
    )
    return matches[0]


def download_file(url: str, destination: Path):
    if destination.exists() and destination.stat().st_size > 0:
        return

    temp_path = destination.with_suffix(destination.suffix + ".part")

    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()

        with open(temp_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    temp_path.replace(destination)


def xy_to_dataset_crs(dataset, lon: float, lat: float):
    if dataset.crs is None:
        raise RuntimeError("DEM raster does not contain a CRS.")

    crs_text = dataset.crs.to_string().upper()
    if crs_text in {"EPSG:4326", "OGC:CRS84"}:
        return lon, lat

    xs, ys = warp_transform("EPSG:4326", dataset.crs, [lon], [lat])
    return xs[0], ys[0]


def pixel_spacing_m(dataset, lat: float):
    transform = dataset.transform

    if dataset.crs and dataset.crs.is_projected:
        dx = abs(transform.a)
        dy = abs(transform.e)
        return dx, dy

    lat_rad = math.radians(lat)
    dx = abs(transform.a) * 111_320.0 * max(math.cos(lat_rad), 1e-6)
    dy = abs(transform.e) * 110_574.0
    return dx, dy


def extract_terrain_features(dataset, lon: float, lat: float):
    x, y = xy_to_dataset_crs(dataset, lon, lat)
    row, col = dataset.index(x, y)

    radius = WINDOW_RADIUS
    size = 2 * radius + 1

    window = Window(
        col_off=col - radius,
        row_off=row - radius,
        width=size,
        height=size,
    )

    arr = dataset.read(
        1,
        window=window,
        boundless=True,
        masked=True,
    ).astype("float64")

    if np.ma.isMaskedArray(arr):
        z = arr.filled(np.nan)
    else:
        z = np.asarray(arr, dtype="float64")

    if dataset.nodata is not None:
        z[z == dataset.nodata] = np.nan

    if z.shape != (size, size):
        return None

    center = radius

    if not np.isfinite(z[center, center]):
        return None

    if np.isnan(z).any():
        local_median = np.nanmedian(z)
        if not np.isfinite(local_median):
            return None
        z = np.where(np.isnan(z), local_median, z)

    dx, dy = pixel_spacing_m(dataset, lat)

    if dx <= 0 or dy <= 0:
        return None

    dz_dy, dz_dx = np.gradient(z, dy, dx)

    gx = float(dz_dx[center, center])
    gy = float(dz_dy[center, center])

    slope_rad = math.atan(math.sqrt(gx * gx + gy * gy))
    slope_deg = math.degrees(slope_rad)

    aspect_deg = (
        math.degrees(math.atan2(-gx, gy)) + 360.0
    ) % 360.0

    if slope_deg < 0.01:
        aspect_deg = np.nan

    d2z_dy2 = np.gradient(dz_dy, dy, axis=0)
    d2z_dx2 = np.gradient(dz_dx, dx, axis=1)
    curvature = float(
        d2z_dx2[center, center] + d2z_dy2[center, center]
    )

    return {
        "elevation_m": float(z[center, center]),
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "curvature_1_per_m": curvature,
        "dem_pixel_x_m": float(dx),
        "dem_pixel_y_m": float(dy),
        "terrain_extraction_ok": True,
        "dem_source": "Copernicus DEM GLO-30 via Microsoft Planetary Computer",
    }


# ============================================================
# WORLDCOVER / MODIS HELPERS
# ============================================================

def choose_worldcover_item(items, lon: float, lat: float):
    matches = [item for item in items if point_inside_bbox(lon, lat, item.bbox)]
    if not matches:
        return None

    matches.sort(
        key=lambda item: (
            0 if str(item.properties.get("esa_worldcover:product_version", "")).startswith("2") else 1,
            abs((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])),
        )
    )
    return matches[0]


def choose_modis_item(items, lon: float, lat: float, event_date: pd.Timestamp):
    event_date = pd.Timestamp(event_date)
    if event_date.tzinfo is None:
        event_date = event_date.tz_localize("UTC")
    else:
        event_date = event_date.tz_convert("UTC")

    candidates = []

    for item in items:
        if not point_inside_bbox(lon, lat, item.bbox):
            continue

        midpoint = item_midpoint(item)
        if pd.isna(midpoint):
            continue

        start, end = item_time_bounds(item)
        contains_date = (
            not pd.isna(start)
            and not pd.isna(end)
            and start.normalize() <= event_date.normalize() <= end.normalize()
        )

        distance_days = abs((midpoint.normalize() - event_date.normalize()).days)

        if contains_date or distance_days <= MAX_NDVI_DATE_DISTANCE_DAYS:
            candidates.append(
                (0 if contains_date else 1, distance_days, item.id, item)
            )

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][3]


# ============================================================
# NASA POWER RAINFALL HELPERS
# ============================================================

def rainfall_cache_key(latitude: float, longitude: float, start: str, end: str) -> str:
    raw = f"{latitude:.6f}|{longitude:.6f}|{start}|{end}|{POWER_PARAMETER}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def fetch_power_daily(
    latitude: float,
    longitude: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict:
    start = start_date.strftime("%Y%m%d")
    end = end_date.strftime("%Y%m%d")

    key = rainfall_cache_key(latitude, longitude, start, end)
    cache_path = POWER_CACHE_DIR / f"{key}.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    params = {
        "parameters": POWER_PARAMETER,
        "community": "AG",
        "longitude": float(longitude),
        "latitude": float(latitude),
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": "UTC",
    }

    last_error = None

    for attempt in range(1, POWER_MAX_RETRIES + 1):
        try:
            response = requests.get(
                POWER_URL,
                params=params,
                headers={
                    "User-Agent": "SIH-Landslide-Research/1.0",
                    "Accept": "application/json",
                },
                timeout=POWER_TIMEOUT,
            )

            if response.status_code == 200:
                payload = response.json()

                cache_path.write_text(
                    json.dumps(payload, indent=2),
                    encoding="utf-8",
                )

                time.sleep(POWER_REQUEST_DELAY_SECONDS)
                return payload

            last_error = RuntimeError(
                f"NASA POWER HTTP {response.status_code}: {response.text[:400]}"
            )

        except Exception as error:
            last_error = error

        if attempt < POWER_MAX_RETRIES:
            time.sleep(1.5 * attempt)

    raise RuntimeError(
        f"NASA POWER request failed after {POWER_MAX_RETRIES} attempts: {last_error}"
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
    if value < 0:
        return np.nan
    return value


def extract_rainfall_windows(payload: dict, event_date: pd.Timestamp):
    try:
        values = payload["properties"]["parameter"][POWER_PARAMETER]
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
        for offset in range(0, RAIN_LOOKBACK_DAYS + 1)
    ]

    series = pd.Series(
        {date: daily.get(date, np.nan) for date in required_dates},
        dtype="float64",
    )

    rain_24h = series.get(event_day, np.nan)

    rain_72_dates = [event_day - pd.Timedelta(days=offset) for offset in range(0, 3)]
    rain_7d_dates = required_dates

    rain_72_values = series.reindex(rain_72_dates)
    rain_7d_values = series.reindex(rain_7d_dates)

    rain_72h = float(rain_72_values.sum()) if rain_72_values.notna().all() else np.nan
    rain_7d = float(rain_7d_values.sum()) if rain_7d_values.notna().all() else np.nan

    return {
        "rainfall_24h_mm": float(rain_24h) if pd.notna(rain_24h) else np.nan,
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
        "rainfall_source": "NASA POWER Daily API / PRECTOTCORR",
        "rainfall_extraction_ok": bool(
            pd.notna(rain_24h) and pd.notna(rain_72h) and pd.notna(rain_7d)
        ),
    }


# ============================================================
# SOIL MOISTURE HELPERS
# ============================================================

def soil_cache_key(latitude: float, longitude: float, start_date: str, end_date: str) -> str:
    raw = (
        f"{latitude:.6f}|{longitude:.6f}|{start_date}|{end_date}|{SOIL_MODEL}|"
        + ",".join(SOIL_DAILY_VARIABLES)
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def fetch_era5_land(
    latitude: float,
    longitude: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict:
    start_text = start_date.strftime("%Y-%m-%d")
    end_text = end_date.strftime("%Y-%m-%d")

    key = soil_cache_key(latitude, longitude, start_text, end_text)
    cache_path = SOIL_CACHE_DIR / f"{key}.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "start_date": start_text,
        "end_date": end_text,
        "daily": ",".join(SOIL_DAILY_VARIABLES),
        "models": SOIL_MODEL,
        "timezone": "GMT",
        "cell_selection": "land",
    }

    last_error = None

    for attempt in range(1, SOIL_MAX_RETRIES + 1):
        try:
            response = requests.get(
                SOIL_API_URL,
                params=params,
                timeout=SOIL_TIMEOUT,
                headers={
                    "User-Agent": "SIH-Landslide-Research/1.0",
                    "Accept": "application/json",
                },
            )

            if response.status_code == 200:
                payload = response.json()

                if payload.get("error"):
                    raise RuntimeError(
                        payload.get("reason", "Open-Meteo returned an API error.")
                    )

                cache_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                time.sleep(SOIL_REQUEST_DELAY_SECONDS)
                return payload

            last_error = RuntimeError(
                f"HTTP {response.status_code}: {response.text[:300]}"
            )

        except Exception as error:
            last_error = error

        if attempt < SOIL_MAX_RETRIES:
            time.sleep(1.5 * attempt)

    raise RuntimeError(
        f"ERA5-Land request failed after {SOIL_MAX_RETRIES} attempts: {last_error}"
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
        {"date": pd.to_datetime(times, errors="coerce", utc=True)}
    )

    for variable in SOIL_DAILY_VARIABLES:
        values = daily.get(variable)
        if not isinstance(values, list):
            values = [None] * len(frame)

        if len(values) < len(frame):
            values = values + [None] * (len(frame) - len(values))
        if len(values) > len(frame):
            values = values[: len(frame)]

        frame[variable] = [clean_value(value) for value in values]

    frame = frame.dropna(subset=["date"]).copy()
    frame["date"] = frame["date"].dt.tz_convert(None).dt.normalize()
    return frame


def mean_if_complete(series: pd.Series):
    if series.empty:
        return np.nan
    if not series.notna().all():
        return np.nan
    return float(series.mean())


def extract_soil_features(payload: dict, event_date: pd.Timestamp):
    frame = payload_to_daily_frame(payload)
    if frame.empty:
        return None

    event_day = pd.Timestamp(event_date)
    if event_day.tzinfo is not None:
        event_day = event_day.tz_convert(None)
    event_day = event_day.normalize()

    indexed = frame.set_index("date")
    result = {}

    day_to_output = {
        "soil_moisture_0_to_7cm_mean": "soil_moisture_surface_m3_m3",
        "soil_moisture_7_to_28cm_mean": "soil_moisture_7_28cm_m3_m3",
        "soil_moisture_28_to_100cm_mean": "soil_moisture_28_100cm_m3_m3",
        "soil_moisture_0_to_100cm_mean": "soil_moisture_0_100cm_m3_m3",
    }

    for variable, output_name in day_to_output.items():
        if event_day in indexed.index and variable in indexed.columns:
            value = indexed.loc[event_day, variable]
            if isinstance(value, pd.Series):
                value = value.iloc[0]
            result[output_name] = float(value) if pd.notna(value) else np.nan
        else:
            result[output_name] = np.nan

    surface_variable = "soil_moisture_0_to_7cm_mean"

    dates_3d = [event_day - pd.Timedelta(days=offset) for offset in range(0, 3)]
    dates_7d = [event_day - pd.Timedelta(days=offset) for offset in range(0, 7)]

    surface_3d = indexed[surface_variable].reindex(dates_3d)
    surface_7d = indexed[surface_variable].reindex(dates_7d)

    result["soil_moisture_surface_3d_mean_m3_m3"] = mean_if_complete(surface_3d)
    result["soil_moisture_surface_7d_mean_m3_m3"] = mean_if_complete(surface_7d)

    event_surface = result["soil_moisture_surface_m3_m3"]
    mean_7d = result["soil_moisture_surface_7d_mean_m3_m3"]
    result["soil_moisture_surface_anomaly_vs_7d"] = (
        float(event_surface - mean_7d)
        if pd.notna(event_surface) and pd.notna(mean_7d)
        else np.nan
    )

    daily_lookup = indexed[surface_variable].reindex(dates_7d)

    result["soil_moisture_surface_7d_values_json"] = json.dumps(
        {
            date.strftime("%Y-%m-%d"): (
                None if pd.isna(value) else float(value)
            )
            for date, value in daily_lookup.sort_index().items()
        }
    )

    required_values = [
        result["soil_moisture_surface_m3_m3"],
        result["soil_moisture_surface_3d_mean_m3_m3"],
        result["soil_moisture_surface_7d_mean_m3_m3"],
    ]

    result["soil_moisture_source"] = "ERA5-Land via Open-Meteo Historical Weather API"
    result["soil_moisture_model"] = "era5_land"
    result["soil_moisture_extraction_ok"] = all(
        pd.notna(value) for value in required_values
    )
    return result


# ============================================================
# OSM HELPERS
# ============================================================

def overpass_cache_key(
    latitude: float,
    longitude: float,
    radius_m: int,
    need_roads: bool,
    need_waterways: bool,
) -> str:
    raw = (
        f"{latitude:.6f}|{longitude:.6f}|{radius_m}|"
        f"roads={int(need_roads)}|waterways={int(need_waterways)}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_query(
    latitude: float,
    longitude: float,
    radius_m: int,
    need_roads: bool,
    need_waterways: bool,
) -> str:
    clauses = []

    if need_roads:
        clauses.append(
            f'way(around:{radius_m},{latitude:.7f},{longitude:.7f})'
            f'["highway"~"^({ROAD_REGEX})$"];'
        )

    if need_waterways:
        clauses.append(
            f'way(around:{radius_m},{latitude:.7f},{longitude:.7f})'
            f'["waterway"~"^({WATERWAY_REGEX})$"];'
        )

    return (
        '[out:json][timeout:60];\n(\n'
        + "\n".join(f"  {clause}" for clause in clauses)
        + '\n);\nout tags geom;'
    )


def fetch_overpass(
    latitude: float,
    longitude: float,
    radius_m: int,
    need_roads: bool,
    need_waterways: bool,
) -> dict:
    key = overpass_cache_key(
        latitude, longitude, radius_m, need_roads, need_waterways
    )
    cache_path = OSM_CACHE_DIR / f"{key}.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    query = build_query(latitude, longitude, radius_m, need_roads, need_waterways)
    last_error = None

    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(1, OVERPASS_MAX_RETRIES_PER_ENDPOINT + 1):
            try:
                response = requests.post(
                    endpoint,
                    data={"data": query},
                    timeout=OVERPASS_TIMEOUT_SECONDS,
                    headers={
                        "User-Agent": "SIH-Landslide-Research/1.0",
                        "Accept": "application/json",
                    },
                )

                if response.status_code == 200:
                    payload = response.json()

                    if "elements" not in payload:
                        raise RuntimeError(
                            "Overpass response did not contain elements."
                        )

                    cache_path.write_text(
                        json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

                    time.sleep(OVERPASS_REQUEST_DELAY_SECONDS)
                    return payload

                last_error = RuntimeError(
                    f"{endpoint} returned HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )

                if response.status_code == 429 or response.status_code >= 500:
                    time.sleep(1.5 * attempt)

            except Exception as error:
                last_error = error
                if attempt < OVERPASS_MAX_RETRIES_PER_ENDPOINT:
                    time.sleep(1.5 * attempt)

    raise RuntimeError(f"All Overpass endpoints failed. Last error: {last_error}")


def local_xy_m(latitude: float, longitude: float, origin_lat: float, origin_lon: float):
    y = (latitude - origin_lat) * 110_574.0
    x = (longitude - origin_lon) * (
        111_320.0 * math.cos(math.radians(origin_lat))
    )
    return x, y


def distance_origin_to_segment_m(ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    denominator = dx * dx + dy * dy

    if denominator <= 0:
        return math.hypot(ax, ay)

    t = -(ax * dx + ay * dy) / denominator
    t = max(0.0, min(1.0, t))

    px = ax + t * dx
    py = ay + t * dy
    return math.hypot(px, py)


def distance_to_geometry_m(origin_lat: float, origin_lon: float, geometry):
    if not geometry:
        return np.nan

    points = []
    for point in geometry:
        try:
            lat = float(point["lat"])
            lon = float(point["lon"])
        except (KeyError, TypeError, ValueError):
            continue

        points.append(local_xy_m(lat, lon, origin_lat, origin_lon))

    if not points:
        return np.nan

    if len(points) == 1:
        return math.hypot(points[0][0], points[0][1])

    best = float("inf")
    for index in range(len(points) - 1):
        ax, ay = points[index]
        bx, by = points[index + 1]
        distance = distance_origin_to_segment_m(ax, ay, bx, by)
        if distance < best:
            best = distance

    return best if math.isfinite(best) else np.nan


def split_elements(payload):
    roads = []
    waterways = []

    for element in payload.get("elements", []):
        if element.get("type") != "way":
            continue

        tags = element.get("tags") or {}

        if tags.get("highway"):
            roads.append(element)
        if tags.get("waterway"):
            waterways.append(element)

    return roads, waterways


def nearest_element(elements, latitude: float, longitude: float):
    best_element = None
    best_distance = float("inf")

    for element in elements:
        distance = distance_to_geometry_m(
            latitude, longitude, element.get("geometry")
        )

        if pd.notna(distance) and distance < best_distance:
            best_distance = distance
            best_element = element

    if best_element is None:
        return None

    return {"distance_m": float(best_distance), "element": best_element}


def extract_osm_timestamp(payload):
    try:
        return payload["osm3s"]["timestamp_osm_base"]
    except (KeyError, TypeError):
        return None


def process_location(latitude: float, longitude: float):
    road_result = None
    river_result = None
    road_radius = None
    river_radius = None
    osm_timestamps = []

    for radius_m in SEARCH_RADII_M:
        need_roads = road_result is None
        need_waterways = river_result is None

        if not need_roads and not need_waterways:
            break

        payload = fetch_overpass(
            latitude, longitude, radius_m, need_roads, need_waterways
        )

        timestamp = extract_osm_timestamp(payload)
        if timestamp:
            osm_timestamps.append(timestamp)

        roads, waterways = split_elements(payload)

        if road_result is None and roads:
            road_result = nearest_element(roads, latitude, longitude)
            if road_result is not None:
                road_radius = radius_m

        if river_result is None and waterways:
            river_result = nearest_element(waterways, latitude, longitude)
            if river_result is not None:
                river_radius = radius_m

    result = {
        "distance_to_road_m": np.nan,
        "distance_to_river_m": np.nan,
        "nearest_road_type": None,
        "nearest_road_name": None,
        "nearest_road_osm_id": np.nan,
        "nearest_waterway_type": None,
        "nearest_waterway_name": None,
        "nearest_waterway_osm_id": np.nan,
        "road_search_radius_m": road_radius,
        "river_search_radius_m": river_radius,
        "road_extraction_ok": road_result is not None,
        "river_extraction_ok": river_result is not None,
        "roads_rivers_extraction_ok": (
            road_result is not None and river_result is not None
        ),
        "osm_source": "OpenStreetMap via Overpass API",
        "osm_snapshot_timestamp": (
            max(osm_timestamps) if osm_timestamps else None
        ),
        "distance_method": "local equirectangular point-to-polyline",
    }

    if road_result is not None:
        element = road_result["element"]
        tags = element.get("tags") or {}

        result["distance_to_road_m"] = road_result["distance_m"]
        result["nearest_road_type"] = tags.get("highway")
        result["nearest_road_name"] = tags.get("name")
        result["nearest_road_osm_id"] = element.get("id")

    if river_result is not None:
        element = river_result["element"]
        tags = element.get("tags") or {}

        result["distance_to_river_m"] = river_result["distance_m"]
        result["nearest_waterway_type"] = tags.get("waterway")
        result["nearest_waterway_name"] = tags.get("name")
        result["nearest_waterway_osm_id"] = element.get("id")

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("STAGE 7B — EXTRACT REAL FEATURES FOR BACKGROUND POINTS")
    print("=" * 80)

    input_file = find_input_file()
    print(f"\nInput background file:\n{input_file}")

    df = pd.read_csv(input_file)

    required = {"latitude", "longitude", "event_date"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["_event_date"] = pd.to_datetime(df["event_date"], errors="coerce", utc=True)

    valid = (
        df["latitude"].notna()
        & df["longitude"].notna()
        & df["_event_date"].notna()
    )

    df = df.loc[valid].copy().reset_index(drop=True)

    print(f"Valid background rows: {len(df)}")
    print(f"Unique coordinate pairs: {df[['latitude','longitude']].drop_duplicates().shape[0]}")

    print("\nConnecting to Microsoft Planetary Computer...")
    catalog = Client.open(STAC_URL)

    # --------------------------------------------------------
    # DEM tile lookup and download
    # --------------------------------------------------------
    dem_items = list(
        catalog.search(collections=[DEM_COLLECTION], bbox=NER_BBOX).items()
    )
    if not dem_items:
        raise RuntimeError("No DEM STAC items found.")

    print(f"DEM tiles found in NER bbox: {len(dem_items)}")

    df["dem_tile_id"] = None
    dem_item_by_id = {item.id: item for item in dem_items}

    for idx, row in df.iterrows():
        item = choose_item_for_point(
            dem_items,
            float(row["longitude"]),
            float(row["latitude"]),
        )
        if item is not None:
            df.at[idx, "dem_tile_id"] = item.id

    required_dem_ids = sorted(df["dem_tile_id"].dropna().unique().tolist())
    print(f"DEM tiles required: {len(required_dem_ids)}")

    tile_paths = {}
    for item_id in required_dem_ids:
        item = dem_item_by_id[item_id]
        if "data" not in item.assets:
            raise RuntimeError(f"DEM item {item_id} has no data asset.")

        signed_href = planetary_computer.sign(item.assets["data"].href)
        local_path = RAW_DEM_DIR / f"{item_id}.tif"
        download_file(signed_href, local_path)
        tile_paths[item_id] = local_path

    # --------------------------------------------------------
    # WorldCover + MODIS item lookup
    # --------------------------------------------------------
    wc_items = list(
        catalog.search(
            collections=[WORLDCOVER_COLLECTION],
            bbox=NER_BBOX,
            datetime="2021-01-01/2021-12-31",
        ).items()
    )
    if not wc_items:
        raise RuntimeError("No WorldCover items found.")

    min_date = df["_event_date"].min() - pd.Timedelta(days=20)
    max_date = df["_event_date"].max() + pd.Timedelta(days=20)

    modis_items = list(
        catalog.search(
            collections=[MODIS_COLLECTION],
            bbox=NER_BBOX,
            datetime=f"{min_date.date()}/{max_date.date()}",
        ).items()
    )
    if not modis_items:
        raise RuntimeError("No MODIS items found for the background date range.")

    print(f"WorldCover items: {len(wc_items)}")
    print(f"MODIS items: {len(modis_items)}")

    # --------------------------------------------------------
    # Group by unique coordinate pair for efficient DEM / OSM reuse
    # --------------------------------------------------------
    unique_locations = (
        df[["latitude", "longitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    terrain_map = {}
    osm_map = {}

    print("\nExtracting DEM terrain + OSM distances for unique locations...")

    open_datasets = {}
    try:
        for idx, loc in unique_locations.iterrows():
            lat = float(loc["latitude"])
            lon = float(loc["longitude"])

            # DEM
            tile_id_series = df.loc[
                (df["latitude"] == lat) & (df["longitude"] == lon),
                "dem_tile_id",
            ]
            tile_id = tile_id_series.iloc[0] if not tile_id_series.empty else None

            terrain_result = {
                "elevation_m": np.nan,
                "slope_deg": np.nan,
                "aspect_deg": np.nan,
                "curvature_1_per_m": np.nan,
                "dem_pixel_x_m": np.nan,
                "dem_pixel_y_m": np.nan,
                "terrain_extraction_ok": False,
                "dem_source": "Copernicus DEM GLO-30 via Microsoft Planetary Computer",
            }

            if pd.notna(tile_id):
                tile_id = str(tile_id)
                if tile_id not in open_datasets:
                    open_datasets[tile_id] = rasterio.open(tile_paths[tile_id])

                dataset = open_datasets[tile_id]
                terrain = extract_terrain_features(dataset, lon, lat)
                if terrain is not None:
                    terrain_result.update(terrain)

            terrain_map[(lat, lon)] = terrain_result

            # OSM
            try:
                osm_result = process_location(lat, lon)
            except Exception:
                osm_result = {
                    "distance_to_road_m": np.nan,
                    "distance_to_river_m": np.nan,
                    "nearest_road_type": None,
                    "nearest_road_name": None,
                    "nearest_road_osm_id": np.nan,
                    "nearest_waterway_type": None,
                    "nearest_waterway_name": None,
                    "nearest_waterway_osm_id": np.nan,
                    "road_search_radius_m": np.nan,
                    "river_search_radius_m": np.nan,
                    "road_extraction_ok": False,
                    "river_extraction_ok": False,
                    "roads_rivers_extraction_ok": False,
                    "osm_source": "OpenStreetMap via Overpass API",
                    "osm_snapshot_timestamp": None,
                    "distance_method": "local equirectangular point-to-polyline",
                }

            osm_map[(lat, lon)] = osm_result

            processed = idx + 1
            if processed % 25 == 0 or processed == len(unique_locations):
                print(f"  processed unique locations: {processed}/{len(unique_locations)}")

    finally:
        for dataset in open_datasets.values():
            dataset.close()

    # --------------------------------------------------------
    # Attach location-based outputs
    # --------------------------------------------------------
    terrain_rows = []
    osm_rows = []

    for _, row in df.iterrows():
        key = (float(row["latitude"]), float(row["longitude"]))
        terrain_rows.append(terrain_map.get(key, {}))
        osm_rows.append(osm_map.get(key, {}))

    df = pd.concat(
        [
            df.reset_index(drop=True),
            pd.DataFrame(terrain_rows).reset_index(drop=True),
            pd.DataFrame(osm_rows).reset_index(drop=True),
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # WorldCover class
    # --------------------------------------------------------
    print("\nSampling WorldCover...")
    df["worldcover_item_id"] = None

    for idx, row in df.iterrows():
        item = choose_worldcover_item(
            wc_items,
            float(row["longitude"]),
            float(row["latitude"]),
        )
        if item is not None:
            df.at[idx, "worldcover_item_id"] = item.id

    wc_by_id = {item.id: item for item in wc_items}

    df["land_cover_code"] = np.nan
    df["land_cover"] = None
    df["land_cover_source"] = "ESA WorldCover 2021 v200"
    df["land_cover_reference_year"] = WORLDCOVER_REFERENCE_YEAR
    df["land_cover_year_gap"] = (
        WORLDCOVER_REFERENCE_YEAR - df["_event_date"].dt.year
    ).abs()

    grouped_wc = df.dropna(subset=["worldcover_item_id"]).groupby("worldcover_item_id")
    done = 0

    for item_id, group in grouped_wc:
        item = wc_by_id[str(item_id)]
        if "map" not in item.assets:
            raise RuntimeError(
                f"WorldCover item {item_id} missing 'map' asset."
            )

        dataset = open_remote_raster(item.assets["map"].href)

        try:
            values = sample_band(
                dataset,
                group["longitude"].tolist(),
                group["latitude"].tolist(),
            )
        finally:
            dataset.close()

        for idx, value in zip(group.index, values):
            if pd.isna(value):
                continue

            code = int(round(value))
            df.at[idx, "land_cover_code"] = code
            df.at[idx, "land_cover"] = WORLDCOVER_CLASSES.get(
                code, f"Unknown class {code}"
            )

        done += len(group)
        if done % 50 == 0 or done == df["worldcover_item_id"].notna().sum():
            print(f"  worldcover sampled: {done}/{df['worldcover_item_id'].notna().sum()}")

    df["land_cover_extraction_ok"] = df["land_cover_code"].notna()

    # --------------------------------------------------------
    # MODIS NDVI
    # --------------------------------------------------------
    print("\nMatching and sampling MODIS NDVI...")
    df["modis_item_id"] = None
    df["ndvi_composite_midpoint"] = None
    df["ndvi_date_distance_days"] = np.nan

    used_modis_items = {}

    for idx, row in df.iterrows():
        item = choose_modis_item(
            modis_items,
            float(row["longitude"]),
            float(row["latitude"]),
            row["_event_date"],
        )

        if item is not None:
            midpoint = item_midpoint(item)
            event_date = row["_event_date"]
            if event_date.tzinfo is None:
                event_date = event_date.tz_localize("UTC")

            distance = abs((midpoint.normalize() - event_date.normalize()).days)

            df.at[idx, "modis_item_id"] = item.id
            df.at[idx, "ndvi_composite_midpoint"] = midpoint.isoformat()
            df.at[idx, "ndvi_date_distance_days"] = distance
            used_modis_items[item.id] = item

    df["ndvi_raw"] = np.nan
    df["ndvi"] = np.nan
    df["ndvi_pixel_reliability"] = np.nan
    df["ndvi_quality"] = None
    df["ndvi_quality_ok"] = False
    df["ndvi_source"] = "NASA MODIS MOD13Q1/MYD13Q1 v6.1 16-day 250m"

    grouped_modis = df.dropna(subset=["modis_item_id"]).groupby("modis_item_id")
    done = 0

    for item_id, group in grouped_modis:
        item = used_modis_items[str(item_id)]

        if MODIS_NDVI_ASSET not in item.assets:
            raise RuntimeError(
                f"MODIS item {item_id} missing asset {MODIS_NDVI_ASSET}"
            )

        ndvi_ds = open_remote_raster(item.assets[MODIS_NDVI_ASSET].href)
        reliability_ds = None

        try:
            raw_values = sample_band(
                ndvi_ds,
                group["longitude"].tolist(),
                group["latitude"].tolist(),
            )

            if MODIS_RELIABILITY_ASSET in item.assets:
                reliability_ds = open_remote_raster(
                    item.assets[MODIS_RELIABILITY_ASSET].href
                )
                reliability_values = sample_band(
                    reliability_ds,
                    group["longitude"].tolist(),
                    group["latitude"].tolist(),
                )
            else:
                reliability_values = [np.nan] * len(group)

        finally:
            if reliability_ds is not None:
                reliability_ds.close()
            ndvi_ds.close()

        for idx, raw_value, reliability in zip(
            group.index,
            raw_values,
            reliability_values,
        ):
            if pd.notna(raw_value):
                df.at[idx, "ndvi_raw"] = raw_value
                if -2000 <= raw_value <= 10000:
                    df.at[idx, "ndvi"] = float(raw_value) * 0.0001

            if pd.notna(reliability):
                reliability = int(round(reliability))
                df.at[idx, "ndvi_pixel_reliability"] = reliability
                df.at[idx, "ndvi_quality"] = MODIS_RELIABILITY.get(
                    reliability, f"Unknown reliability {reliability}"
                )
                df.at[idx, "ndvi_quality_ok"] = reliability in {0, 1}

        done += len(group)
        if done % 50 == 0 or done == df["modis_item_id"].notna().sum():
            print(f"  modis sampled: {done}/{df['modis_item_id'].notna().sum()}")

    df["ndvi_extraction_ok"] = (
        df["ndvi"].notna() & df["ndvi_pixel_reliability"].notna()
    )

    df["ndvi_model_value"] = (
        pd.to_numeric(df["ndvi"], errors="coerce")
        .where(df["ndvi_quality_ok"].fillna(False))
    )

    # --------------------------------------------------------
    # Rainfall + Soil for each row
    # --------------------------------------------------------
    print("\nExtracting rainfall + soil moisture for each background row...")

    rain_rows = []
    soil_rows = []

    for idx, row in df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        event_date = row["_event_date"]

        # Rainfall
        try:
            rain_payload = fetch_power_daily(
                lat,
                lon,
                event_date - pd.Timedelta(days=RAIN_LOOKBACK_DAYS),
                event_date,
            )
            rain_result = extract_rainfall_windows(rain_payload, event_date)
        except Exception:
            rain_result = None

        if rain_result is None:
            rain_result = {
                "rainfall_24h_mm": np.nan,
                "rainfall_72h_mm": np.nan,
                "rainfall_7d_mm": np.nan,
                "rainfall_daily_values_json": None,
                "rainfall_source": "NASA POWER Daily API / PRECTOTCORR",
                "rainfall_extraction_ok": False,
            }

        rain_rows.append(rain_result)

        # Soil
        try:
            soil_payload = fetch_era5_land(
                lat,
                lon,
                event_date - pd.Timedelta(days=SOIL_LOOKBACK_DAYS),
                event_date,
            )
            soil_result = extract_soil_features(soil_payload, event_date)
        except Exception:
            soil_result = None

        if soil_result is None:
            soil_result = {
                "soil_moisture_source": "ERA5-Land via Open-Meteo Historical Weather API",
                "soil_moisture_model": "era5_land",
                "soil_moisture_extraction_ok": False,
                "soil_moisture_surface_m3_m3": np.nan,
                "soil_moisture_7_28cm_m3_m3": np.nan,
                "soil_moisture_28_100cm_m3_m3": np.nan,
                "soil_moisture_0_100cm_m3_m3": np.nan,
                "soil_moisture_surface_3d_mean_m3_m3": np.nan,
                "soil_moisture_surface_7d_mean_m3_m3": np.nan,
                "soil_moisture_surface_anomaly_vs_7d": np.nan,
                "soil_moisture_surface_7d_values_json": None,
            }

        soil_rows.append(soil_result)

        processed = idx + 1
        if processed % 25 == 0 or processed == len(df):
            print(f"  processed rows: {processed}/{len(df)}")

    df = pd.concat(
        [
            df.reset_index(drop=True),
            pd.DataFrame(rain_rows).reset_index(drop=True),
            pd.DataFrame(soil_rows).reset_index(drop=True),
        ],
        axis=1,
    )

    output = df.drop(columns=["_event_date"])
    output.to_csv(OUTPUT_CSV, index=False)

    summary = {
        "rows": int(len(output)),
        "terrain_ok": int(output["terrain_extraction_ok"].fillna(False).sum()),
        "rainfall_ok": int(output["rainfall_extraction_ok"].fillna(False).sum()),
        "land_cover_ok": int(output["land_cover_extraction_ok"].fillna(False).sum()),
        "ndvi_ok": int(output["ndvi_extraction_ok"].fillna(False).sum()),
        "ndvi_model_usable": int(output["ndvi_model_value"].notna().sum()),
        "soil_ok": int(output["soil_moisture_extraction_ok"].fillna(False).sum()),
        "roads_ok": int(output["road_extraction_ok"].fillna(False).sum()),
        "rivers_ok": int(output["river_extraction_ok"].fillna(False).sum()),
        "both_osm_ok": int(output["roads_rivers_extraction_ok"].fillna(False).sum()),
        "output_csv": str(OUTPUT_CSV),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("STAGE 7B COMPLETE")
    print("=" * 80)
    print(f"Rows processed: {len(output)}")
    print(f"Terrain extracted: {summary['terrain_ok']}/{len(output)}")
    print(f"Rainfall extracted: {summary['rainfall_ok']}/{len(output)}")
    print(f"Land cover extracted: {summary['land_cover_ok']}/{len(output)}")
    print(f"Historical NDVI extracted: {summary['ndvi_ok']}/{len(output)}")
    print(f"Model-usable NDVI: {summary['ndvi_model_usable']}/{len(output)}")
    print(f"Soil moisture extracted: {summary['soil_ok']}/{len(output)}")
    print(f"Road distance extracted: {summary['roads_ok']}/{len(output)}")
    print(f"River distance extracted: {summary['rivers_ok']}/{len(output)}")
    print(f"Both road + river extracted: {summary['both_osm_ok']}/{len(output)}")
    print(f"\nOutput:\n{OUTPUT_CSV}")
    print(
        "\nIMPORTANT: Background rows are pseudo-absence samples. "
        "They now have the same real feature pipeline as positives."
    )


if __name__ == "__main__":
    main()
