from __future__ import annotations

from pathlib import Path
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import math
import os
import time
import unicodedata
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point, shape
from shapely.prepared import prep
from pystac_client import Client
import planetary_computer as pc


ROOT = Path(__file__).resolve().parents[1]

BOUNDARY_FILE = ROOT / "data" / "raw" / "boundaries" / "geoBoundaries_IND_ADM1.geojson"
MODEL_BUNDLE = ROOT / "models" / "v2" / "v2_ensemble_bundle.joblib"

OUT_DIR = ROOT / "data" / "processed" / "inference"
CACHE_DIR = ROOT / "data" / "raw" / "stage10_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

TARGET_STATES = {
    "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "sikkim": "Sikkim",
    "tripura": "Tripura",
}

STATE_KEYS = ["shapeName", "state", "STATE", "name", "NAME_1", "NAME"]

LAND_COVER_MAP = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

ROAD_REGEX = (
    "motorway|motorway_link|trunk|trunk_link|primary|primary_link|"
    "secondary|secondary_link|tertiary|tertiary_link|unclassified|"
    "residential|service|living_street|road|track"
)
WATERWAY_REGEX = "river|stream|canal|drain"

OVERPASS_ENDPOINTS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.maprva.org/api/interpreter",
    "https://ethiopia.overpass.openplaceguide.org/api/interpreter",
]

HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "SIH-Landslide-Research/1.0"})


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def normalize_name(value: Any) -> str:
    if value is None:
        return ""

    # geoBoundaries may use diacritics/transliterated spellings such as
    # "Arunāchal Pradesh", "Meghālaya", "Nāgāland".
    # Strip Unicode combining marks so they match our canonical ASCII names.
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.split())


def get_state_name(properties: dict) -> str | None:
    for key in STATE_KEYS:
        if properties.get(key):
            return str(properties[key]).strip()
    return None


def frange(start: float, stop: float, step: float):
    x = start
    while x <= stop + 1e-12:
        yield round(x, 8)
        x += step


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def safe_float(value):
    try:
        value = float(value)
        if not np.isfinite(value):
            return np.nan
        return value
    except Exception:
        return np.nan


def save_work(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


def stac_asset_key(item, preferred: list[str]) -> str:
    for key in preferred:
        if key in item.assets:
            return key
    for key, asset in item.assets.items():
        href = str(asset.href).lower()
        media = str(asset.media_type or "").lower()
        if ".tif" in href or "geotiff" in media or "cloud-optimized" in media:
            return key
    if item.assets:
        return next(iter(item.assets.keys()))
    raise RuntimeError(f"No assets found for STAC item {item.id}")


def item_contains(item, lon: float, lat: float) -> bool:
    if item.bbox and len(item.bbox) >= 4:
        minx, miny, maxx, maxy = item.bbox[:4]
        return minx <= lon <= maxx and miny <= lat <= maxy
    return shape(item.geometry).covers(Point(lon, lat))


def sign_href(item, asset_key: str) -> str:
    signed = pc.sign(item)
    return signed.assets[asset_key].href


def download_file(url: str, path: Path, retries: int = 4) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path

    part = path.with_suffix(path.suffix + ".part")
    last = None

    for attempt in range(1, retries + 1):
        try:
            with HTTP.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with part.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            part.replace(path)
            return path
        except Exception as exc:
            last = exc
            if part.exists():
                try:
                    part.unlink()
                except Exception:
                    pass
            time.sleep(min(8, attempt * 1.5))

    raise RuntimeError(f"Download failed: {url}\n{last}")


# ---------------------------------------------------------------------------
# 1. NER boundaries + grid
# ---------------------------------------------------------------------------

def _select_ner_boundaries(data: dict) -> dict[str, Any]:
    selected = {}

    for feature in data.get("features", []):
        raw = get_state_name(feature.get("properties") or {})
        norm = normalize_name(raw)
        if norm in TARGET_STATES:
            canonical = TARGET_STATES[norm]
            geom = shape(feature["geometry"])
            selected[canonical] = (
                selected[canonical].union(geom)
                if canonical in selected
                else geom
            )

    return selected


def _download_full_india_adm1() -> dict:
    """
    Download the complete India ADM1 layer from the official geoBoundaries API.
    We keep this separate from the historical 5-state boundary file used earlier.
    """
    api_url = "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM1/"
    cache_file = CACHE_DIR / "boundaries" / "geoBoundaries_IND_ADM1_full.geojson"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and cache_file.stat().st_size > 0:
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            selected = _select_ner_boundaries(cached)
            if len(selected) == 8:
                print(f"Using cached full India ADM1 boundary:\n{cache_file}")
                return cached
        except Exception:
            pass

    print("Local boundary file does not contain all 8 NER states.")
    print("Downloading complete India ADM1 boundary from geoBoundaries...")

    meta = HTTP.get(api_url, timeout=60)
    meta.raise_for_status()
    meta_json = meta.json()

    gj_url = meta_json.get("gjDownloadURL")
    if not gj_url:
        raise RuntimeError("geoBoundaries API response has no gjDownloadURL.")

    r = HTTP.get(gj_url, timeout=120)
    r.raise_for_status()
    data = r.json()

    if data.get("type") != "FeatureCollection":
        raise RuntimeError("Downloaded geoBoundaries file is not a FeatureCollection.")

    cache_file.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved complete India ADM1 boundary:\n{cache_file}")
    return data


def load_ner_boundaries() -> dict[str, Any]:
    # First try the project's existing boundary file.
    data = None
    if BOUNDARY_FILE.exists():
        try:
            data = json.loads(BOUNDARY_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = None

    if data is not None:
        selected = _select_ner_boundaries(data)
        missing = [s for s in TARGET_STATES.values() if s not in selected]

        if not missing:
            print("Existing boundary file contains all 8 NER states.")
            return selected

        print(f"Existing boundary file is missing: {missing}")

    # Stage 10 needs all 8 NER states, so automatically fetch a complete
    # India ADM1 layer instead of failing or silently omitting states.
    full_data = _download_full_india_adm1()
    selected = _select_ner_boundaries(full_data)

    missing = [s for s in TARGET_STATES.values() if s not in selected]
    if missing:
        available = []
        for feature in full_data.get("features", []):
            name = get_state_name(feature.get("properties") or {})
            if name:
                available.append(name)

        raise RuntimeError(
            f"Even the complete India ADM1 layer is missing NER states: {missing}. "
            f"Available ADM1 names: {sorted(set(available))}"
        )

    print("All 8 NER state boundaries loaded successfully.")
    return selected


def generate_grid(boundaries: dict[str, Any], step: float) -> pd.DataFrame:
    rows = []

    for state_name in TARGET_STATES.values():
        geom = boundaries[state_name]
        prepared = prep(geom)
        minx, miny, maxx, maxy = geom.bounds

        lon0 = math.floor(minx / step) * step
        lon1 = math.ceil(maxx / step) * step
        lat0 = math.floor(miny / step) * step
        lat1 = math.ceil(maxy / step) * step

        count = 0
        for lat in frange(lat0, lat1, step):
            for lon in frange(lon0, lon1, step):
                if prepared.covers(Point(lon, lat)):
                    rows.append(
                        {
                            "state": state_name,
                            "latitude": round(lat, 6),
                            "longitude": round(lon, 6),
                        }
                    )
                    count += 1

        print(f"  {state_name:20s}: {count:5d} points")

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(["state", "latitude", "longitude"])
        .sort_values(["state", "latitude", "longitude"])
        .reset_index(drop=True)
    )
    df.insert(0, "grid_id", [f"NER_{i:06d}" for i in range(1, len(df) + 1)])
    return df


# ---------------------------------------------------------------------------
# 2. Copernicus DEM terrain
# ---------------------------------------------------------------------------

def sample_dem_window(path: Path, lon: float, lat: float):
    with rasterio.open(path) as ds:
        try:
            row, col = ds.index(lon, lat)
        except Exception:
            return [np.nan] * 4

        if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
            return [np.nan] * 4

        half = 4
        r0 = max(0, row - half)
        c0 = max(0, col - half)
        h = min(ds.height - r0, 2 * half + 1)
        w = min(ds.width - c0, 2 * half + 1)

        arr = ds.read(
            1,
            window=Window(c0, r0, w, h),
            masked=True,
        ).astype(float)

        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)

        if arr.size == 0 or np.isnan(arr).all():
            return [np.nan] * 4

        center_r = row - r0
        center_c = col - c0
        elev = arr[center_r, center_c]
        if not np.isfinite(elev):
            elev = float(np.nanmedian(arr))

        center_lat = lat
        dx = abs(ds.transform.a) * 111320.0 * math.cos(math.radians(center_lat))
        dy = abs(ds.transform.e) * 110574.0

        if dx <= 0 or dy <= 0:
            return elev, np.nan, np.nan, np.nan

        # fill local holes for stable finite differences
        fill = np.nanmedian(arr)
        z = np.where(np.isfinite(arr), arr, fill)

        gy, gx = np.gradient(z, dy, dx)
        slope = np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))

        # Downslope aspect clockwise from north
        aspect = (np.degrees(np.arctan2(-gx, gy)) + 360.0) % 360.0

        dgy_dy, _ = np.gradient(gy, dy, dx)
        _, dgx_dx = np.gradient(gx, dy, dx)
        curvature = dgx_dx + dgy_dy

        return (
            float(elev),
            float(slope[center_r, center_c]),
            float(aspect[center_r, center_c]),
            float(curvature[center_r, center_c]),
        )


def extract_dem(df: pd.DataFrame, stac: Client) -> pd.DataFrame:
    print("\n[2/8] Copernicus DEM / terrain")
    bbox = [
        float(df.longitude.min()) - 0.2,
        float(df.latitude.min()) - 0.2,
        float(df.longitude.max()) + 0.2,
        float(df.latitude.max()) + 0.2,
    ]

    items = list(
        stac.search(
            collections=["cop-dem-glo-30"],
            bbox=bbox,
        ).items()
    )
    if not items:
        raise RuntimeError("No Copernicus DEM STAC items found.")

    print(f"  DEM items found: {len(items)}")

    cache = CACHE_DIR / "dem"
    item_local = {}

    for item in items:
        key = stac_asset_key(item, ["data"])
        href = sign_href(item, key)
        local = cache / f"{item.id}.tif"
        item_local[item.id] = download_file(href, local)

    out = df.copy()
    vals = []

    for i, row in out.iterrows():
        lon = float(row.longitude)
        lat = float(row.latitude)
        item = next((it for it in items if item_contains(it, lon, lat)), None)

        if item is None:
            vals.append((np.nan, np.nan, np.nan, np.nan))
        else:
            vals.append(sample_dem_window(item_local[item.id], lon, lat))

        if (i + 1) % 100 == 0 or i + 1 == len(out):
            print(f"  terrain {i+1}/{len(out)}")

    arr = np.asarray(vals, dtype=float)
    out["elevation_m"] = arr[:, 0]
    out["slope_deg"] = arr[:, 1]
    out["aspect_deg"] = arr[:, 2]
    out["curvature_1_per_m"] = arr[:, 3]
    return out


# ---------------------------------------------------------------------------
# 3. ESA WorldCover + MODIS NDVI
# ---------------------------------------------------------------------------

def sample_raster_value(path: Path, lon: float, lat: float):
    with rasterio.open(path) as ds:
        value = next(ds.sample([(lon, lat)]))[0]
        if ds.nodata is not None and value == ds.nodata:
            return np.nan
        return safe_float(value)


def extract_worldcover(df: pd.DataFrame, stac: Client) -> pd.DataFrame:
    print("\n[3/8] ESA WorldCover")
    bbox = [
        float(df.longitude.min()) - 0.2,
        float(df.latitude.min()) - 0.2,
        float(df.longitude.max()) + 0.2,
        float(df.latitude.max()) + 0.2,
    ]

    items = list(
        stac.search(
            collections=["esa-worldcover"],
            bbox=bbox,
        ).items()
    )
    if not items:
        raise RuntimeError("No ESA WorldCover items found.")

    print(f"  WorldCover items found: {len(items)}")

    cache = CACHE_DIR / "worldcover"
    local_map = {}

    for item in items:
        key = stac_asset_key(item, ["map", "data"])
        href = sign_href(item, key)
        local_map[item.id] = download_file(href, cache / f"{item.id}.tif")

    out = df.copy()
    codes = []

    for i, row in out.iterrows():
        lon = float(row.longitude)
        lat = float(row.latitude)
        item = next((it for it in items if item_contains(it, lon, lat)), None)
        code = np.nan if item is None else sample_raster_value(local_map[item.id], lon, lat)
        codes.append(code)

    out["land_cover_code"] = pd.to_numeric(pd.Series(codes), errors="coerce")
    out["land_cover"] = out["land_cover_code"].round().map(LAND_COVER_MAP).astype("string")
    return out


def item_datetime(item) -> datetime:
    if item.datetime:
        dt = item.datetime
    else:
        raw = item.properties.get("datetime") or item.properties.get("start_datetime")
        dt = pd.Timestamp(raw).to_pydatetime()
    if dt.tzinfo:
        dt = dt.replace(tzinfo=None)
    return dt


def extract_ndvi(df: pd.DataFrame, stac: Client, inference_date: date) -> pd.DataFrame:
    print("\n[4/8] MODIS historical/current-proxy NDVI")

    bbox = [
        float(df.longitude.min()) - 0.2,
        float(df.latitude.min()) - 0.2,
        float(df.longitude.max()) + 0.2,
        float(df.latitude.max()) + 0.2,
    ]

    start = inference_date - timedelta(days=50)
    end = inference_date + timedelta(days=5)

    items = list(
        stac.search(
            collections=["modis-13Q1-061"],
            bbox=bbox,
            datetime=f"{start.isoformat()}/{end.isoformat()}",
        ).items()
    )

    if not items:
        start = inference_date - timedelta(days=100)
        items = list(
            stac.search(
                collections=["modis-13Q1-061"],
                bbox=bbox,
                datetime=f"{start.isoformat()}/{end.isoformat()}",
            ).items()
        )

    print(f"  MODIS items found: {len(items)}")
    if not items:
        out = df.copy()
        out["ndvi_model_value"] = np.nan
        out["ndvi_missing"] = 1
        return out

    target_dt = datetime.combine(inference_date, datetime.min.time())
    cache = CACHE_DIR / "modis_ndvi"

    # Determine the single best item per point first.
    chosen = []
    for _, row in df.iterrows():
        lon, lat = float(row.longitude), float(row.latitude)
        candidates = [it for it in items if item_contains(it, lon, lat)]
        if not candidates:
            chosen.append(None)
        else:
            chosen.append(
                min(
                    candidates,
                    key=lambda it: abs((item_datetime(it) - target_dt).total_seconds()),
                )
            )

    unique_items = {it.id: it for it in chosen if it is not None}
    print(f"  MODIS items actually used: {len(unique_items)}")

    local_assets = {}
    for item_id, item in unique_items.items():
        ndvi_key = (
            "250m_16_days_NDVI"
            if "250m_16_days_NDVI" in item.assets
            else stac_asset_key(item, ["250m_16_days_NDVI"])
        )
        rel_key = (
            "250m_16_days_pixel_reliability"
            if "250m_16_days_pixel_reliability" in item.assets
            else None
        )

        ndvi_path = download_file(
            sign_href(item, ndvi_key),
            cache / f"{item.id}_ndvi.tif",
        )

        rel_path = None
        if rel_key:
            rel_path = download_file(
                sign_href(item, rel_key),
                cache / f"{item.id}_reliability.tif",
            )

        local_assets[item_id] = (ndvi_path, rel_path)

    ndvi_values = []
    ndvi_reliability = []

    for idx, row in df.iterrows():
        item = chosen[idx]
        if item is None:
            ndvi_values.append(np.nan)
            ndvi_reliability.append(np.nan)
            continue

        lon, lat = float(row.longitude), float(row.latitude)
        ndvi_path, rel_path = local_assets[item.id]
        raw = sample_raster_value(ndvi_path, lon, lat)
        rel = sample_raster_value(rel_path, lon, lat) if rel_path else np.nan

        if np.isfinite(raw):
            ndvi = raw * 0.0001
            if ndvi < -0.2 or ndvi > 1.0:
                ndvi = np.nan
        else:
            ndvi = np.nan

        # Reliability: 0=good, 1=marginal, 2=snow/ice, 3=cloudy.
        if np.isfinite(rel) and int(round(rel)) not in (0, 1):
            model_value = np.nan
        else:
            model_value = ndvi

        ndvi_values.append(model_value)
        ndvi_reliability.append(rel)

    out = df.copy()
    out["ndvi_model_value"] = ndvi_values
    out["ndvi_pixel_reliability"] = ndvi_reliability
    out["ndvi_missing"] = out["ndvi_model_value"].isna().astype(int)
    return out


# ---------------------------------------------------------------------------
# 5. OSM road / river distances
# ---------------------------------------------------------------------------

def local_xy(lat, lon, lat0, lon0):
    return (
        (lon - lon0) * 111320.0 * math.cos(math.radians(lat0)),
        (lat - lat0) * 110574.0,
    )


def point_segment_distance(ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    den = dx * dx + dy * dy
    if den <= 0:
        return math.hypot(ax, ay)
    t = -(ax * dx + ay * dy) / den
    t = max(0.0, min(1.0, t))
    return math.hypot(ax + t * dx, ay + t * dy)


def geometry_distance(lat0, lon0, geometry):
    pts = []
    for p in geometry or []:
        try:
            pts.append(local_xy(float(p["lat"]), float(p["lon"]), lat0, lon0))
        except Exception:
            pass

    if not pts:
        return np.nan
    if len(pts) == 1:
        return math.hypot(*pts[0])

    return min(
        point_segment_distance(*pts[i], *pts[i + 1])
        for i in range(len(pts) - 1)
    )


def overpass_request(query: str, cache_file: Path):
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    last = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            r = HTTP.post(
                endpoint,
                data={"data": query},
                timeout=120,
            )
            if r.status_code == 200:
                payload = r.json()
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(payload), encoding="utf-8")
                time.sleep(0.5)
                return payload

            last = RuntimeError(f"{endpoint}: HTTP {r.status_code}")
            if r.status_code in (429, 502, 503, 504):
                time.sleep(2)
        except Exception as exc:
            last = exc

    raise RuntimeError(f"All Overpass endpoints failed: {last}")


def tile_overpass_query(south, west, north, east):
    return f"""
[out:json][timeout:80];
(
  way({south:.6f},{west:.6f},{north:.6f},{east:.6f})["highway"~"^({ROAD_REGEX})$"];
  way({south:.6f},{west:.6f},{north:.6f},{east:.6f})["waterway"~"^({WATERWAY_REGEX})$"];
);
out tags geom;
"""


def nearest_osm(elements, lat, lon, kind):
    best = float("inf")
    for el in elements:
        if el.get("type") != "way":
            continue
        tags = el.get("tags") or {}
        if kind == "road" and not tags.get("highway"):
            continue
        if kind == "river" and not tags.get("waterway"):
            continue
        d = geometry_distance(lat, lon, el.get("geometry"))
        if np.isfinite(d) and d < best:
            best = d
    return best if np.isfinite(best) else np.nan


def point_overpass_fallback(lat, lon):
    for radius in (5000, 15000, 30000):
        query = f"""
[out:json][timeout:45];
(
  way(around:{radius},{lat:.7f},{lon:.7f})["highway"~"^({ROAD_REGEX})$"];
  way(around:{radius},{lat:.7f},{lon:.7f})["waterway"~"^({WATERWAY_REGEX})$"];
);
out tags geom;
"""
        key = sha1_text(f"point|{lat:.6f}|{lon:.6f}|{radius}")
        try:
            payload = overpass_request(
                query,
                CACHE_DIR / "osm_point" / f"{key}.json",
            )
        except Exception:
            continue

        elements = payload.get("elements", [])
        road = nearest_osm(elements, lat, lon, "road")
        river = nearest_osm(elements, lat, lon, "river")

        if np.isfinite(road) and np.isfinite(river):
            return road, river

    return np.nan, np.nan


def extract_osm(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[5/8] OpenStreetMap road / river proximity")
    out = df.copy()
    out["distance_to_road_m"] = np.nan
    out["distance_to_river_m"] = np.nan

    tile_size = 1.0
    margin = 0.35  # ~35 km at NER latitudes

    out["_tile_lat"] = np.floor(out["latitude"] / tile_size).astype(int)
    out["_tile_lon"] = np.floor(out["longitude"] / tile_size).astype(int)

    groups = list(out.groupby(["_tile_lat", "_tile_lon"]))
    print(f"  OSM tiles: {len(groups)}")

    for tile_no, ((ilat, ilon), idxs) in enumerate(groups, start=1):
        south = ilat * tile_size - margin
        north = (ilat + 1) * tile_size + margin
        west = ilon * tile_size - margin
        east = (ilon + 1) * tile_size + margin

        query = tile_overpass_query(south, west, north, east)
        key = sha1_text(f"{south}|{west}|{north}|{east}")

        try:
            payload = overpass_request(
                query,
                CACHE_DIR / "osm_tiles" / f"{key}.json",
            )
            elements = payload.get("elements", [])
        except Exception as exc:
            print(f"  tile {tile_no}/{len(groups)} failed: {str(exc)[:140]}")
            elements = []

        for idx in idxs.index:
            lat = float(out.at[idx, "latitude"])
            lon = float(out.at[idx, "longitude"])
            road = nearest_osm(elements, lat, lon, "road") if elements else np.nan
            river = nearest_osm(elements, lat, lon, "river") if elements else np.nan
            out.at[idx, "distance_to_road_m"] = road
            out.at[idx, "distance_to_river_m"] = river

        print(
            f"  tile {tile_no}/{len(groups)} "
            f"rows={len(idxs)} ways={len(elements)}"
        )

    missing = out[
        out["distance_to_road_m"].isna()
        | out["distance_to_river_m"].isna()
    ].index.tolist()

    if missing:
        print(f"  point fallback for {len(missing)} rows")
        for n, idx in enumerate(missing, start=1):
            lat = float(out.at[idx, "latitude"])
            lon = float(out.at[idx, "longitude"])
            road, river = point_overpass_fallback(lat, lon)

            if pd.isna(out.at[idx, "distance_to_road_m"]) and np.isfinite(road):
                out.at[idx, "distance_to_road_m"] = road
            if pd.isna(out.at[idx, "distance_to_river_m"]) and np.isfinite(river):
                out.at[idx, "distance_to_river_m"] = river

            if n % 25 == 0 or n == len(missing):
                print(f"  OSM fallback {n}/{len(missing)}")

    out = out.drop(columns=["_tile_lat", "_tile_lon"])
    return out


# ---------------------------------------------------------------------------
# 6. NASA POWER rainfall
# ---------------------------------------------------------------------------

def power_one(lat: float, lon: float, inference_date: date):
    start = inference_date - timedelta(days=6)
    key = sha1_text(f"{lat:.6f}|{lon:.6f}|{inference_date.isoformat()}")
    path = CACHE_DIR / "rainfall" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = None
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = None

    if payload is None:
        params = {
            "parameters": "PRECTOTCORR",
            "community": "AG",
            "longitude": f"{lon:.6f}",
            "latitude": f"{lat:.6f}",
            "start": start.strftime("%Y%m%d"),
            "end": inference_date.strftime("%Y%m%d"),
            "format": "JSON",
        }

        last = None
        for attempt in range(1, 5):
            try:
                r = requests.get(
                    POWER_URL,
                    params=params,
                    timeout=45,
                    headers={"User-Agent": "SIH-Landslide-Research/1.0"},
                )
                r.raise_for_status()
                payload = r.json()
                path.write_text(json.dumps(payload), encoding="utf-8")
                break
            except Exception as exc:
                last = exc
                time.sleep(min(6, 1.5 * attempt))

        if payload is None:
            return (np.nan, np.nan, np.nan, str(last))

    raw = (
        payload.get("properties", {})
        .get("parameter", {})
        .get("PRECTOTCORR", {})
    )

    vals = []
    for i in range(7):
        d = start + timedelta(days=i)
        v = safe_float(raw.get(d.strftime("%Y%m%d")))
        if not np.isfinite(v) or v <= -900:
            v = np.nan
        vals.append(v)

    arr = np.asarray(vals, dtype=float)
    if np.isnan(arr).any():
        return (np.nan, np.nan, np.nan, "incomplete rainfall window")

    return (
        float(arr[-1]),
        float(arr[-3:].sum()),
        float(arr.sum()),
        None,
    )


def extract_rainfall(df: pd.DataFrame, inference_date: date, workers: int) -> pd.DataFrame:
    print("\n[6/8] NASA POWER rainfall")
    out = df.copy()

    results = [None] * len(out)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                power_one,
                float(row.latitude),
                float(row.longitude),
                inference_date,
            ): i
            for i, row in out.iterrows()
        }

        completed = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:
                results[i] = (np.nan, np.nan, np.nan, str(exc))

            completed += 1
            if completed % 50 == 0 or completed == len(out):
                print(f"  rainfall {completed}/{len(out)}")

    out["rainfall_24h_mm"] = [r[0] for r in results]
    out["rainfall_72h_mm"] = [r[1] for r in results]
    out["rainfall_7d_mm"] = [r[2] for r in results]
    return out


# ---------------------------------------------------------------------------
# 7. ERA5-Land soil moisture through Open-Meteo archive
# ---------------------------------------------------------------------------

SOIL_DAILY_VARS = [
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
    "soil_moisture_28_to_100cm_mean",
    "soil_moisture_0_to_100cm_mean",
]


def soil_one(lat: float, lon: float, inference_date: date):
    start = inference_date - timedelta(days=6)
    key = sha1_text(f"{lat:.6f}|{lon:.6f}|{inference_date.isoformat()}")
    path = CACHE_DIR / "soil" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = None
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = None

    if payload is None:
        params = {
            "latitude": f"{lat:.6f}",
            "longitude": f"{lon:.6f}",
            "start_date": start.isoformat(),
            "end_date": inference_date.isoformat(),
            "daily": ",".join(SOIL_DAILY_VARS),
            "models": "era5_land",
            "timezone": "UTC",
        }

        last = None
        for attempt in range(1, 5):
            try:
                r = requests.get(
                    OPEN_METEO_ARCHIVE,
                    params=params,
                    timeout=45,
                    headers={"User-Agent": "SIH-Landslide-Research/1.0"},
                )
                r.raise_for_status()
                payload = r.json()
                path.write_text(json.dumps(payload), encoding="utf-8")
                break
            except Exception as exc:
                last = exc
                time.sleep(min(6, 1.5 * attempt))

        if payload is None:
            return [np.nan] * 7 + [str(last)]

    daily = payload.get("daily") or {}

    arrays = {}
    for var in SOIL_DAILY_VARS:
        vals = [safe_float(v) for v in (daily.get(var) or [])]
        arrays[var] = np.asarray(vals, dtype=float)

    surface = arrays["soil_moisture_0_to_7cm_mean"]
    if len(surface) < 7 or np.isnan(surface[-7:]).any():
        return [np.nan] * 7 + ["incomplete soil-moisture window"]

    day_surface = float(surface[-1])
    day_7_28 = float(arrays["soil_moisture_7_to_28cm_mean"][-1])
    day_28_100 = float(arrays["soil_moisture_28_to_100cm_mean"][-1])
    day_0_100 = float(arrays["soil_moisture_0_to_100cm_mean"][-1])
    mean3 = float(np.mean(surface[-3:]))
    mean7 = float(np.mean(surface[-7:]))
    anomaly = float(day_surface - mean7)

    return [
        day_surface,
        day_7_28,
        day_28_100,
        day_0_100,
        mean3,
        mean7,
        anomaly,
        None,
    ]


def extract_soil(df: pd.DataFrame, inference_date: date, workers: int) -> pd.DataFrame:
    print("\n[7/8] ERA5-Land soil moisture")
    out = df.copy()
    results = [None] * len(out)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                soil_one,
                float(row.latitude),
                float(row.longitude),
                inference_date,
            ): i
            for i, row in out.iterrows()
        }

        completed = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:
                results[i] = [np.nan] * 7 + [str(exc)]

            completed += 1
            if completed % 50 == 0 or completed == len(out):
                print(f"  soil {completed}/{len(out)}")

    cols = [
        "soil_moisture_surface_m3_m3",
        "soil_moisture_7_28cm_m3_m3",
        "soil_moisture_28_100cm_m3_m3",
        "soil_moisture_0_100cm_m3_m3",
        "soil_moisture_surface_3d_mean_m3_m3",
        "soil_moisture_surface_7d_mean_m3_m3",
        "soil_moisture_surface_anomaly_vs_7d",
    ]

    for j, col in enumerate(cols):
        out[col] = [r[j] for r in results]

    return out


# ---------------------------------------------------------------------------
# 8. Model inference + exports
# ---------------------------------------------------------------------------

def run_model(df: pd.DataFrame, inference_date: date, grid_step: float) -> pd.DataFrame:
    print("\n[8/8] Ensemble inference + exports")

    if not MODEL_BUNDLE.exists():
        raise FileNotFoundError(MODEL_BUNDLE)

    bundle = joblib.load(MODEL_BUNDLE)

    out = df.copy()
    radians = np.deg2rad(pd.to_numeric(out["aspect_deg"], errors="coerce"))
    out["aspect_sin"] = np.sin(radians)
    out["aspect_cos"] = np.cos(radians)
    out["ndvi_missing"] = out["ndvi_model_value"].isna().astype(int)

    features = bundle["features"]
    missing = [f for f in features if f not in out.columns]
    if missing:
        raise RuntimeError(f"Inference features missing: {missing}")

    X = out[features].copy()

    model_probs = {}
    for name, model in bundle["models"].items():
        print(f"  predicting {name}")
        model_probs[name] = model.predict_proba(X)[:, 1]
        out[f"prob_{name}"] = model_probs[name]

    weights = bundle.get("ensemble_weights") or {
        k: 1 / len(model_probs) for k in model_probs
    }

    total_weight = sum(float(weights.get(k, 0)) for k in model_probs)
    if total_weight <= 0:
        raise RuntimeError("Invalid ensemble weights")

    ensemble = np.zeros(len(out), dtype=float)
    for name, probs in model_probs.items():
        ensemble += probs * float(weights.get(name, 0)) / total_weight

    out["risk_probability"] = ensemble

    # Dashboard bands only; not field-calibrated warning thresholds.
    out["risk_level"] = pd.cut(
        out["risk_probability"],
        bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=["LOW", "MODERATE", "HIGH", "CRITICAL"],
        right=False,
    ).astype(str)

    alert_threshold = float(
        bundle.get("prototype_oof_f1_threshold", 0.30)
    )
    out["prototype_alert_threshold"] = alert_threshold
    out["prototype_alert"] = out["risk_probability"] >= alert_threshold
    out["inference_date"] = inference_date.isoformat()
    out["grid_resolution_deg"] = grid_step

    return out


def export_outputs(df: pd.DataFrame, inference_date: date, grid_step: float):
    date_tag = inference_date.isoformat()
    step_tag = str(grid_step).replace(".", "p")

    csv_path = OUT_DIR / f"ner_risk_predictions_{date_tag}_{step_tag}deg.csv"
    geojson_path = OUT_DIR / f"ner_risk_predictions_{date_tag}_{step_tag}deg.geojson"
    summary_path = OUT_DIR / f"ner_risk_summary_{date_tag}_{step_tag}deg.json"

    df.to_csv(csv_path, index=False)

    features = []
    for row in df.itertuples(index=False):
        props = {
            "grid_id": row.grid_id,
            "state": row.state,
            "risk_probability": float(row.risk_probability),
            "risk_level": row.risk_level,
            "prototype_alert": bool(row.prototype_alert),
            "inference_date": row.inference_date,
        }

        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        float(row.longitude),
                        float(row.latitude),
                    ],
                },
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}
    geojson_path.write_text(
        json.dumps(geojson, ensure_ascii=False),
        encoding="utf-8",
    )

    counts = (
        df["risk_level"]
        .value_counts()
        .reindex(["LOW", "MODERATE", "HIGH", "CRITICAL"], fill_value=0)
        .to_dict()
    )

    state_summary = (
        df.groupby("state")
        .agg(
            grid_points=("grid_id", "count"),
            mean_risk=("risk_probability", "mean"),
            max_risk=("risk_probability", "max"),
            alerts=("prototype_alert", "sum"),
        )
        .round(4)
        .to_dict(orient="index")
    )

    summary = {
        "inference_date": inference_date.isoformat(),
        "grid_resolution_deg": grid_step,
        "grid_points": int(len(df)),
        "risk_level_counts": {k: int(v) for k, v in counts.items()},
        "prototype_alert_threshold": float(df["prototype_alert_threshold"].iloc[0]),
        "prototype_alert_count": int(df["prototype_alert"].sum()),
        "state_summary": state_summary,
        "model_validation_reference": (
            "Stage 9 geographic CV: ensemble ROC-AUC ~0.806; "
            "OOF-selected prototype threshold ~0.30 had recall ~0.827. "
            "These are prototype validation metrics, not operational field accuracy."
        ),
        "important_caveats": [
            "Background training samples are matched pseudo-absences, not confirmed stable locations.",
            "Risk bands are dashboard visualization bands, not government warning thresholds.",
            "The alert threshold was selected from out-of-fold prototype predictions and is not independently field validated.",
            "WorldCover 2021 and current OSM are static susceptibility proxies.",
            "This pipeline uses a conservative data-complete inference date rather than claiming live real-time nowcasting.",
            "NASA POWER rainfall and ERA5-Land soil moisture are gridded products and may not capture local convective or slope-scale conditions.",
        ],
        "outputs": {
            "csv": str(csv_path),
            "geojson": str(geojson_path),
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return csv_path, geojson_path, summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete Stage 10 NER-wide landslide inference pipeline."
    )
    parser.add_argument(
        "--date",
        default=None,
        help=(
            "Inference date YYYY-MM-DD. Default: today minus 7 days "
            "to improve availability across NASA POWER, MODIS and ERA5-Land."
        ),
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.20,
        help="Prediction grid spacing in degrees. Default: 0.20 (~22 km N-S).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent API workers for rainfall/soil. Default: 6.",
    )
    args = parser.parse_args()

    if args.date:
        inference_date = pd.Timestamp(args.date).date()
    else:
        inference_date = date.today() - timedelta(days=7)

    if args.grid_step <= 0 or args.grid_step > 1:
        raise ValueError("--grid-step must be >0 and <=1 degree")

    print("=" * 96)
    print("STAGE 10 — COMPLETE NER-WIDE LANDSLIDE RISK INFERENCE")
    print("=" * 96)
    print(f"Inference date: {inference_date}")
    print(f"Grid step: {args.grid_step} degree")
    print(f"Workers: {args.workers}")
    print(
        "\nThis is a prototype risk-map pipeline. "
        "It does not claim field-validated real-time warning accuracy."
    )

    step_tag = str(args.grid_step).replace(".", "p")
    working = OUT_DIR / f"stage10_working_{inference_date.isoformat()}_{step_tag}.csv"

    boundaries = load_ner_boundaries()

    # If a working file exists from an interrupted run, we reuse it only when
    # it already contains the columns required by the next stage.
    if working.exists():
        df = pd.read_csv(working)
        print(f"\nResuming from working file with {len(df)} rows:\n{working}")
    else:
        print("\n[1/8] Generate NER grid")
        df = generate_grid(boundaries, args.grid_step)
        df["inference_date"] = inference_date.isoformat()
        save_work(df, working)
        print(f"  total grid points: {len(df)}")

    stac = Client.open(STAC_URL)

    if not {"elevation_m", "slope_deg", "aspect_deg", "curvature_1_per_m"}.issubset(df.columns):
        df = extract_dem(df, stac)
        save_work(df, working)

    if "land_cover" not in df.columns:
        df = extract_worldcover(df, stac)
        save_work(df, working)

    if "ndvi_model_value" not in df.columns:
        df = extract_ndvi(df, stac, inference_date)
        save_work(df, working)

    if not {"distance_to_road_m", "distance_to_river_m"}.issubset(df.columns):
        df = extract_osm(df)
        save_work(df, working)

    if not {"rainfall_24h_mm", "rainfall_72h_mm", "rainfall_7d_mm"}.issubset(df.columns):
        df = extract_rainfall(df, inference_date, args.workers)
        save_work(df, working)

    soil_cols = {
        "soil_moisture_surface_m3_m3",
        "soil_moisture_7_28cm_m3_m3",
        "soil_moisture_28_100cm_m3_m3",
        "soil_moisture_0_100cm_m3_m3",
        "soil_moisture_surface_3d_mean_m3_m3",
        "soil_moisture_surface_7d_mean_m3_m3",
        "soil_moisture_surface_anomaly_vs_7d",
    }

    if not soil_cols.issubset(df.columns):
        df = extract_soil(df, inference_date, args.workers)
        save_work(df, working)

    pred = run_model(df, inference_date, args.grid_step)
    csv_path, geojson_path, summary_path = export_outputs(
        pred,
        inference_date,
        args.grid_step,
    )

    print("\n" + "=" * 96)
    print("STAGE 10 COMPLETE")
    print("=" * 96)
    print(f"Grid points: {len(pred)}")
    print(f"Risk predictions CSV:\n{csv_path}")
    print(f"\nDashboard GeoJSON:\n{geojson_path}")
    print(f"\nRisk summary:\n{summary_path}")
    print(f"\nResume/cache working file:\n{working}")

    print("\nRisk-level counts:")
    print(
        pred["risk_level"]
        .value_counts()
        .reindex(["LOW", "MODERATE", "HIGH", "CRITICAL"], fill_value=0)
        .to_string()
    )
    print(
        f"\nPrototype alerts (>= {pred['prototype_alert_threshold'].iloc[0]:.3f}): "
        f"{int(pred['prototype_alert'].sum())}/{len(pred)}"
    )


if __name__ == "__main__":
    main()
