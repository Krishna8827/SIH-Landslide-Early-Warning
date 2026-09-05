from pathlib import Path
from datetime import timezone
import json
import math
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
from pystac_client import Client
import planetary_computer

# Stable GDAL options for reading remote Cloud-Optimized GeoTIFFs.
# We intentionally avoid manually entering/exiting nested rasterio.Env
# objects because nested Env teardown can produce:
# rasterio.errors.EnvError: No GDAL environment exists
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_CANDIDATES = [
    ROOT / "data" / "processed" / "rainfall" / "ner_events_with_dem_rainfall.csv",
    ROOT / "data" / "processed" / "terrain" / "ner_events_with_copernicus_dem.csv",
]

OUT_DIR = ROOT / "data" / "processed" / "landcover_ndvi"
META_DIR = ROOT / "data" / "raw" / "satellite_metadata"

OUT_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUT_DIR / "ner_events_with_dem_rainfall_landcover_ndvi.csv"
SUMMARY_JSON = OUT_DIR / "landcover_ndvi_summary.json"
WORLDCOVER_META = META_DIR / "esa_worldcover_2021_items.json"
MODIS_META = META_DIR / "modis_ndvi_items_used.json"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

WORLDCOVER_COLLECTION = "esa-worldcover"
MODIS_COLLECTION = "modis-13Q1-061"

WORLDCOVER_REFERENCE_YEAR = 2021

NER_BBOX = [87.5, 21.5, 97.5, 29.5]

MODIS_NDVI_ASSET = "250m_16_days_NDVI"
MODIS_RELIABILITY_ASSET = "250m_16_days_pixel_reliability"

# NDVI product is a 16-day composite. We accept the nearest composite
# within this distance from the historical event date.
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


# ============================================================
# HELPERS
# ============================================================

def find_input_file() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find Stage 3 output.\n"
        "Expected:\n"
        "data/processed/rainfall/ner_events_with_dem_rainfall.csv"
    )


def point_inside_bbox(lon: float, lat: float, bbox) -> bool:
    if bbox is None or len(bbox) < 4:
        return False

    west, south, east, north = bbox[:4]
    eps = 1e-9

    return (
        west - eps <= lon <= east + eps
        and south - eps <= lat <= north + eps
    )


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


def choose_worldcover_item(items, lon: float, lat: float):
    matches = [
        item
        for item in items
        if point_inside_bbox(lon, lat, item.bbox)
    ]

    if not matches:
        return None

    # Prefer 2021/v2 product explicitly if metadata is available.
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

        distance_days = abs(
            (midpoint.normalize() - event_date.normalize()).days
        )

        if contains_date or distance_days <= MAX_NDVI_DATE_DISTANCE_DAYS:
            candidates.append(
                (
                    0 if contains_date else 1,
                    distance_days,
                    item.id,
                    item,
                )
            )

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][3]


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
    coords = transformed_xy(
        dataset,
        lon_values,
        lat_values,
    )

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
    """
    Open a signed Planetary Computer raster directly.

    Important:
    We do NOT manually call rasterio.Env().__enter__/__exit__ here.
    Rasterio/GDAL manages the dataset lifecycle through dataset.close(),
    which avoids nested GDAL-environment teardown errors on Windows.
    """
    signed_href = planetary_computer.sign(href)
    return rasterio.open(signed_href)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 76)
    print("STAGE 4 — REAL LAND COVER + HISTORICAL NDVI")
    print("=" * 76)

    input_file = find_input_file()
    print(f"\nInput file:\n{input_file}")

    df = pd.read_csv(input_file)

    required = {"latitude", "longitude", "event_date"}
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["_event_date"] = pd.to_datetime(df["event_date"], errors="coerce", utc=True)

    valid = (
        df["latitude"].notna()
        & df["longitude"].notna()
        & df["_event_date"].notna()
    )

    print(f"Rows: {len(df)}")
    print(f"Rows with valid coordinate + date: {int(valid.sum())}")

    if not valid.any():
        raise RuntimeError("No valid events available for Stage 4.")

    min_date = df.loc[valid, "_event_date"].min() - pd.Timedelta(days=20)
    max_date = df.loc[valid, "_event_date"].max() + pd.Timedelta(days=20)

    print("\nConnecting to Microsoft Planetary Computer STAC...")
    catalog = Client.open(STAC_URL)

    # ========================================================
    # ESA WORLDCOVER 2021
    # ========================================================

    print("\nSearching ESA WorldCover 2021...")
    wc_search = catalog.search(
        collections=[WORLDCOVER_COLLECTION],
        bbox=NER_BBOX,
        datetime="2021-01-01/2021-12-31",
    )

    wc_items = list(wc_search.items())

    print(f"WorldCover items found: {len(wc_items)}")

    if not wc_items:
        raise RuntimeError("No ESA WorldCover items found for NER.")

    WORLDCOVER_META.write_text(
        json.dumps(
            [
                {
                    "id": item.id,
                    "bbox": item.bbox,
                    "datetime": (
                        None
                        if item.datetime is None
                        else item.datetime.isoformat()
                    ),
                    "product_version": item.properties.get(
                        "esa_worldcover:product_version"
                    ),
                    "assets": list(item.assets.keys()),
                }
                for item in wc_items
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    df["worldcover_item_id"] = None

    for idx, row in df.loc[valid].iterrows():
        item = choose_worldcover_item(
            wc_items,
            float(row["longitude"]),
            float(row["latitude"]),
        )

        if item is not None:
            df.at[idx, "worldcover_item_id"] = item.id

    # ========================================================
    # MODIS NDVI
    # ========================================================

    print("\nSearching historical MODIS 16-day NDVI...")
    print(f"Date range: {min_date.date()} to {max_date.date()}")

    modis_search = catalog.search(
        collections=[MODIS_COLLECTION],
        bbox=NER_BBOX,
        datetime=f"{min_date.date()}/{max_date.date()}",
    )

    modis_items = list(modis_search.items())

    print(f"MODIS items returned: {len(modis_items)}")

    if not modis_items:
        raise RuntimeError("No MODIS NDVI items returned for the event period.")

    df["modis_item_id"] = None
    df["ndvi_composite_midpoint"] = None
    df["ndvi_date_distance_days"] = np.nan

    used_modis_items = {}

    print("\nMatching each event to the closest valid MODIS composite...")

    for counter, (idx, row) in enumerate(df.loc[valid].iterrows(), start=1):
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

            distance = abs(
                (
                    midpoint.normalize()
                    - event_date.normalize()
                ).days
            )

            df.at[idx, "modis_item_id"] = item.id
            df.at[idx, "ndvi_composite_midpoint"] = midpoint.isoformat()
            df.at[idx, "ndvi_date_distance_days"] = distance

            used_modis_items[item.id] = item

        if counter % 50 == 0 or counter == int(valid.sum()):
            print(f"  matched {counter}/{int(valid.sum())} events")

    MODIS_META.write_text(
        json.dumps(
            [
                {
                    "id": item.id,
                    "bbox": item.bbox,
                    "midpoint": (
                        None
                        if pd.isna(item_midpoint(item))
                        else item_midpoint(item).isoformat()
                    ),
                    "assets": list(item.assets.keys()),
                }
                for item in used_modis_items.values()
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # SAMPLE WORLDCOVER
    # ========================================================

    df["land_cover_code"] = np.nan
    df["land_cover"] = None
    df["land_cover_source"] = "ESA WorldCover 2021 v200"
    df["land_cover_reference_year"] = WORLDCOVER_REFERENCE_YEAR
    df["land_cover_year_gap"] = (
        WORLDCOVER_REFERENCE_YEAR
        - df["_event_date"].dt.year
    ).abs()

    wc_by_id = {item.id: item for item in wc_items}

    print("\nSampling ESA WorldCover classes...")

    grouped_wc = df.dropna(subset=["worldcover_item_id"]).groupby(
        "worldcover_item_id"
    )

    processed = 0

    for item_id, group in grouped_wc:
        item = wc_by_id[str(item_id)]

        if "map" not in item.assets:
            raise RuntimeError(
                f"WorldCover item {item_id} does not contain the 'map' asset."
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
                code,
                f"Unknown class {code}",
            )

        processed += len(group)
        print(f"  sampled {processed}/{df['worldcover_item_id'].notna().sum()} events")

    # ========================================================
    # SAMPLE MODIS NDVI + RELIABILITY
    # ========================================================

    df["ndvi_raw"] = np.nan
    df["ndvi"] = np.nan
    df["ndvi_pixel_reliability"] = np.nan
    df["ndvi_quality"] = None
    df["ndvi_quality_ok"] = False
    df["ndvi_source"] = "NASA MODIS MOD13Q1/MYD13Q1 v6.1 16-day 250m"

    print("\nSampling MODIS NDVI and pixel reliability...")

    grouped_modis = df.dropna(subset=["modis_item_id"]).groupby(
        "modis_item_id"
    )

    processed = 0

    for item_id, group in grouped_modis:
        item = used_modis_items[str(item_id)]

        if MODIS_NDVI_ASSET not in item.assets:
            raise RuntimeError(
                f"MODIS item {item_id} missing asset {MODIS_NDVI_ASSET}"
            )

        ndvi_ds = open_remote_raster(
            item.assets[MODIS_NDVI_ASSET].href
        )

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
            # Close datasets only. No manual rasterio.Env teardown.
            # Close the most recently opened raster first.
            if reliability_ds is not None:
                reliability_ds.close()

            if ndvi_ds is not None:
                ndvi_ds.close()

        for idx, raw_value, reliability in zip(
            group.index,
            raw_values,
            reliability_values,
        ):
            if pd.notna(raw_value):
                df.at[idx, "ndvi_raw"] = raw_value

                # MOD13Q1 valid NDVI scaled integer range.
                if -2000 <= raw_value <= 10000:
                    df.at[idx, "ndvi"] = float(raw_value) * 0.0001

            if pd.notna(reliability):
                reliability = int(round(reliability))
                df.at[idx, "ndvi_pixel_reliability"] = reliability
                df.at[idx, "ndvi_quality"] = MODIS_RELIABILITY.get(
                    reliability,
                    f"Unknown reliability {reliability}",
                )

                # 0=good and 1=marginal are considered usable.
                df.at[idx, "ndvi_quality_ok"] = reliability in {0, 1}

        processed += len(group)
        print(f"  sampled {processed}/{df['modis_item_id'].notna().sum()} events")

    # ========================================================
    # FINAL STATUS FLAGS
    # ========================================================

    df["land_cover_extraction_ok"] = df["land_cover_code"].notna()

    df["ndvi_extraction_ok"] = (
        df["ndvi"].notna()
        & df["ndvi_pixel_reliability"].notna()
    )

    # Remove internal parsed datetime before save.
    output = df.drop(columns=["_event_date"])

    output.to_csv(OUTPUT_CSV, index=False)

    landcover_ok = int(output["land_cover_extraction_ok"].sum())
    ndvi_ok = int(output["ndvi_extraction_ok"].sum())
    ndvi_quality_ok = int(output["ndvi_quality_ok"].sum())

    summary = {
        "rows": int(len(output)),
        "esa_worldcover_2021_success": landcover_ok,
        "modis_ndvi_success": ndvi_ok,
        "modis_ndvi_good_or_marginal_quality": ndvi_quality_ok,
        "worldcover_collection": WORLDCOVER_COLLECTION,
        "modis_collection": MODIS_COLLECTION,
        "output_csv": str(OUTPUT_CSV),
        "important_note": (
            "WorldCover 2021 is used as a static susceptibility proxy. "
            "Historical MODIS NDVI is matched near each event date."
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("STAGE 4 COMPLETE")
    print("=" * 76)

    print(f"Rows processed: {len(output)}")
    print(f"Land-cover extractions: {landcover_ok}/{len(output)}")
    print(f"Historical NDVI extractions: {ndvi_ok}/{len(output)}")
    print(
        f"NDVI good/marginal quality: "
        f"{ndvi_quality_ok}/{len(output)}"
    )

    if landcover_ok:
        print("\nLand-cover distribution:")
        print(
            output["land_cover"]
            .fillna("Missing")
            .value_counts()
        )

    if ndvi_ok:
        print("\nNDVI summary:")
        print(
            output.loc[
                output["ndvi_extraction_ok"] == True,
                "ndvi",
            ].describe().round(4)
        )

        print("\nNDVI reliability distribution:")
        print(
            output["ndvi_quality"]
            .fillna("Missing")
            .value_counts()
        )

    print(f"\nProcessed output:\n{OUTPUT_CSV}")
    print(f"\nWorldCover metadata:\n{WORLDCOVER_META}")
    print(f"\nMODIS metadata:\n{MODIS_META}")

    print(
        "\nIMPORTANT: ESA WorldCover is a 2021 static land-cover map, "
        "so it is a susceptibility proxy for older events rather than "
        "a reconstruction of land cover on the historical event date."
    )


if __name__ == "__main__":
    main()
