from pathlib import Path
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

EVENT_CANDIDATES = [
    ROOT / "data" / "raw" / "historical" / "ner_landslide_events_cleaned.csv",
    ROOT / "data" / "ner_landslide_events_cleaned.csv",
    ROOT / "ner_landslide_events_cleaned.csv",
]

RAW_DEM_DIR = ROOT / "data" / "raw" / "dem" / "copernicus_glo30"
PROCESSED_DIR = ROOT / "data" / "processed" / "terrain"

RAW_DEM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = PROCESSED_DIR / "ner_events_with_copernicus_dem.csv"
STAC_METADATA = RAW_DEM_DIR / "stac_items.json"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "cop-dem-glo-30"

# NER study box: [west, south, east, north]
NER_BBOX = [87.5, 21.5, 97.5, 29.5]

WINDOW_RADIUS = 4  # 9x9 raster neighborhood


# ============================================================
# HELPERS
# ============================================================

def find_event_file() -> Path:
    for path in EVENT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find ner_landslide_events_cleaned.csv.\n"
        "Put it at:\n"
        "data/raw/historical/ner_landslide_events_cleaned.csv"
    )


def point_inside_bbox(lon: float, lat: float, bbox) -> bool:
    west, south, east, north = bbox
    # small epsilon avoids edge-rounding issues
    eps = 1e-9
    return (
        west - eps <= lon <= east + eps
        and south - eps <= lat <= north + eps
    )


def choose_item_for_point(items, lon: float, lat: float):
    matches = [item for item in items if point_inside_bbox(lon, lat, item.bbox)]
    if not matches:
        return None

    # If a point lies exactly on tile boundaries, choose the smallest-area bbox.
    matches.sort(
        key=lambda item: abs(
            (item.bbox[2] - item.bbox[0]) *
            (item.bbox[3] - item.bbox[1])
        )
    )
    return matches[0]


def download_file(url: str, destination: Path):
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using cached DEM tile: {destination.name}")
        return

    temp_path = destination.with_suffix(destination.suffix + ".part")

    print(f"Downloading {destination.name} ...")

    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(temp_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue

                handle.write(chunk)
                downloaded += len(chunk)

                if total > 0:
                    percent = 100.0 * downloaded / total
                    print(
                        f"\r  {downloaded / 1024 / 1024:.1f} MB "
                        f"/ {total / 1024 / 1024:.1f} MB "
                        f"({percent:.1f}%)",
                        end="",
                    )

    print()
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
    """
    Return approximate x/y pixel spacing in meters.
    Handles both geographic and projected DEM rasters.
    """
    transform = dataset.transform

    if dataset.crs and dataset.crs.is_projected:
        dx = abs(transform.a)
        dy = abs(transform.e)
        return dx, dy

    # For geographic rasters, convert degree spacing to meters locally.
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

    # Fill occasional missing neighboring cells using the local median.
    if np.isnan(z).any():
        local_median = np.nanmedian(z)
        if not np.isfinite(local_median):
            return None
        z = np.where(np.isnan(z), local_median, z)

    dx, dy = pixel_spacing_m(dataset, lat)

    if dx <= 0 or dy <= 0:
        return None

    # np.gradient returns [gradient along rows(y), gradient along columns(x)].
    dz_dy, dz_dx = np.gradient(z, dy, dx)

    gx = float(dz_dx[center, center])
    gy = float(dz_dy[center, center])

    slope_rad = math.atan(math.sqrt(gx * gx + gy * gy))
    slope_deg = math.degrees(slope_rad)

    # Aspect: downslope direction clockwise from north.
    aspect_deg = (
        math.degrees(math.atan2(-gx, gy)) + 360.0
    ) % 360.0

    if slope_deg < 0.01:
        aspect_deg = np.nan

    # Simple profile-independent Laplacian curvature approximation.
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
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 72)
    print("STAGE 2 — COPERNICUS GLO-30 TERRAIN FEATURE EXTRACTION")
    print("=" * 72)

    event_file = find_event_file()
    print(f"\nHistorical event file:\n{event_file}")

    events = pd.read_csv(event_file)

    required = {"latitude", "longitude"}
    missing = required - set(events.columns)
    if missing:
        raise RuntimeError(f"Missing columns in event CSV: {sorted(missing)}")

    events["latitude"] = pd.to_numeric(events["latitude"], errors="coerce")
    events["longitude"] = pd.to_numeric(events["longitude"], errors="coerce")
    events = events.dropna(subset=["latitude", "longitude"]).copy()

    print(f"Valid event coordinates: {len(events)}")
    print(f"Unique coordinate pairs: {events[['latitude', 'longitude']].drop_duplicates().shape[0]}")

    print("\nConnecting to Microsoft Planetary Computer STAC...")
    catalog = Client.open(STAC_URL)

    search = catalog.search(
        collections=[COLLECTION],
        bbox=NER_BBOX,
    )

    items = list(search.items())
    print(f"Copernicus GLO-30 tiles found in NER bbox: {len(items)}")

    if not items:
        raise RuntimeError(
            "No Copernicus DEM tiles returned. "
            "Check internet access to planetarycomputer.microsoft.com."
        )

    # Save tile metadata for provenance.
    STAC_METADATA.write_text(
        json.dumps(
            [
                {
                    "id": item.id,
                    "bbox": item.bbox,
                    "collection": item.collection_id,
                    "assets": list(item.assets.keys()),
                }
                for item in items
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    # Determine which DEM tiles are actually needed by event coordinates.
    event_item_ids = []
    item_by_id = {item.id: item for item in items}

    for _, row in events.iterrows():
        item = choose_item_for_point(
            items,
            float(row["longitude"]),
            float(row["latitude"]),
        )
        event_item_ids.append(None if item is None else item.id)

    events["dem_tile_id"] = event_item_ids

    missing_tile_count = int(events["dem_tile_id"].isna().sum())
    if missing_tile_count:
        print(f"WARNING: {missing_tile_count} events could not be matched to a DEM tile.")

    required_item_ids = sorted(
        events["dem_tile_id"].dropna().unique().tolist()
    )

    print(f"\nDEM tiles required for event locations: {len(required_item_ids)}")

    tile_paths = {}

    for i, item_id in enumerate(required_item_ids, start=1):
        item = item_by_id[item_id]

        if "data" not in item.assets:
            raise RuntimeError(
                f"DEM STAC item {item_id} has no 'data' asset. "
                f"Available assets: {list(item.assets.keys())}"
            )

        signed_href = planetary_computer.sign(item.assets["data"].href)

        local_path = RAW_DEM_DIR / f"{item_id}.tif"

        print(f"\nTile {i}/{len(required_item_ids)}")
        download_file(signed_href, local_path)

        tile_paths[item_id] = local_path

    print("\nExtracting elevation, slope, aspect and curvature...")

    results = []
    open_datasets = {}

    try:
        for index, row in events.iterrows():
            item_id = row["dem_tile_id"]

            feature_result = {
                "dem_source": "Copernicus DEM GLO-30 via Microsoft Planetary Computer",
                "terrain_extraction_ok": False,
            }

            if pd.isna(item_id):
                results.append(feature_result)
                continue

            item_id = str(item_id)

            if item_id not in open_datasets:
                open_datasets[item_id] = rasterio.open(tile_paths[item_id])

            dataset = open_datasets[item_id]

            terrain = extract_terrain_features(
                dataset,
                float(row["longitude"]),
                float(row["latitude"]),
            )

            if terrain is not None:
                feature_result.update(terrain)
                feature_result["terrain_extraction_ok"] = True

            results.append(feature_result)

            processed = len(results)
            if processed % 25 == 0 or processed == len(events):
                print(f"  processed {processed}/{len(events)} events")

    finally:
        for dataset in open_datasets.values():
            dataset.close()

    terrain_df = pd.DataFrame(results)

    output = pd.concat(
        [
            events.reset_index(drop=True),
            terrain_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Reliability note based on source-coordinate uncertainty.
    if "location_precision" in output.columns:
        output["fine_scale_terrain_reliable"] = (
            output["location_precision"]
            .astype(str)
            .str.contains("High", case=False, na=False)
        )
    else:
        output["fine_scale_terrain_reliable"] = pd.NA

    output.to_csv(OUTPUT_CSV, index=False)

    successful = int(output["terrain_extraction_ok"].sum())

    print("\n" + "=" * 72)
    print("STAGE 2 COMPLETE")
    print("=" * 72)

    print(f"Events processed: {len(output)}")
    print(f"Successful terrain extractions: {successful}")
    print(f"Failed terrain extractions: {len(output) - successful}")

    if successful:
        valid = output[output["terrain_extraction_ok"] == True]

        print("\nTerrain feature summary:")
        print(
            valid[
                [
                    "elevation_m",
                    "slope_deg",
                    "aspect_deg",
                    "curvature_1_per_m",
                ]
            ].describe().round(3)
        )

    print(f"\nProcessed output:\n{OUTPUT_CSV}")
    print(f"\nRaw DEM directory:\n{RAW_DEM_DIR}")
    print(f"\nSTAC metadata:\n{STAC_METADATA}")

    print(
        "\nIMPORTANT: Copernicus GLO-30 is a DSM, so heights may include "
        "vegetation/buildings. Also, fine-scale terrain values are least reliable "
        "for historical events with large coordinate uncertainty."
    )


if __name__ == "__main__":
    main()
