from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

INPUT_CANDIDATES = [
    ROOT / "data" / "processed" / "soil_moisture" / "ner_events_with_dem_rainfall_landcover_ndvi_soil_moisture.csv",
    ROOT / "data" / "processed" / "landcover_ndvi" / "ner_events_with_dem_rainfall_landcover_ndvi.csv",
]

RAW_DIR = ROOT / "data" / "raw" / "osm"
CACHE_DIR = RAW_DIR / "overpass_cache"
OUT_DIR = ROOT / "data" / "processed" / "roads_rivers"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUT_DIR / "ner_events_with_all_positive_features.csv"
SUMMARY_JSON = OUT_DIR / "roads_rivers_summary.json"

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

SEARCH_RADII_M = [5000, 15000, 30000]
ROAD_REGEX = (
    "motorway|motorway_link|trunk|trunk_link|primary|primary_link|"
    "secondary|secondary_link|tertiary|tertiary_link|unclassified|"
    "residential|service|living_street|road|track"
)
WATERWAY_REGEX = "river|stream|canal|drain"
MAX_RETRIES_PER_ENDPOINT = 2
TIMEOUT_SECONDS = 90
REQUEST_DELAY_SECONDS = 0.35


def find_input_file() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find Stage 5 output. Expected:\n"
        "data/processed/soil_moisture/"
        "ner_events_with_dem_rainfall_landcover_ndvi_soil_moisture.csv"
    )


def cache_key(latitude, longitude, radius_m, need_roads, need_waterways):
    raw = (
        f"{latitude:.6f}|{longitude:.6f}|{radius_m}|"
        f"roads={int(need_roads)}|waterways={int(need_waterways)}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_query(latitude, longitude, radius_m, need_roads, need_waterways):
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
        + "\n".join(f"  {c}" for c in clauses)
        + '\n);\nout tags geom;'
    )


def fetch_overpass(latitude, longitude, radius_m, need_roads, need_waterways):
    key = cache_key(latitude, longitude, radius_m, need_roads, need_waterways)
    cache_path = CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    query = build_query(latitude, longitude, radius_m, need_roads, need_waterways)
    last_error = None

    for endpoint_index, endpoint in enumerate(OVERPASS_ENDPOINTS, start=1):
        for attempt in range(1, MAX_RETRIES_PER_ENDPOINT + 1):
            try:
                response = requests.post(
                    endpoint,
                    data={"data": query},
                    timeout=TIMEOUT_SECONDS,
                    headers={
                        "User-Agent": "SIH-Landslide-Research/1.0",
                        "Accept": "application/json",
                    },
                )
                if response.status_code == 200:
                    payload = response.json()
                    if "elements" not in payload:
                        raise RuntimeError("Overpass response did not contain elements.")
                    cache_path.write_text(
                        json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    time.sleep(REQUEST_DELAY_SECONDS)
                    return payload

                last_error = RuntimeError(
                    f"{endpoint} returned HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )
                if response.status_code == 429 or response.status_code >= 500:
                    wait_seconds = 1.5 * attempt
                    print(f"      server busy; retrying in {wait_seconds:.1f}s")
                    time.sleep(wait_seconds)
            except Exception as error:
                last_error = error
                if attempt < MAX_RETRIES_PER_ENDPOINT:
                    time.sleep(1.5 * attempt)

        print(
            "      switching Overpass server "
            f"({endpoint_index}/{len(OVERPASS_ENDPOINTS)})"
        )

    raise RuntimeError(f"All Overpass endpoints failed. Last error: {last_error}")


def local_xy_m(latitude, longitude, origin_lat, origin_lon):
    y = (latitude - origin_lat) * 110_574.0
    x = (
        (longitude - origin_lon)
        * 111_320.0
        * math.cos(math.radians(origin_lat))
    )
    return x, y


def distance_origin_to_segment_m(ax, ay, bx, by):
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


def distance_to_geometry_m(origin_lat, origin_lon, geometry):
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
    for i in range(len(points) - 1):
        d = distance_origin_to_segment_m(
            points[i][0], points[i][1], points[i + 1][0], points[i + 1][1]
        )
        best = min(best, d)
    return best if math.isfinite(best) else np.nan


def split_elements(payload):
    roads, waterways = [], []
    for element in payload.get("elements", []):
        if element.get("type") != "way":
            continue
        tags = element.get("tags") or {}
        if tags.get("highway"):
            roads.append(element)
        if tags.get("waterway"):
            waterways.append(element)
    return roads, waterways


def nearest_element(elements, latitude, longitude):
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


def process_location(latitude, longitude):
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

        print(
            f"    search radius {radius_m / 1000:.0f} km "
            f"(roads={need_roads}, rivers={need_waterways})"
        )

        payload = fetch_overpass(
            latitude,
            longitude,
            radius_m,
            need_roads,
            need_waterways,
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
        "osm_snapshot_timestamp": max(osm_timestamps) if osm_timestamps else None,
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


def main():
    print("=" * 78)
    print("STAGE 6 — OPENSTREETMAP ROAD + RIVER DISTANCE EXTRACTION")
    print("=" * 78)

    input_file = find_input_file()
    print(f"\nInput file:\n{input_file}")

    df = pd.read_csv(input_file)
    required = {"latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    valid_mask = df["latitude"].notna() & df["longitude"].notna()

    print(f"Rows: {len(df)}")
    print(f"Rows with valid coordinates: {int(valid_mask.sum())}")

    unique_locations = (
        df.loc[valid_mask, ["latitude", "longitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"Unique coordinate pairs: {len(unique_locations)}")
    print(
        "\nNOTE: Public Overpass servers can occasionally be busy. "
        "Successful queries are cached and fallback servers are used automatically."
    )

    location_results = []

    for row_number, row in unique_locations.iterrows():
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])
        print(
            f"\n[{row_number + 1}/{len(unique_locations)}] "
            f"lat={latitude:.5f} lon={longitude:.5f}"
        )

        result = {"latitude": latitude, "longitude": longitude}
        try:
            result.update(process_location(latitude, longitude))
            road_text = (
                "missing"
                if pd.isna(result["distance_to_road_m"])
                else f"{result['distance_to_road_m']:.1f} m"
            )
            river_text = (
                "missing"
                if pd.isna(result["distance_to_river_m"])
                else f"{result['distance_to_river_m']:.1f} m"
            )
            print(f"    nearest road: {road_text}")
            print(f"    nearest river/stream: {river_text}")
        except Exception as error:
            print(f"    ERROR: {error}")
            result.update(
                {
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
            )

        location_results.append(result)

        if (row_number + 1) % 25 == 0:
            pd.DataFrame(location_results).to_csv(
                OUT_DIR / "roads_rivers_checkpoint.csv",
                index=False,
            )

    features = pd.DataFrame(location_results)
    output = df.merge(
        features,
        on=["latitude", "longitude"],
        how="left",
        validate="many_to_one",
    )
    output.to_csv(OUTPUT_CSV, index=False)

    road_ok = int(output["road_extraction_ok"].fillna(False).sum())
    river_ok = int(output["river_extraction_ok"].fillna(False).sum())
    both_ok = int(output["roads_rivers_extraction_ok"].fillna(False).sum())

    summary = {
        "rows": int(len(output)),
        "unique_coordinate_pairs": int(len(unique_locations)),
        "road_extractions": road_ok,
        "river_extractions": river_ok,
        "both_features_extracted": both_ok,
        "source": "OpenStreetMap via Overpass API",
        "search_radii_m": SEARCH_RADII_M,
        "output_csv": str(OUTPUT_CSV),
        "important_note": (
            "Current OpenStreetMap infrastructure is used as a static "
            "susceptibility proxy, not as a guaranteed historical reconstruction."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("STAGE 6 COMPLETE")
    print("=" * 78)
    print(f"Rows processed: {len(output)}")
    print(f"Road distances extracted: {road_ok}/{len(output)}")
    print(f"River/stream distances extracted: {river_ok}/{len(output)}")
    print(f"Both available: {both_ok}/{len(output)}")

    if road_ok:
        print("\nDistance-to-road summary (m):")
        print(output["distance_to_road_m"].dropna().describe().round(2))
        print("\nNearest road types:")
        print(output["nearest_road_type"].fillna("Missing").value_counts().head(15))

    if river_ok:
        print("\nDistance-to-river/stream summary (m):")
        print(output["distance_to_river_m"].dropna().describe().round(2))
        print("\nNearest waterway types:")
        print(output["nearest_waterway_type"].fillna("Missing").value_counts().head(15))

    print(f"\nProcessed output:\n{OUTPUT_CSV}")
    print(f"\nOverpass cache:\n{CACHE_DIR}")
    print(
        "\nIMPORTANT: Current OSM roads/waterways are static susceptibility "
        "proxies. Do not describe them as event-date historical road maps."
    )


if __name__ == "__main__":
    main()
