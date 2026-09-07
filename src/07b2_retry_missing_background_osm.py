from pathlib import Path
import json, math, time
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data" / "processed" / "background_features" / "background_osm_features.csv"
CHECKPOINT = ROOT / "data" / "processed" / "background_features" / "background_osm_checkpoint.csv"
SUMMARY = ROOT / "data" / "processed" / "background_features" / "background_osm_retry_summary.json"

ENDPOINTS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.maprva.org/api/interpreter",
    "https://ethiopia.overpass.openplaceguide.org/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

RADII = [5000, 15000, 30000]
ROAD_RE = (
    "motorway|motorway_link|trunk|trunk_link|primary|primary_link|"
    "secondary|secondary_link|tertiary|tertiary_link|unclassified|"
    "residential|service|living_street|road|track"
)
RIVER_RE = "river|stream|canal|drain"

def query(lat, lon, radius, kind):
    if kind == "road":
        clause = f'way(around:{radius},{lat:.7f},{lon:.7f})["highway"~"^({ROAD_RE})$"];'
    else:
        clause = f'way(around:{radius},{lat:.7f},{lon:.7f})["waterway"~"^({RIVER_RE})$"];'
    return f"[out:json][timeout:35];({clause});out tags geom;"

def fetch(lat, lon, radius, kind):
    q = query(lat, lon, radius, kind)
    last = None
    for i, endpoint in enumerate(ENDPOINTS, 1):
        try:
            print(f"      server {i}/{len(ENDPOINTS)}")
            r = requests.post(
                endpoint,
                data={"data": q},
                headers={"User-Agent": "SIH-Landslide-Research/1.0"},
                timeout=45,
            )
            if r.status_code == 200:
                return r.json()
            last = RuntimeError(f"HTTP {r.status_code}")
            if r.status_code == 429:
                time.sleep(3)
            elif r.status_code >= 500:
                time.sleep(1)
        except Exception as e:
            last = e
    raise RuntimeError(last)

def local_xy(lat, lon, lat0, lon0):
    return (
        (lon-lon0) * 111320.0 * math.cos(math.radians(lat0)),
        (lat-lat0) * 110574.0,
    )

def segdist(ax, ay, bx, by):
    dx, dy = bx-ax, by-ay
    den = dx*dx + dy*dy
    if den <= 0:
        return math.hypot(ax, ay)
    t = -(ax*dx + ay*dy)/den
    t = max(0.0, min(1.0, t))
    return math.hypot(ax+t*dx, ay+t*dy)

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
    return min(segdist(*pts[i], *pts[i+1]) for i in range(len(pts)-1))

def nearest(payload, lat, lon, kind):
    best = None
    best_d = float("inf")
    for e in payload.get("elements", []):
        if e.get("type") != "way":
            continue
        tags = e.get("tags") or {}
        if kind == "road" and not tags.get("highway"):
            continue
        if kind == "river" and not tags.get("waterway"):
            continue
        d = geometry_distance(lat, lon, e.get("geometry"))
        if pd.notna(d) and d < best_d:
            best_d, best = d, e
    return None if best is None else (float(best_d), best)

def find_feature(lat, lon, kind):
    for radius in RADII:
        print(f"    {kind}: radius {radius//1000} km")
        try:
            result = nearest(fetch(lat, lon, radius, kind), lat, lon, kind)
        except Exception as e:
            print(f"      failed: {str(e)[:150]}")
            continue
        if result:
            d, e = result
            return d, e, radius
    return None

def save(df):
    df.to_csv(FILE, index=False)
    df.to_csv(CHECKPOINT, index=False)

def main():
    print("="*80)
    print("STAGE 7B-2R — RETRY ONLY MISSING BACKGROUND OSM ROWS")
    print("="*80)

    if not FILE.exists():
        raise FileNotFoundError(FILE)

    df = pd.read_csv(FILE)

    for col in [
        "distance_to_road_m","distance_to_river_m",
        "nearest_road_type","nearest_road_name","nearest_road_osm_id",
        "nearest_waterway_type","nearest_waterway_name","nearest_waterway_osm_id",
        "road_search_radius_m","river_search_radius_m",
        "road_extraction_ok","river_extraction_ok","roads_rivers_extraction_ok",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    road_before = df["distance_to_road_m"].notna()
    river_before = df["distance_to_river_m"].notna()
    retry_idx = df.index[~(road_before & river_before)].tolist()

    print(f"Rows: {len(df)}")
    print(f"Road before: {int(road_before.sum())}/{len(df)}")
    print(f"River before: {int(river_before.sum())}/{len(df)}")
    print(f"Rows to retry: {len(retry_idx)}")

    for n, idx in enumerate(retry_idx, 1):
        row = df.loc[idx]
        bgid = row["background_id"]
        lat, lon = float(row["latitude"]), float(row["longitude"])
        print(f"\n[{n}/{len(retry_idx)}] {bgid} lat={lat:.5f} lon={lon:.5f}")

        if pd.isna(df.at[idx, "distance_to_road_m"]):
            found = find_feature(lat, lon, "road")
            if found:
                d, e, radius = found
                tags = e.get("tags") or {}
                df.at[idx, "distance_to_road_m"] = d
                df.at[idx, "nearest_road_type"] = tags.get("highway")
                df.at[idx, "nearest_road_name"] = tags.get("name")
                df.at[idx, "nearest_road_osm_id"] = e.get("id")
                df.at[idx, "road_search_radius_m"] = radius
                print(f"    ROAD FOUND: {d:.1f} m")

        if pd.isna(df.at[idx, "distance_to_river_m"]):
            found = find_feature(lat, lon, "river")
            if found:
                d, e, radius = found
                tags = e.get("tags") or {}
                df.at[idx, "distance_to_river_m"] = d
                df.at[idx, "nearest_waterway_type"] = tags.get("waterway")
                df.at[idx, "nearest_waterway_name"] = tags.get("name")
                df.at[idx, "nearest_waterway_osm_id"] = e.get("id")
                df.at[idx, "river_search_radius_m"] = radius
                print(f"    RIVER FOUND: {d:.1f} m")

        road_ok = pd.notna(df.at[idx, "distance_to_road_m"])
        river_ok = pd.notna(df.at[idx, "distance_to_river_m"])
        df.at[idx, "road_extraction_ok"] = road_ok
        df.at[idx, "river_extraction_ok"] = river_ok
        df.at[idx, "roads_rivers_extraction_ok"] = road_ok and river_ok
        save(df)

    road_after = df["distance_to_road_m"].notna()
    river_after = df["distance_to_river_m"].notna()
    both_after = road_after & river_after

    summary = {
        "rows": int(len(df)),
        "road_before": int(road_before.sum()),
        "river_before": int(river_before.sum()),
        "road_after": int(road_after.sum()),
        "river_after": int(river_after.sum()),
        "both_after": int(both_after.sum()),
        "remaining_missing": int((~both_after).sum()),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "="*80)
    print("STAGE 7B-2R COMPLETE")
    print("="*80)
    print(f"Road distance: {summary['road_after']}/{len(df)}")
    print(f"River distance: {summary['river_after']}/{len(df)}")
    print(f"Both available: {summary['both_after']}/{len(df)}")
    print(f"Still missing either: {summary['remaining_missing']}")
    print(f"\nUpdated file:\n{FILE}")

if __name__ == "__main__":
    main()
