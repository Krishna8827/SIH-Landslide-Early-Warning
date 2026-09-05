from pathlib import Path
import json, math, random, requests
import pandas as pd
from shapely.geometry import Point, shape

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "roads_rivers" / "ner_events_with_all_positive_features.csv"
RAW_DIR = ROOT / "data" / "raw" / "boundaries"
OUT_DIR = ROOT / "data" / "processed" / "background"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARY_API = "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM1/"
BOUNDARY_GEOJSON = RAW_DIR / "geoBoundaries_IND_ADM1.geojson"
OUTPUT = OUT_DIR / "ner_background_points_matched.csv"
SUMMARY = OUT_DIR / "background_sampling_summary.json"

RANDOM_SEED = 42
MIN_EVENT_DISTANCE_KM = 5.0
MIN_BACKGROUND_DISTANCE_KM = 1.0
MAX_ATTEMPTS = 20000

NER_STATES = {
    "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "sikkim": "Sikkim",
    "tripura": "Tripura",
}

def clean_state(v):
    if pd.isna(v):
        return None
    s = " ".join(str(v).strip(",; ").split())
    return NER_STATES.get(s.lower(), s)

def state_column(df):
    for c in ["state", "STATE", "State", "state_name", "STATE_NAME"]:
        if c in df.columns:
            return c
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.atan2(math.sqrt(a), math.sqrt(1-a))

def min_dist(lat, lon, pts):
    if not pts:
        return float("inf")
    return min(haversine_km(lat, lon, a, b) for a, b in pts)

def download_boundaries():
    if BOUNDARY_GEOJSON.exists() and BOUNDARY_GEOJSON.stat().st_size > 1000:
        return
    meta = requests.get(
        BOUNDARY_API,
        timeout=90,
        headers={"User-Agent": "SIH-Landslide-Research/1.0"},
    )
    meta.raise_for_status()
    url = meta.json()["gjDownloadURL"]
    r = requests.get(
        url,
        timeout=180,
        headers={"User-Agent": "SIH-Landslide-Research/1.0"},
    )
    r.raise_for_status()
    BOUNDARY_GEOJSON.write_bytes(r.content)
    (RAW_DIR / "geoBoundaries_IND_ADM1_metadata.json").write_text(
        json.dumps(meta.json(), indent=2), encoding="utf-8"
    )

def load_polygons():
    download_boundaries()
    payload = json.loads(BOUNDARY_GEOJSON.read_text(encoding="utf-8"))
    polygons = {}
    for f in payload.get("features", []):
        p = f.get("properties") or {}
        name = None
        for candidate in [p.get("shapeName"), p.get("NAME_1"), p.get("name"), p.get("ST_NM")]:
            c = clean_state(candidate)
            if c in NER_STATES.values():
                name = c
                break
        if not name or not f.get("geometry"):
            continue
        geom = shape(f["geometry"])
        polygons[name] = polygons[name].union(geom) if name in polygons else geom
    return polygons

def random_inside(poly, rng):
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(MAX_ATTEMPTS):
        lon, lat = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if poly.contains(Point(lon, lat)):
            return lat, lon
    return None

def main():
    print("=" * 76)
    print("STAGE 7A — MATCHED BACKGROUND / PSEUDO-ABSENCE SAMPLING")
    print("=" * 76)

    if not INPUT.exists():
        raise FileNotFoundError(f"Stage 6 output not found:\n{INPUT}")

    df = pd.read_csv(INPUT)
    sc = state_column(df)
    if sc is None:
        raise RuntimeError("No state column found.")

    for c in ["latitude", "longitude", "event_date"]:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}")

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["_state"] = df[sc].apply(clean_state)
    df["_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "_state", "_date"]).reset_index(drop=True)

    print(f"Positive rows: {len(df)}")
    print("\nPositive events by state:")
    print(df["_state"].value_counts())

    polygons = load_polygons()
    positive_pts = list(zip(df["latitude"].astype(float), df["longitude"].astype(float)))
    bg_pts = []
    rows = []
    rng = random.Random(RANDOM_SEED)

    for i, row in df.iterrows():
        state = row["_state"]
        if state not in polygons:
            raise RuntimeError(f"Missing boundary polygon for state: {state}")

        chosen = None
        for _ in range(MAX_ATTEMPTS):
            sample = random_inside(polygons[state], rng)
            if sample is None:
                continue
            lat, lon = sample
            d_event = min_dist(lat, lon, positive_pts)
            d_bg = min_dist(lat, lon, bg_pts)
            if d_event >= MIN_EVENT_DISTANCE_KM and d_bg >= MIN_BACKGROUND_DISTANCE_KM:
                chosen = (lat, lon, d_event, d_bg)
                break

        if chosen is None:
            raise RuntimeError(
                f"Could not sample background point for {state}. "
                "Reduce MIN_EVENT_DISTANCE_KM if necessary."
            )

        lat, lon, d_event, d_bg = chosen
        bg_pts.append((lat, lon))

        event_id = row["event_id"] if "event_id" in df.columns else i + 1
        rows.append({
            "background_id": f"BG_{i+1:04d}",
            "matched_positive_event_id": event_id,
            "state": state,
            "latitude": lat,
            "longitude": lon,
            "event_date": row["_date"].strftime("%Y-%m-%d"),
            "landslide": 0,
            "sample_type": "matched_background_pseudo_absence",
            "is_confirmed_stable": False,
            "min_distance_to_known_event_km": d_event,
            "min_distance_to_other_background_km": d_bg,
            "random_seed": RANDOM_SEED,
        })

        if (i + 1) % 25 == 0 or i + 1 == len(df):
            print(f"  generated {i+1}/{len(df)} background points")

    bg = pd.DataFrame(rows)
    bg.to_csv(OUTPUT, index=False)

    summary = {
        "positive_rows": int(len(df)),
        "background_rows": int(len(bg)),
        "same_state_matching": True,
        "same_date_matching": True,
        "min_distance_from_known_event_km": MIN_EVENT_DISTANCE_KM,
        "min_distance_between_background_km": MIN_BACKGROUND_DISTANCE_KM,
        "random_seed": RANDOM_SEED,
        "label_note": "landslide=0 is pseudo-absence/background, not confirmed stable",
        "boundary_source": "geoBoundaries gbOpen India ADM1",
        "output_csv": str(OUTPUT),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print("STAGE 7A COMPLETE")
    print("=" * 76)
    print(f"Background rows generated: {len(bg)}")
    print("\nBy state:")
    print(bg["state"].value_counts())
    print("\nDistance from nearest known event (km):")
    print(bg["min_distance_to_known_event_km"].describe().round(2))
    print(f"\nOutput:\n{OUTPUT}")
    print("\nIMPORTANT: These are pseudo-absence/background samples, not confirmed stable sites.")

if __name__ == "__main__":
    main()
