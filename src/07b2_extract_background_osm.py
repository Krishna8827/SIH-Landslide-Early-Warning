from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = (
    ROOT / "data" / "processed" / "background" /
    "ner_background_points_matched.csv"
)
STAGE6 = ROOT / "src" / "06_extract_osm_roads_rivers.py"
OUT_DIR = ROOT / "data" / "processed" / "background_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = OUT_DIR / "background_osm_checkpoint.csv"
OUTPUT = OUT_DIR / "background_osm_features.csv"

if not BACKGROUND.exists():
    raise FileNotFoundError(BACKGROUND)
if not STAGE6.exists():
    raise FileNotFoundError(
        "Stage 6 OSM script not found:\n"
        "src\\06_extract_osm_roads_rivers.py"
    )

spec = importlib.util.spec_from_file_location("stage6_osm", STAGE6)
osm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(osm)


def failed_result():
    return {
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


def main():
    print("=" * 80)
    print("STAGE 7B-2 — BACKGROUND OSM ONLY (POWER-OFF RESUMABLE)")
    print("=" * 80)

    df = pd.read_csv(BACKGROUND)
    required = {"background_id", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["background_id", "latitude", "longitude"]).copy()
    df["background_id"] = df["background_id"].astype(str)
    df = df.reset_index(drop=True)

    completed = {}
    if CHECKPOINT.exists():
        cp = pd.read_csv(CHECKPOINT)
        if "background_id" in cp.columns:
            cp["background_id"] = cp["background_id"].astype(str)
            for _, row in cp.iterrows():
                completed[row["background_id"]] = row.to_dict()
            print(f"Resuming checkpoint: {len(completed)} rows already saved")

    results = []

    for i, row in df.iterrows():
        bgid = row["background_id"]

        if bgid in completed:
            results.append(completed[bgid])
            if (i + 1) % 25 == 0:
                print(f"  reused/resumed {i+1}/{len(df)}")
            continue

        lat = float(row["latitude"])
        lon = float(row["longitude"])
        print(f"[{i+1}/{len(df)}] {bgid} lat={lat:.5f} lon={lon:.5f}")

        result = {
            "background_id": bgid,
            "latitude": lat,
            "longitude": lon,
        }

        try:
            result.update(osm.process_location(lat, lon))
            print(
                "  road=", result.get("distance_to_road_m"),
                "river=", result.get("distance_to_river_m")
            )
        except Exception as error:
            print(f"  ERROR: {error}")
            result.update(failed_result())

        results.append(result)

        # Save after EVERY completed location. If power goes off, rerunning
        # this script skips all saved background_id values.
        pd.DataFrame(results).to_csv(CHECKPOINT, index=False)

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT, index=False)

    road_ok = int(out["road_extraction_ok"].fillna(False).sum())
    river_ok = int(out["river_extraction_ok"].fillna(False).sum())
    both_ok = int(out["roads_rivers_extraction_ok"].fillna(False).sum())

    print("\n" + "=" * 80)
    print("STAGE 7B-2 COMPLETE")
    print("=" * 80)
    print(f"Rows processed: {len(out)}")
    print(f"Road distance extracted: {road_ok}/{len(out)}")
    print(f"River distance extracted: {river_ok}/{len(out)}")
    print(f"Both available: {both_ok}/{len(out)}")
    print(f"\nOutput:\n{OUTPUT}")
    print(f"\nResume checkpoint:\n{CHECKPOINT}")


if __name__ == "__main__":
    main()
