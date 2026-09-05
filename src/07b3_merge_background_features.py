from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed" / "background_features"
NON_OSM = OUT_DIR / "background_non_osm_features.csv"
OSM = OUT_DIR / "background_osm_features.csv"
OUTPUT = OUT_DIR / "ner_background_points_with_all_features.csv"
SUMMARY = OUT_DIR / "background_features_final_summary.json"


def bool_count(df, column):
    if column not in df.columns:
        return 0
    return int(df[column].fillna(False).astype(bool).sum())


def main():
    print("=" * 80)
    print("STAGE 7B-3 — MERGE BACKGROUND FEATURES")
    print("=" * 80)

    if not NON_OSM.exists():
        raise FileNotFoundError(NON_OSM)
    if not OSM.exists():
        raise FileNotFoundError(OSM)

    left = pd.read_csv(NON_OSM)
    right = pd.read_csv(OSM)

    if "background_id" not in left.columns or "background_id" not in right.columns:
        raise RuntimeError("background_id is required in both Stage 7B files")

    left["background_id"] = left["background_id"].astype(str)
    right["background_id"] = right["background_id"].astype(str)

    # The 7B-1 file contains intentionally blank OSM placeholder columns.
    # Drop them before attaching the real 7B-2 OSM outputs.
    osm_columns = [
        "distance_to_road_m", "distance_to_river_m",
        "nearest_road_type", "nearest_road_name", "nearest_road_osm_id",
        "nearest_waterway_type", "nearest_waterway_name", "nearest_waterway_osm_id",
        "road_search_radius_m", "river_search_radius_m",
        "road_extraction_ok", "river_extraction_ok", "roads_rivers_extraction_ok",
        "osm_source", "osm_snapshot_timestamp", "distance_method",
    ]
    left = left.drop(columns=[c for c in osm_columns if c in left.columns], errors="ignore")

    right_keep = ["background_id"] + [c for c in osm_columns if c in right.columns]
    merged = left.merge(
        right[right_keep],
        on="background_id",
        how="left",
        validate="one_to_one",
    )

    merged["landslide"] = 0
    merged["sample_type"] = "matched_background_pseudo_absence"
    merged["is_confirmed_stable"] = False

    merged.to_csv(OUTPUT, index=False)

    summary = {
        "rows": int(len(merged)),
        "terrain_ok": bool_count(merged, "terrain_extraction_ok"),
        "rainfall_ok": bool_count(merged, "rainfall_extraction_ok"),
        "land_cover_ok": bool_count(merged, "land_cover_extraction_ok"),
        "ndvi_ok": bool_count(merged, "ndvi_extraction_ok"),
        "ndvi_model_usable": int(merged["ndvi_model_value"].notna().sum()) if "ndvi_model_value" in merged.columns else 0,
        "soil_ok": bool_count(merged, "soil_moisture_extraction_ok"),
        "road_ok": bool_count(merged, "road_extraction_ok"),
        "river_ok": bool_count(merged, "river_extraction_ok"),
        "output_csv": str(OUTPUT),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("STAGE 7B COMPLETE")
    print("=" * 80)
    print(f"Rows: {len(merged)}")
    print(f"Terrain: {summary['terrain_ok']}/{len(merged)}")
    print(f"Rainfall: {summary['rainfall_ok']}/{len(merged)}")
    print(f"Land cover: {summary['land_cover_ok']}/{len(merged)}")
    print(f"Historical NDVI: {summary['ndvi_ok']}/{len(merged)}")
    print(f"Model-usable NDVI: {summary['ndvi_model_usable']}/{len(merged)}")
    print(f"Soil moisture: {summary['soil_ok']}/{len(merged)}")
    print(f"Road distance: {summary['road_ok']}/{len(merged)}")
    print(f"River distance: {summary['river_ok']}/{len(merged)}")
    print(f"\nFinal background feature file:\n{OUTPUT}")


if __name__ == "__main__":
    main()
