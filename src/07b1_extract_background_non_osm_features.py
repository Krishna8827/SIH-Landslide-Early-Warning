from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "07b_extract_features_for_background_points.py"

if not SOURCE.exists():
    raise FileNotFoundError(
        "Original Stage 7B script not found:\n"
        "src\\07b_extract_features_for_background_points.py"
    )

spec = importlib.util.spec_from_file_location("stage7b_original", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Use a separate non-OSM output so the original file is not overwritten.
module.OUTPUT_CSV = (
    ROOT / "data" / "processed" / "background_features" /
    "background_non_osm_features.csv"
)
module.SUMMARY_JSON = (
    ROOT / "data" / "processed" / "background_features" /
    "background_non_osm_summary.json"
)

# Disable Overpass completely in this pass. All other real feature stages in
# the original 7B pipeline still run: DEM, WorldCover, MODIS NDVI, rainfall,
# and ERA5-Land soil moisture.
def no_osm(latitude, longitude):
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
        "osm_source": "Deferred to Stage 7B-2",
        "osm_snapshot_timestamp": None,
        "distance_method": None,
    }

module.process_location = no_osm

print("=" * 80)
print("STAGE 7B-1 — BACKGROUND FEATURES EXCEPT OSM")
print("=" * 80)
print("OSM is intentionally disabled in this pass.")
print("The old combined 7B script is being reused only for DEM/rainfall/")
print("WorldCover/MODIS/ERA5-Land extraction.\n")

module.main()

print("\n" + "=" * 80)
print("STAGE 7B-1 FINISHED")
print("=" * 80)
print(f"Non-OSM output:\n{module.OUTPUT_CSV}")
print("Next run: python src\\07b2_extract_background_osm.py")
