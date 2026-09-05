from pathlib import Path
import io

import pandas as pd
import requests


# ============================================================
# NORTH-EAST INDIA BOUNDING BOX
# ============================================================

LAT_MIN = 21.5
LAT_MAX = 29.5

LON_MIN = 87.5
LON_MAX = 97.5


# ============================================================
# NASA OFFICIAL GLOBAL LANDSLIDE CATALOG CSV
# ============================================================

# Primary official NASA Open Data download.
NASA_URLS = [
    "https://data.nasa.gov/api/views/dd9e-wu2v/rows.csv?accessType=DOWNLOAD",

    # Official legacy NASA mirror.
    "https://data.nasa.gov/docs/legacy/"
    "Global_Landslide_Catalog_Export/"
    "Global_Landslide_Catalog_Export_rows.csv",
]


# ============================================================
# PROJECT PATHS
# ============================================================

# File location:
# project/src/01_download_nasa_coolr.py
#
# parents[1] = project root
ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "data" / "raw" / "nasa_glc"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_CSV_PATH = (
    OUT_DIR
    / "nasa_global_landslide_catalog_full.csv"
)

NER_CSV_PATH = (
    OUT_DIR
    / "nasa_glc_north_east_india.csv"
)

NER_GEOJSON_PATH = (
    OUT_DIR
    / "nasa_glc_north_east_india.geojson"
)


# ============================================================
# DOWNLOAD
# ============================================================

def download_nasa_csv():

    print("=" * 70)
    print("NASA GLOBAL LANDSLIDE CATALOG DOWNLOADER")
    print("=" * 70)

    for url in NASA_URLS:

        print("\nTrying NASA source:")
        print(url)

        try:

            response = requests.get(
                url,
                timeout=120,
                headers={
                    "User-Agent":
                    "SIH-Landslide-Research/1.0"
                },
            )

            print(
                "HTTP status:",
                response.status_code
            )

            if response.status_code != 200:

                print(
                    "Source unavailable. "
                    "Trying next NASA URL..."
                )

                continue

            # Make sure we did not receive an HTML page.
            content_type = (
                response.headers
                .get("content-type", "")
                .lower()
            )

            print(
                "Content-Type:",
                content_type
            )

            text_start = (
                response.content[:200]
                .decode(
                    "utf-8",
                    errors="ignore",
                )
                .lower()
            )

            if (
                "<html" in text_start
                or "<!doctype html" in text_start
            ):

                print(
                    "Received HTML instead of CSV."
                )

                continue

            RAW_CSV_PATH.write_bytes(
                response.content
            )

            print("\nNASA dataset downloaded.")
            print(
                "Raw file:",
                RAW_CSV_PATH
            )

            return RAW_CSV_PATH

        except Exception as error:

            print(
                "Download attempt failed:"
            )

            print(error)

    raise RuntimeError(
        "Both official NASA CSV download "
        "locations failed."
    )


# ============================================================
# FIND COLUMN SAFELY
# ============================================================

def find_column(df, candidates):

    lower_map = {
        col.lower().strip(): col
        for col in df.columns
    }

    for candidate in candidates:

        candidate = (
            candidate.lower()
            .strip()
        )

        if candidate in lower_map:

            return lower_map[candidate]

    return None


# ============================================================
# PROCESS DATA
# ============================================================

def process_catalog(csv_path):

    print("\n" + "=" * 70)
    print("PROCESSING NASA DATA")
    print("=" * 70)

    df = pd.read_csv(
        csv_path,
        low_memory=False,
    )

    print(
        "\nFull NASA catalog rows:",
        len(df)
    )

    print(
        "Number of columns:",
        len(df.columns)
    )

    print("\nAvailable columns:")

    for col in df.columns:
        print(" -", col)


    # --------------------------------------------------------
    # Identify latitude / longitude columns
    # --------------------------------------------------------

    latitude_col = find_column(
        df,
        [
            "latitude",
            "lat",
            "event_latitude",
        ],
    )

    longitude_col = find_column(
        df,
        [
            "longitude",
            "lon",
            "lng",
            "event_longitude",
        ],
    )


    if latitude_col is None:

        raise RuntimeError(
            "Could not locate latitude column."
        )


    if longitude_col is None:

        raise RuntimeError(
            "Could not locate longitude column."
        )


    print(
        "\nLatitude column:",
        latitude_col
    )

    print(
        "Longitude column:",
        longitude_col
    )


    # --------------------------------------------------------
    # Convert coordinates to numbers
    # --------------------------------------------------------

    df[latitude_col] = pd.to_numeric(
        df[latitude_col],
        errors="coerce",
    )

    df[longitude_col] = pd.to_numeric(
        df[longitude_col],
        errors="coerce",
    )


    df = df.dropna(
        subset=[
            latitude_col,
            longitude_col,
        ]
    )


    # --------------------------------------------------------
    # Geographic filter
    # --------------------------------------------------------

    ner = df[
        df[latitude_col].between(
            LAT_MIN,
            LAT_MAX,
            inclusive="both",
        )
        &
        df[longitude_col].between(
            LON_MIN,
            LON_MAX,
            inclusive="both",
        )
    ].copy()


    # Standard names for our future pipeline.
    ner = ner.rename(
        columns={
            latitude_col: "latitude",
            longitude_col: "longitude",
        }
    )


    # --------------------------------------------------------
    # India filter if country code exists
    # --------------------------------------------------------

    country_code_col = find_column(
        ner,
        [
            "country_code",
            "country code",
        ],
    )


    country_name_col = find_column(
        ner,
        [
            "country_name",
            "country",
        ],
    )


    if country_code_col:

        india_mask = (
            ner[country_code_col]
            .astype(str)
            .str.upper()
            .isin(
                [
                    "IN",
                    "IND",
                    "INDIA",
                ]
            )
        )

        # Only apply it if it actually matches rows.
        if india_mask.any():

            ner = ner[
                india_mask
            ].copy()


    elif country_name_col:

        india_mask = (
            ner[country_name_col]
            .astype(str)
            .str.lower()
            .str.contains(
                "india",
                na=False,
            )
        )

        if india_mask.any():

            ner = ner[
                india_mask
            ].copy()


    # --------------------------------------------------------
    # Remove duplicate coordinates/events
    # --------------------------------------------------------

    event_id_col = find_column(
        ner,
        [
            "event_id",
            "event id",
            "id",
        ],
    )


    if event_id_col:

        ner = ner.drop_duplicates(
            subset=[event_id_col]
        )

    else:

        ner = ner.drop_duplicates()


    # --------------------------------------------------------
    # Save filtered CSV
    # --------------------------------------------------------

    ner.to_csv(
        NER_CSV_PATH,
        index=False,
        encoding="utf-8",
    )


    # --------------------------------------------------------
    # GeoJSON
    #
    # We create GeoJSON ourselves so that geopandas is
    # not required yet.
    # --------------------------------------------------------

    features = []

    for _, row in ner.iterrows():

        properties = {}

        for column in ner.columns:

            value = row[column]

            if pd.isna(value):

                properties[column] = None

            elif isinstance(
                value,
                (
                    int,
                    float,
                    str,
                    bool,
                ),
            ):

                properties[column] = value

            else:

                properties[column] = str(value)


        feature = {
            "type": "Feature",

            "geometry": {
                "type": "Point",

                "coordinates": [
                    float(
                        row["longitude"]
                    ),
                    float(
                        row["latitude"]
                    ),
                ],
            },

            "properties":
            properties,
        }

        features.append(feature)


    import json

    geojson = {
        "type":
        "FeatureCollection",

        "name":
        "NASA_GLC_North_East_India",

        "features":
        features,
    }


    NER_GEOJSON_PATH.write_text(
        json.dumps(
            geojson,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "NASA GLC EXTRACTION COMPLETE"
    )

    print("=" * 70)


    print(
        "\nFull NASA catalog:",
        len(df)
    )


    print(
        "North-East India records:",
        len(ner)
    )


    print("\nFiltered CSV:")

    print(
        NER_CSV_PATH
    )


    print("\nGeoJSON:")

    print(
        NER_GEOJSON_PATH
    )


    # --------------------------------------------------------
    # Event date summary
    # --------------------------------------------------------

    date_col = find_column(
        ner,
        [
            "event_date",
            "event date",
            "date",
        ],
    )


    if date_col:

        dates = pd.to_datetime(
            ner[date_col],
            errors="coerce",
        )

        valid_dates = (
            dates.dropna()
        )

        if not valid_dates.empty:

            print(
                "\nEvent date range:"
            )

            print(
                valid_dates.min(),
                "to",
                valid_dates.max(),
            )


    # --------------------------------------------------------
    # Trigger summary
    # --------------------------------------------------------

    trigger_col = find_column(
        ner,
        [
            "landslide_trigger",
            "trigger",
        ],
    )


    if trigger_col:

        print(
            "\nTop triggers:"
        )

        print(
            ner[trigger_col]
            .fillna("Unknown")
            .value_counts()
            .head(10)
        )


    # --------------------------------------------------------
    # Category summary
    # --------------------------------------------------------

    category_col = find_column(
        ner,
        [
            "landslide_category",
            "category",
        ],
    )


    if category_col:

        print(
            "\nTop landslide categories:"
        )

        print(
            ner[category_col]
            .fillna("Unknown")
            .value_counts()
            .head(10)
        )


    print("\nIMPORTANT:")

    print(
        "NASA Open Data GLC export is a "
        "historical dataset. We will document "
        "its date coverage in the final project."
    )


    print(
        "\nSTAGE 1 STATUS: COMPLETE"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    csv_path = (
        download_nasa_csv()
    )

    process_catalog(
        csv_path
    )


if __name__ == "__main__":

    main()