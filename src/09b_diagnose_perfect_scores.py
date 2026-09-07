from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = ROOT / "data" / "processed" / "training" / "v2_training_matrix.csv"
MANIFEST_FILE = ROOT / "data" / "processed" / "training" / "v2_feature_manifest.json"

OUT_DIR = ROOT / "data" / "processed" / "training" / "stage9_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_AUC_CSV = OUT_DIR / "univariate_numeric_separability.csv"
CLASS_SUMMARY_CSV = OUT_DIR / "feature_summary_by_class.csv"
ABLATION_CSV = OUT_DIR / "ablation_grouped_cv.csv"
PAIR_AUDIT_JSON = OUT_DIR / "pair_spatial_audit.json"
SUMMARY_JSON = OUT_DIR / "diagnostic_summary.json"

RANDOM_STATE = 42
N_SPLITS = 5


def safe_auc(y, x):
    mask = np.isfinite(x)
    if mask.sum() < 10:
        return np.nan
    yy = y[mask]
    xx = x[mask]
    if len(np.unique(yy)) < 2:
        return np.nan
    auc = roc_auc_score(yy, xx)
    return float(max(auc, 1.0 - auc))


def numeric_univariate_report(df, numeric_features):
    rows = []
    y = df["label"].astype(int).to_numpy()

    for feature in numeric_features:
        x = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
        sep_auc = safe_auc(y, x)

        pos = x[y == 1]
        neg = x[y == 0]

        rows.append(
            {
                "feature": feature,
                "separability_auc": sep_auc,
                "positive_nonmissing": int(np.isfinite(pos).sum()),
                "background_nonmissing": int(np.isfinite(neg).sum()),
                "positive_mean": float(np.nanmean(pos)),
                "background_mean": float(np.nanmean(neg)),
                "positive_median": float(np.nanmedian(pos)),
                "background_median": float(np.nanmedian(neg)),
                "positive_min": float(np.nanmin(pos)),
                "positive_max": float(np.nanmax(pos)),
                "background_min": float(np.nanmin(neg)),
                "background_max": float(np.nanmax(neg)),
            }
        )

    out = pd.DataFrame(rows).sort_values(
        "separability_auc",
        ascending=False,
        na_position="last",
    )
    out.to_csv(NUMERIC_AUC_CSV, index=False)
    return out


def class_summary(df, features):
    rows = []

    for feature in features:
        for label, name in [(1, "positive"), (0, "background")]:
            s = df.loc[df["label"] == label, feature]

            row = {
                "feature": feature,
                "class": name,
                "rows": int(len(s)),
                "missing": int(s.isna().sum()),
                "missing_pct": float(100.0 * s.isna().mean()),
            }

            numeric = pd.to_numeric(s, errors="coerce")
            if numeric.notna().sum() >= max(5, int(0.5 * s.notna().sum())):
                row.update(
                    {
                        "mean": float(numeric.mean()),
                        "median": float(numeric.median()),
                        "q25": float(numeric.quantile(0.25)),
                        "q75": float(numeric.quantile(0.75)),
                        "min": float(numeric.min()),
                        "max": float(numeric.max()),
                    }
                )
            else:
                vc = s.astype("string").value_counts(dropna=False).head(10)
                row["top_values"] = json.dumps(
                    {str(k): int(v) for k, v in vc.items()},
                    ensure_ascii=False,
                )

            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(CLASS_SUMMARY_CSV, index=False)
    return out


def make_pipeline(features, categorical_features):
    numeric = [f for f in features if f not in categorical_features]
    categorical = [f for f in features if f in categorical_features]

    transformers = []

    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )

    pre = ColumnTransformer(transformers, remainder="drop")

    rf = RandomForestClassifier(
        n_estimators=350,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        [
            ("preprocessor", pre),
            ("model", rf),
        ]
    )


def grouped_cv(df, features, categorical_features, scheme):
    X = df[features]
    y = df["label"].astype(int).to_numpy()

    if scheme == "spatial":
        groups = df["spatial_block_0p5deg"].astype(str).to_numpy()
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        splits = splitter.split(X, y, groups)
    elif scheme == "pair":
        groups = df["pair_id"].astype(str).to_numpy()
        splitter = GroupKFold(n_splits=N_SPLITS)
        splits = splitter.split(X, y, groups)
    else:
        raise ValueError(scheme)

    oof = np.full(len(df), np.nan)

    for train_idx, val_idx in splits:
        model = make_pipeline(features, categorical_features)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y, oof)),
        "pr_auc": float(average_precision_score(y, oof)),
    }


def pair_spatial_audit(df):
    pair_counts = df.groupby("pair_id").size()
    malformed_pairs = int((pair_counts != 2).sum())

    pair_blocks = df.groupby("pair_id")["spatial_block_0p5deg"].nunique()
    same_block = int((pair_blocks == 1).sum())
    different_block = int((pair_blocks > 1).sum())

    # Reproduce the spatial CV assignment and see how often members of the
    # same matched pair land in different validation folds.
    X_dummy = np.zeros((len(df), 1))
    y = df["label"].astype(int).to_numpy()
    groups = df["spatial_block_0p5deg"].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_assignment = np.full(len(df), -1, dtype=int)

    for fold, (_, val_idx) in enumerate(
        splitter.split(X_dummy, y, groups),
        start=1,
    ):
        fold_assignment[val_idx] = fold

    tmp = df[["pair_id"]].copy()
    tmp["fold"] = fold_assignment

    pair_folds = tmp.groupby("pair_id")["fold"].nunique()
    split_across_folds = int((pair_folds > 1).sum())
    same_fold = int((pair_folds == 1).sum())

    return {
        "pairs_total": int(pair_counts.size),
        "malformed_pairs_not_size_2": malformed_pairs,
        "pairs_same_0p5deg_block": same_block,
        "pairs_different_0p5deg_blocks": different_block,
        "pairs_same_spatial_cv_fold": same_fold,
        "pairs_split_across_spatial_cv_folds": split_across_folds,
    }


def main():
    print("=" * 90)
    print("STAGE 9B — DIAGNOSE PERFECT V2 CROSS-VALIDATION SCORES")
    print("=" * 90)

    df = pd.read_csv(TRAIN_FILE)
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))

    features = manifest["model_features"]
    numeric_features = manifest["numeric_features"]
    categorical_features = manifest["categorical_features"]

    print(f"Rows: {len(df)}")
    print(f"Features: {len(features)}")
    print()

    uni = numeric_univariate_report(df, numeric_features)
    class_summary(df, features)

    print("Top univariate numeric separators:")
    print(
        uni[
            [
                "feature",
                "separability_auc",
                "positive_median",
                "background_median",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    pair_audit = pair_spatial_audit(df)
    PAIR_AUDIT_JSON.write_text(
        json.dumps(pair_audit, indent=2),
        encoding="utf-8",
    )

    print("\nMatched-pair / spatial audit:")
    for k, v in pair_audit.items():
        print(f"  {k}: {v}")

    terrain = [
        "elevation_m",
        "slope_deg",
        "aspect_sin",
        "aspect_cos",
        "curvature_1_per_m",
        "land_cover",
    ]

    rainfall = [
        "rainfall_24h_mm",
        "rainfall_72h_mm",
        "rainfall_7d_mm",
    ]

    soil = [
        "soil_moisture_surface_m3_m3",
        "soil_moisture_7_28cm_m3_m3",
        "soil_moisture_28_100cm_m3_m3",
        "soil_moisture_0_100cm_m3_m3",
        "soil_moisture_surface_3d_mean_m3_m3",
        "soil_moisture_surface_7d_mean_m3_m3",
        "soil_moisture_surface_anomaly_vs_7d",
    ]

    vegetation = [
        "ndvi_model_value",
        "ndvi_missing",
    ]

    infrastructure = [
        "distance_to_road_m",
        "distance_to_river_m",
    ]

    feature_sets = {
        "all_features": features,
        "without_roads_rivers": [
            f for f in features if f not in infrastructure
        ],
        "terrain_landcover_only": terrain,
        "rainfall_only": rainfall,
        "soil_only": soil,
        "roads_rivers_only": infrastructure,
        "rainfall_plus_soil": rainfall + soil,
        "terrain_plus_vegetation": terrain + vegetation,
    }

    rows = []

    print("\nGrouped CV ablations (Random Forest diagnostic only):")

    for set_name, feature_set in feature_sets.items():
        for scheme in ["spatial", "pair"]:
            result = grouped_cv(
                df,
                feature_set,
                categorical_features,
                scheme,
            )

            rows.append(
                {
                    "feature_set": set_name,
                    "validation_scheme": scheme,
                    "n_features": len(feature_set),
                    **result,
                }
            )

            print(
                f"  {set_name:25s} | {scheme:7s} | "
                f"ROC-AUC={result['roc_auc']:.4f} | "
                f"PR-AUC={result['pr_auc']:.4f}"
            )

    ablation = pd.DataFrame(rows)
    ablation.to_csv(ABLATION_CSV, index=False)

    suspicious = uni[
        uni["separability_auc"] >= 0.95
    ]["feature"].tolist()

    summary = {
        "perfect_stage9_scores_are_suspicious": True,
        "numeric_features_with_univariate_separability_auc_ge_0p95": suspicious,
        "pair_spatial_audit": pair_audit,
        "outputs": {
            "univariate_numeric_separability": str(NUMERIC_AUC_CSV),
            "feature_summary_by_class": str(CLASS_SUMMARY_CSV),
            "ablation_grouped_cv": str(ABLATION_CSV),
            "pair_spatial_audit": str(PAIR_AUDIT_JSON),
        },
        "interpretation_note": (
            "A perfect score across XGBoost, LightGBM and Random Forest in every "
            "geographic fold is a data/evaluation red flag. Do not present the "
            "1.0 metrics as operational performance until the source of the "
            "separability is understood."
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print("STAGE 9B COMPLETE")
    print("=" * 90)
    print(f"Univariate report:\n{NUMERIC_AUC_CSV}")
    print(f"\nAblation report:\n{ABLATION_CSV}")
    print(f"\nPair audit:\n{PAIR_AUDIT_JSON}")
    print(f"\nDiagnostic summary:\n{SUMMARY_JSON}")


if __name__ == "__main__":
    main()
