from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit(
        "xgboost is not installed in this venv. Run: pip install xgboost"
    ) from exc

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:
    raise SystemExit(
        "lightgbm is not installed in this venv. Run: pip install lightgbm"
    ) from exc


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    ROOT / "data" / "processed" / "training" / "v2_training_matrix.csv"
)
MANIFEST_FILE = (
    ROOT / "data" / "processed" / "training" / "v2_feature_manifest.json"
)

OUT_DIR = ROOT / "models" / "v2"
REPORT_DIR = ROOT / "data" / "processed" / "training" / "stage9_results"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_METRICS_CSV = REPORT_DIR / "geographic_cv_fold_metrics.csv"
OOF_CSV = REPORT_DIR / "oof_predictions.csv"
SUMMARY_JSON = REPORT_DIR / "model_validation_summary.json"
IMPORTANCE_CSV = REPORT_DIR / "feature_importance.csv"
FINAL_BUNDLE = OUT_DIR / "v2_ensemble_bundle.joblib"

RANDOM_STATE = 42
N_SPLITS = 5


def make_preprocessor(numeric_features, categorical_features):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_models(numeric_features, categorical_features):
    def pipe(model):
        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_preprocessor(
                        numeric_features,
                        categorical_features,
                    ),
                ),
                ("model", model),
            ]
        )

    models = {
        "xgboost": pipe(
            XGBClassifier(
                n_estimators=450,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=2,
                reg_lambda=1.5,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        "lightgbm": pipe(
            LGBMClassifier(
                n_estimators=450,
                learning_rate=0.03,
                num_leaves=24,
                max_depth=-1,
                min_child_samples=18,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            )
        ),
        "random_forest": pipe(
            RandomForestClassifier(
                n_estimators=600,
                max_depth=12,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
    }

    return models


def safe_metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(
            precision_score(y_true, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, pred, zero_division=0)
        ),
    }


def find_best_f1_threshold(y_true, prob):
    thresholds = np.linspace(0.05, 0.95, 181)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        pred = (prob >= threshold).astype(int)
        score = f1_score(
            y_true,
            pred,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, float(best_f1)


def metric_mean_std(fold_df, model_name):
    part = fold_df[fold_df["model"] == model_name]

    result = {}
    for metric in ["roc_auc", "pr_auc", "f1", "precision", "recall"]:
        result[f"{metric}_mean"] = float(part[metric].mean())
        result[f"{metric}_std"] = float(part[metric].std(ddof=1))

    return result


def export_feature_importance(
    fitted_models,
    numeric_features,
    categorical_features,
):
    rows = []

    for model_name, pipeline in fitted_models.items():
        preprocessor = pipeline.named_steps["preprocessor"]
        estimator = pipeline.named_steps["model"]

        names = preprocessor.get_feature_names_out()

        if not hasattr(estimator, "feature_importances_"):
            continue

        values = np.asarray(
            estimator.feature_importances_,
            dtype=float,
        )

        if len(names) != len(values):
            continue

        total = float(values.sum())
        normalized = (
            values / total
            if total > 0
            else np.zeros_like(values)
        )

        for feature_name, importance in zip(names, normalized):
            clean_name = str(feature_name)
            clean_name = clean_name.replace("num__", "")
            clean_name = clean_name.replace("cat__", "")

            rows.append(
                {
                    "model": model_name,
                    "transformed_feature": clean_name,
                    "importance_normalized": float(importance),
                }
            )

    importance_df = pd.DataFrame(rows)

    if not importance_df.empty:
        importance_df = importance_df.sort_values(
            ["model", "importance_normalized"],
            ascending=[True, False],
        )

    importance_df.to_csv(
        IMPORTANCE_CSV,
        index=False,
    )


def main():
    print("=" * 88)
    print("STAGE 9 — V2 MODEL TRAINING + GEOGRAPHIC CROSS-VALIDATION")
    print("=" * 88)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(TRAIN_FILE)

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(MANIFEST_FILE)

    df = pd.read_csv(TRAIN_FILE)

    manifest = json.loads(
        MANIFEST_FILE.read_text(encoding="utf-8")
    )

    features = manifest["model_features"]
    numeric_features = manifest["numeric_features"]
    categorical_features = manifest["categorical_features"]

    required = (
        features
        + [
            "label",
            "sample_id",
            "pair_id",
            "state",
            "latitude",
            "longitude",
            "event_date",
            "spatial_block_0p5deg",
        ]
    )

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"Training matrix is missing columns: {missing}"
        )

    if len(df) != 762:
        print(
            f"WARNING: expected 762 rows from Stage 8, "
            f"found {len(df)}."
        )

    y = df["label"].astype(int).to_numpy()
    X = df[features].copy()
    groups = df["spatial_block_0p5deg"].astype(str).to_numpy()

    print(f"Rows: {len(df)}")
    print(
        f"Positive: {(y == 1).sum()} | "
        f"Background: {(y == 0).sum()}"
    )
    print(f"Model features: {len(features)}")
    print(
        f"Geographic groups (0.5 degree): "
        f"{pd.Series(groups).nunique()}"
    )
    print(f"Cross-validation folds: {N_SPLITS}")
    print()

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model_names = [
        "xgboost",
        "lightgbm",
        "random_forest",
    ]

    oof = {
        name: np.full(len(df), np.nan, dtype=float)
        for name in model_names
    }

    fold_rows = []

    splits = list(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    for fold, (train_idx, val_idx) in enumerate(
        splits,
        start=1,
    ):
        print("-" * 88)
        print(
            f"FOLD {fold}/{N_SPLITS} | "
            f"train={len(train_idx)} validation={len(val_idx)}"
        )

        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])

        overlap = train_groups.intersection(val_groups)

        if overlap:
            raise RuntimeError(
                f"Geographic group leakage detected in fold {fold}: "
                f"{list(overlap)[:5]}"
            )

        fold_models = make_models(
            numeric_features,
            categorical_features,
        )

        for name in model_names:
            print(f"  training {name}...")

            pipeline = fold_models[name]
            pipeline.fit(
                X.iloc[train_idx],
                y[train_idx],
            )

            prob = pipeline.predict_proba(
                X.iloc[val_idx]
            )[:, 1]

            oof[name][val_idx] = prob

            metrics = safe_metrics(
                y[val_idx],
                prob,
                threshold=0.5,
            )

            fold_rows.append(
                {
                    "fold": fold,
                    "model": name,
                    "train_rows": int(len(train_idx)),
                    "validation_rows": int(len(val_idx)),
                    "train_groups": int(len(train_groups)),
                    "validation_groups": int(len(val_groups)),
                    **metrics,
                }
            )

            print(
                f"    ROC-AUC={metrics['roc_auc']:.4f} | "
                f"PR-AUC={metrics['pr_auc']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"Recall={metrics['recall']:.4f}"
            )

    for name in model_names:
        if np.isnan(oof[name]).any():
            raise RuntimeError(
                f"OOF predictions incomplete for {name}."
            )

    ensemble_prob = (
        oof["xgboost"]
        + oof["lightgbm"]
        + oof["random_forest"]
    ) / 3.0

    # Fold-level ensemble metrics at the neutral 0.50 threshold.
    for fold, (_, val_idx) in enumerate(
        splits,
        start=1,
    ):
        metrics = safe_metrics(
            y[val_idx],
            ensemble_prob[val_idx],
            threshold=0.5,
        )

        fold_rows.append(
            {
                "fold": fold,
                "model": "ensemble_equal_weight",
                "train_rows": int(len(df) - len(val_idx)),
                "validation_rows": int(len(val_idx)),
                "train_groups": np.nan,
                "validation_groups": int(
                    pd.Series(groups[val_idx]).nunique()
                ),
                **metrics,
            }
        )

    fold_df = pd.DataFrame(fold_rows)

    fold_df.to_csv(
        FOLD_METRICS_CSV,
        index=False,
    )

    oof_df = df[
        [
            "sample_id",
            "pair_id",
            "label",
            "sample_type",
            "state",
            "latitude",
            "longitude",
            "event_date",
            "spatial_block_0p5deg",
        ]
    ].copy()

    for name in model_names:
        oof_df[f"prob_{name}"] = oof[name]

    oof_df["prob_ensemble_equal_weight"] = ensemble_prob

    threshold_f1, threshold_f1_score = find_best_f1_threshold(
        y,
        ensemble_prob,
    )

    oof_df["pred_ensemble_0p5"] = (
        ensemble_prob >= 0.5
    ).astype(int)

    oof_df["pred_ensemble_oof_f1_threshold"] = (
        ensemble_prob >= threshold_f1
    ).astype(int)

    oof_df.to_csv(
        OOF_CSV,
        index=False,
    )

    pooled_results = {}

    for name in model_names:
        pooled_results[name] = safe_metrics(
            y,
            oof[name],
            threshold=0.5,
        )

    pooled_results["ensemble_equal_weight_0p5"] = safe_metrics(
        y,
        ensemble_prob,
        threshold=0.5,
    )

    pooled_results[
        "ensemble_equal_weight_oof_f1_threshold"
    ] = {
        "threshold": threshold_f1,
        **safe_metrics(
            y,
            ensemble_prob,
            threshold=threshold_f1,
        ),
    }

    mean_std = {
        model_name: metric_mean_std(
            fold_df,
            model_name,
        )
        for model_name in [
            "xgboost",
            "lightgbm",
            "random_forest",
            "ensemble_equal_weight",
        ]
    }

    print("\n" + "=" * 88)
    print("POOLED OUT-OF-FOLD RESULTS")
    print("=" * 88)

    for name in model_names:
        m = pooled_results[name]
        print(
            f"{name:15s} "
            f"ROC-AUC={m['roc_auc']:.4f} | "
            f"PR-AUC={m['pr_auc']:.4f} | "
            f"F1={m['f1']:.4f} | "
            f"Precision={m['precision']:.4f} | "
            f"Recall={m['recall']:.4f}"
        )

    m = pooled_results["ensemble_equal_weight_0p5"]

    print(
        f"{'ensemble':15s} "
        f"ROC-AUC={m['roc_auc']:.4f} | "
        f"PR-AUC={m['pr_auc']:.4f} | "
        f"F1={m['f1']:.4f} | "
        f"Precision={m['precision']:.4f} | "
        f"Recall={m['recall']:.4f}"
    )

    tuned = pooled_results[
        "ensemble_equal_weight_oof_f1_threshold"
    ]

    print(
        f"\nOOF-selected F1 threshold: "
        f"{threshold_f1:.3f}"
    )
    print(
        f"At that threshold: "
        f"F1={tuned['f1']:.4f} | "
        f"Precision={tuned['precision']:.4f} | "
        f"Recall={tuned['recall']:.4f}"
    )

    print(
        "\nNOTE: the F1 threshold above was selected from the same "
        "OOF predictions. Treat it as a prototype operating threshold, "
        "not as an independently validated deployment threshold."
    )

    # Fit final full-data models after evaluation.
    print("\n" + "=" * 88)
    print("FITTING FINAL MODELS ON ALL 762 ROWS")
    print("=" * 88)

    final_models = make_models(
        numeric_features,
        categorical_features,
    )

    for name in model_names:
        print(f"Training final {name}...")
        final_models[name].fit(
            X,
            y,
        )

    export_feature_importance(
        final_models,
        numeric_features,
        categorical_features,
    )

    bundle = {
        "model_version": "v2_real_geospatial_pipeline",
        "models": final_models,
        "ensemble_weights": {
            "xgboost": 1 / 3,
            "lightgbm": 1 / 3,
            "random_forest": 1 / 3,
        },
        "default_probability_threshold": 0.5,
        "prototype_oof_f1_threshold": threshold_f1,
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "validation_grouping": "0.5-degree spatial blocks",
        "training_rows": int(len(df)),
        "positive_rows": int((y == 1).sum()),
        "background_rows": int((y == 0).sum()),
        "methodology_note": (
            "Negative examples are matched pseudo-absence/background "
            "points, not confirmed stable locations. Geographic CV "
            "results are prototype validation metrics and must not be "
            "described as real-world operational accuracy."
        ),
    }

    joblib.dump(
        bundle,
        FINAL_BUNDLE,
    )

    summary = {
        "dataset": {
            "rows": int(len(df)),
            "positive": int((y == 1).sum()),
            "background": int((y == 0).sum()),
            "features": len(features),
            "spatial_groups": int(
                pd.Series(groups).nunique()
            ),
            "cv_folds": N_SPLITS,
        },
        "pooled_oof_metrics_at_0p5": {
            name: pooled_results[name]
            for name in model_names
        },
        "ensemble_equal_weight_at_0p5": pooled_results[
            "ensemble_equal_weight_0p5"
        ],
        "ensemble_oof_selected_f1_threshold": tuned,
        "fold_metric_mean_std": mean_std,
        "artifacts": {
            "fold_metrics_csv": str(FOLD_METRICS_CSV),
            "oof_predictions_csv": str(OOF_CSV),
            "feature_importance_csv": str(IMPORTANCE_CSV),
            "final_model_bundle": str(FINAL_BUNDLE),
        },
        "important_caveats": [
            "Background samples are matched pseudo-absences, not confirmed stable locations.",
            "Geographic cross-validation reduces spatial leakage but is still prototype validation.",
            "WorldCover 2021 and current OSM roads/waterways are static susceptibility proxies for historical events.",
            "Historical event-day daily rainfall does not by itself establish a strict 6-24 hour operational lead time.",
            "Do not describe ROC-AUC as accuracy.",
        ],
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("STAGE 9 COMPLETE")
    print("=" * 88)
    print(f"Fold metrics:\n{FOLD_METRICS_CSV}")
    print(f"\nOOF predictions:\n{OOF_CSV}")
    print(f"\nValidation summary:\n{SUMMARY_JSON}")
    print(f"\nFeature importance:\n{IMPORTANCE_CSV}")
    print(f"\nFinal model bundle:\n{FINAL_BUNDLE}")


if __name__ == "__main__":
    main()
