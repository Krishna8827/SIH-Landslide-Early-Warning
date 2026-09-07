from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "processed" / "training" / "v2_training_matrix.csv"

OUT_DIR = ROOT / "data" / "processed" / "training" / "stage9_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIR_CSV = OUT_DIR / "rainfall_pair_audit.csv"
SUMMARY_JSON = OUT_DIR / "rainfall_audit_summary.json"
TREE_TXT = OUT_DIR / "rainfall_shallow_tree_rules.txt"

RAIN = ["rainfall_24h_mm", "rainfall_72h_mm", "rainfall_7d_mm"]


def qstats(s):
    return {
        "count": int(s.notna().sum()),
        "min": float(s.min()),
        "q10": float(s.quantile(0.10)),
        "q25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q75": float(s.quantile(0.75)),
        "q90": float(s.quantile(0.90)),
        "max": float(s.max()),
        "mean": float(s.mean()),
    }


def main():
    print("=" * 88)
    print("STAGE 9C — RAINFALL LEAKAGE / SAMPLING AUDIT")
    print("=" * 88)

    df = pd.read_csv(TRAIN_FILE)

    required = [
        "pair_id", "label", "state", "event_date",
        "latitude", "longitude"
    ] + RAIN

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    # Derived rainfall-shape features.
    df["rain_prev_48h_mm"] = df["rainfall_72h_mm"] - df["rainfall_24h_mm"]
    df["rain_prev_4d_mm"] = df["rainfall_7d_mm"] - df["rainfall_72h_mm"]
    df["ratio_24_to_72"] = df["rainfall_24h_mm"] / df["rainfall_72h_mm"].replace(0, np.nan)
    df["ratio_72_to_7d"] = df["rainfall_72h_mm"] / df["rainfall_7d_mm"].replace(0, np.nan)

    # Verify matched pairs really share date/state.
    pair_meta = df.groupby("pair_id").agg(
        rows=("label", "size"),
        labels=("label", "nunique"),
        states=("state", "nunique"),
        dates=("event_date", "nunique"),
    )

    malformed_pairs = int(
        (
            (pair_meta["rows"] != 2)
            | (pair_meta["labels"] != 2)
            | (pair_meta["states"] != 1)
            | (pair_meta["dates"] != 1)
        ).sum()
    )

    print(f"Matched pairs: {len(pair_meta)}")
    print(f"Malformed date/state/label pairs: {malformed_pairs}")

    # Class summaries.
    class_summary = {}
    print("\nRainfall summary by class:")
    for feature in RAIN + [
        "rain_prev_48h_mm", "rain_prev_4d_mm",
        "ratio_24_to_72", "ratio_72_to_7d"
    ]:
        class_summary[feature] = {}
        print(f"\n{feature}")
        for label, name in [(1, "positive"), (0, "background")]:
            s = pd.to_numeric(
                df.loc[df["label"] == label, feature],
                errors="coerce"
            )
            stats = qstats(s)
            class_summary[feature][name] = stats
            print(
                f"  {name:10s} median={stats['median']:.3f} "
                f"q25={stats['q25']:.3f} q75={stats['q75']:.3f} "
                f"min={stats['min']:.3f} max={stats['max']:.3f}"
            )

    # Pairwise positive-vs-background comparison.
    pos = df[df["label"] == 1].set_index("pair_id")
    neg = df[df["label"] == 0].set_index("pair_id")
    common = pos.index.intersection(neg.index)

    pair_rows = pd.DataFrame(index=common)
    pair_rows["state"] = pos.loc[common, "state"]
    pair_rows["event_date"] = pos.loc[common, "event_date"]

    pair_comparison = {}

    for feature in RAIN + [
        "rain_prev_48h_mm", "rain_prev_4d_mm",
        "ratio_24_to_72", "ratio_72_to_7d"
    ]:
        p = pd.to_numeric(pos.loc[common, feature], errors="coerce")
        n = pd.to_numeric(neg.loc[common, feature], errors="coerce")

        pair_rows[f"positive_{feature}"] = p.values
        pair_rows[f"background_{feature}"] = n.values
        pair_rows[f"delta_{feature}"] = (p - n).values

        valid = p.notna() & n.notna()

        gt = float((p[valid] > n[valid]).mean()) if valid.any() else np.nan
        lt = float((p[valid] < n[valid]).mean()) if valid.any() else np.nan
        eq = float((p[valid] == n[valid]).mean()) if valid.any() else np.nan

        pair_comparison[feature] = {
            "valid_pairs": int(valid.sum()),
            "positive_gt_background_fraction": gt,
            "positive_lt_background_fraction": lt,
            "equal_fraction": eq,
            "median_pair_delta": float((p[valid] - n[valid]).median()) if valid.any() else np.nan,
        }

    pair_rows.reset_index().to_csv(PAIR_CSV, index=False)

    print("\nWithin matched pairs:")
    for feature in RAIN:
        x = pair_comparison[feature]
        print(
            f"  {feature:18s} "
            f"positive>background={x['positive_gt_background_fraction']:.3f} | "
            f"positive<background={x['positive_lt_background_fraction']:.3f} | "
            f"median delta={x['median_pair_delta']:.3f}"
        )

    # Check monotonic accumulation consistency.
    monotonic_bad = int(
        (
            (df["rainfall_72h_mm"] + 1e-9 < df["rainfall_24h_mm"])
            | (df["rainfall_7d_mm"] + 1e-9 < df["rainfall_72h_mm"])
        ).sum()
    )
    print(f"\nRows violating 24h <= 72h <= 7d: {monotonic_bad}/{len(df)}")

    # A shallow tree can reveal a simple rainfall-window fingerprint.
    X = df[RAIN].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df["label"].astype(int).to_numpy()
    groups = df["pair_id"].astype(str).to_numpy()

    splitter = GroupKFold(n_splits=5)
    oof = np.full(len(df), np.nan)

    for train_idx, val_idx in splitter.split(X, y, groups):
        model = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=12,
            random_state=42,
        )
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]

    shallow_auc = float(roc_auc_score(y, oof))
    print(f"Shallow depth-3 tree, pair-grouped ROC-AUC: {shallow_auc:.4f}")

    final_tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=12,
        random_state=42,
    )
    final_tree.fit(X, y)
    rules = export_text(final_tree, feature_names=RAIN)
    TREE_TXT.write_text(rules, encoding="utf-8")

    print("\nShallow rainfall tree rules:")
    print(rules)

    # Exact triplet duplication across opposite classes can tell us whether
    # the model is simply memorising identical/near-identical rainfall tuples.
    trip = df[RAIN].round(3).astype(str).agg("|".join, axis=1)
    tmp = pd.DataFrame({"triplet": trip, "label": y})
    mixed_triplets = int(
        (tmp.groupby("triplet")["label"].nunique() > 1).sum()
    )
    unique_triplets = int(tmp["triplet"].nunique())

    print(f"Unique rounded rainfall triplets: {unique_triplets}/{len(df)}")
    print(f"Rounded triplets appearing in both classes: {mixed_triplets}")

    summary = {
        "rows": int(len(df)),
        "matched_pairs": int(len(pair_meta)),
        "malformed_pairs": malformed_pairs,
        "class_summary": class_summary,
        "pair_comparison": pair_comparison,
        "rows_violating_rainfall_accumulation_monotonicity": monotonic_bad,
        "pair_grouped_depth3_tree_roc_auc": shallow_auc,
        "unique_rounded_rainfall_triplets": unique_triplets,
        "rounded_triplets_shared_by_both_classes": mixed_triplets,
        "interpretation": (
            "If matched pairs share date/state but rainfall-only models remain "
            "near-perfect, the pseudo-absence sampling is likely too easy or "
            "the positive/background rainfall extraction pipelines are not "
            "equivalent. Inspect the class summaries, pair deltas and shallow "
            "tree rules before using rainfall in the final model."
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("STAGE 9C COMPLETE")
    print("=" * 88)
    print(f"Pair audit CSV:\n{PAIR_CSV}")
    print(f"\nShallow-tree rules:\n{TREE_TXT}")
    print(f"\nSummary:\n{SUMMARY_JSON}")


if __name__ == "__main__":
    main()
