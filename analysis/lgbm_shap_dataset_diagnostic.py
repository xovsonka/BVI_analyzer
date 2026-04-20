from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import NUMERIC_FEATURES, ensure_features


LABELS = ["legit", "phishing", "spam", "financial_fraud"]

EXTRA_NUMERIC_FEATURES = [
    "heuristic_score",
    "external_domain_url_count",
    "sender_url_domain_mismatch_ratio",
    "credential_keyword_density",
    "urgent_keyword_density",
    "unicode_mixed_script_flag",
    "punycode_url_count",
    "obfuscated_url_count",
    "mismatched_anchor_count",
    "message_id_domain_mismatch",
    "display_name_spoof_flag",
    "received_anomaly_flag",
    "attachment_count_total",
]

SUSPICIOUS_FEATURE_CANDIDATES = [
    "mailhog_id",
    "message_id",
    "file",
    "file_key",
    "text_hash",
    "campaign_name",
    "dataset_id_hint",
    "benchmark_group",
    "benchmark_source",
    "split",
    "source",
    "source_profile",
    "source_subgroup_id",
    "row_id",
    "cluster_id",
    "template_norm",
    "exact_template_id",
    "near_signature",
    "is_synthetic",
    "synthetic_method",
    "synthetic_parent_row_id",
]

SUSPICIOUS_RE = re.compile(
    r"(id$|_id_|^id_|hash|file|source|campaign|dataset|template|signature|split|synthetic)",
    re.IGNORECASE,
)


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def build_feature_sets(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    clean = [c for c in NUMERIC_FEATURES + EXTRA_NUMERIC_FEATURES if c in df.columns]
    suspicious = [c for c in SUSPICIOUS_FEATURE_CANDIDATES if c in df.columns]

    leaky = list(clean)
    for c in suspicious:
        if c not in leaky:
            leaky.append(c)

    if not clean:
        raise RuntimeError("No clean features found in input CSV")
    return clean, leaky, suspicious


def prepare_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = df[features].copy()
    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            x[col] = _as_numeric(x[col])
        else:
            x[col] = x[col].astype("string").fillna("missing").astype("category")
    return x


def _mean_abs_shap_per_feature(
    shap_values: object,
    feature_count: int,
) -> np.ndarray:
    if isinstance(shap_values, list):
        per_class = [np.abs(v).mean(axis=0) for v in shap_values]
        return np.mean(np.vstack(per_class), axis=0)

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # (rows, features, classes)
        return np.mean(np.abs(arr), axis=(0, 2))
    if arr.ndim == 2 and arr.shape[1] == feature_count:
        return np.mean(np.abs(arr), axis=0)
    raise RuntimeError(f"Unsupported SHAP output shape: {arr.shape}")


def train_eval_and_shap(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(np.unique(y_train)),
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    pred_proba = model.predict_proba(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
    }

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    importance = _mean_abs_shap_per_feature(
        shap_values, feature_count=len(feature_names)
    )
    shap_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": importance,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    return metrics, shap_df, pred, pred_proba


def plot_top_shap(
    shap_df: pd.DataFrame, out_path: Path, title: str, top_n: int
) -> None:
    top = shap_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2f5f98")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LightGBM + SHAP diagnostic pass for dataset quality issues"
    )
    parser.add_argument(
        "--input-csv",
        default="results/experiment_5/exp5_benchmark_with_predictions.csv",
    )
    parser.add_argument(
        "--label-col",
        default="label",
    )
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--high-confidence", type=float, default=0.9)
    parser.add_argument("--out-dir", default="results/diagnostics_lgbm_shap")
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {in_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, low_memory=False)
    if args.label_col not in df.columns:
        raise RuntimeError(f"Label column '{args.label_col}' not found")

    df = df[df[args.label_col].isin(LABELS)].copy()
    if df.empty:
        raise RuntimeError("No labeled rows found after filtering expected classes")

    model_ready = ensure_features(df.copy())
    for col in model_ready.columns:
        if col not in df.columns:
            df[col] = model_ready[col]

    clean_features, leaky_features, suspicious_features = build_feature_sets(df)

    le = LabelEncoder()
    y = le.fit_transform(df[args.label_col].astype(str))

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    y_train = y[train_idx]
    y_test = y[test_idx]

    x_train_clean = prepare_matrix(train_df, clean_features)
    x_test_clean = prepare_matrix(test_df, clean_features)
    clean_metrics, shap_clean, pred_clean, proba_clean = train_eval_and_shap(
        x_train_clean,
        x_test_clean,
        y_train,
        y_test,
        clean_features,
    )

    x_train_leaky = prepare_matrix(train_df, leaky_features)
    x_test_leaky = prepare_matrix(test_df, leaky_features)
    leaky_metrics, shap_leaky, pred_leaky, _ = train_eval_and_shap(
        x_train_leaky,
        x_test_leaky,
        y_train,
        y_test,
        leaky_features,
    )

    suspicious_top = shap_leaky.head(args.top_n).copy()
    suspicious_top["is_suspicious_name"] = suspicious_top["feature"].map(
        lambda x: bool(SUSPICIOUS_RE.search(str(x)))
    )
    suspicious_top["is_known_suspicious_feature"] = suspicious_top["feature"].isin(
        suspicious_features
    )

    leak_importance = float(
        suspicious_top.loc[
            suspicious_top["is_suspicious_name"]
            | suspicious_top["is_known_suspicious_feature"],
            "mean_abs_shap",
        ].sum()
    )
    top_importance = float(suspicious_top["mean_abs_shap"].sum())
    leak_pressure_ratio = (
        (leak_importance / top_importance) if top_importance > 0 else 0.0
    )

    test_out = test_df.copy()
    test_out["y_true"] = le.inverse_transform(y_test)
    test_out["pred_clean"] = le.inverse_transform(pred_clean)
    test_out["pred_leaky"] = le.inverse_transform(pred_leaky)
    test_out["proba_clean_max"] = np.max(proba_clean, axis=1)
    test_out["is_clean_error"] = test_out["pred_clean"] != test_out["y_true"]

    high_conf_errors = test_out[
        test_out["is_clean_error"]
        & (test_out["proba_clean_max"] >= args.high_confidence)
    ].copy()

    id_cols = [
        c
        for c in [
            "file",
            "file_key",
            "message_id",
            "mailhog_id",
            "campaign_name",
            "subject",
        ]
        if c in high_conf_errors.columns
    ]
    high_conf_cols = id_cols + [
        "y_true",
        "pred_clean",
        "proba_clean_max",
    ]
    high_conf_errors = high_conf_errors[high_conf_cols].sort_values(
        "proba_clean_max", ascending=False
    )

    metrics_df = pd.DataFrame(
        [
            {
                "setup": "clean_features",
                **clean_metrics,
                "feature_count": len(clean_features),
            },
            {
                "setup": "clean_plus_suspicious",
                **leaky_metrics,
                "feature_count": len(leaky_features),
            },
        ]
    )
    metrics_df["delta_vs_clean"] = metrics_df["f1_macro"] - float(
        clean_metrics["f1_macro"]
    )

    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)
    shap_clean.to_csv(out_dir / "shap_top_clean.csv", index=False)
    shap_leaky.to_csv(out_dir / "shap_top_leaky.csv", index=False)
    suspicious_top.to_csv(out_dir / "leakage_red_flags.csv", index=False)
    high_conf_errors.to_csv(out_dir / "high_confidence_mislabels.csv", index=False)

    plot_top_shap(
        shap_clean,
        out_dir / "shap_top_clean.png",
        "Top SHAP features (clean feature set)",
        top_n=args.top_n,
    )
    plot_top_shap(
        shap_leaky,
        out_dir / "shap_top_leaky.png",
        "Top SHAP features (clean + suspicious feature set)",
        top_n=args.top_n,
    )

    summary = {
        "input_csv": str(in_path),
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "label_distribution": df[args.label_col].value_counts().to_dict(),
        "clean_feature_count": int(len(clean_features)),
        "suspicious_feature_count": int(len(suspicious_features)),
        "clean_metrics": clean_metrics,
        "leaky_metrics": leaky_metrics,
        "f1_macro_gain_with_suspicious_features": float(
            leaky_metrics["f1_macro"] - clean_metrics["f1_macro"]
        ),
        "leak_pressure_ratio_top_n": float(leak_pressure_ratio),
        "high_confidence_error_count": int(len(high_conf_errors)),
    }
    (out_dir / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Saved diagnostics to: {out_dir}")
    print(f"Clean metrics: {clean_metrics}")
    print(f"Leaky metrics: {leaky_metrics}")
    print(f"Leak pressure ratio (top {args.top_n}): {leak_pressure_ratio:.4f}")
    print(f"High-confidence error candidates: {len(high_conf_errors)}")


if __name__ == "__main__":
    main()
