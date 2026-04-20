from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import NUMERIC_FEATURES, ensure_features


CANDIDATE_FEATURES = [
    "display_name_spoof_flag",
    "message_id_domain_mismatch",
    "external_domain_url_count",
    "sender_url_domain_mismatch_ratio",
    "obfuscated_url_count",
    "mismatched_anchor_count",
    "punycode_url_count",
]


def _split_path(shared_dir: Path, split_name: str) -> Path:
    mapping = {
        "val_iid": [shared_dir / "val_iid.csv", shared_dir / "val.csv"],
        "test_iid": [shared_dir / "test_iid.csv", shared_dir / "test.csv"],
        "test_hard_source": [shared_dir / "test_hard_source.csv"],
        "test_hard_cluster": [shared_dir / "test_hard_cluster.csv"],
        "test_deployment": [shared_dir / "test_deployment.csv"],
    }
    candidates = mapping.get(split_name)
    if not candidates:
        raise ValueError(f"Unsupported split name: {split_name}")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing split file for {split_name}: {candidates}")


def _mean_abs_shap_per_feature(shap_values: object, feature_count: int) -> np.ndarray:
    if isinstance(shap_values, list):
        per_class = [np.abs(v).mean(axis=0) for v in shap_values]
        return np.mean(np.vstack(per_class), axis=0)

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return np.mean(np.abs(arr), axis=(0, 2))
    if arr.ndim == 2 and arr.shape[1] == feature_count:
        return np.mean(np.abs(arr), axis=0)
    raise RuntimeError(f"Unsupported SHAP output shape: {arr.shape}")


def _plot_top_shap(shap_df: pd.DataFrame, out_path: Path, title: str, top_n: int) -> None:
    top = shap_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2f5f98")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _train_model(train_df: pd.DataFrame, features: list[str]) -> tuple[lgb.LGBMClassifier, LabelEncoder]:
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"].astype(str))
    x_train = train_df[features].copy()

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(le.classes_),
        n_estimators=320,
        learning_rate=0.05,
        num_leaves=63,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    return model, le


def _evaluate_split(
    model: lgb.LGBMClassifier,
    label_encoder: LabelEncoder,
    split_df: pd.DataFrame,
    features: list[str],
    max_shap_rows: int,
) -> tuple[dict, pd.DataFrame]:
    x_eval = split_df[features].copy()
    y_true = label_encoder.transform(split_df["label"].astype(str))

    pred = model.predict(x_eval)
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "rows": int(len(split_df)),
    }

    if len(split_df) > max_shap_rows:
        shap_df_source = split_df.sample(n=max_shap_rows, random_state=42)
    else:
        shap_df_source = split_df

    x_shap = shap_df_source[features].copy()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_shap)
    importance = _mean_abs_shap_per_feature(shap_values, feature_count=len(features))
    shap_df = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap": importance,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    return metrics, shap_df


def _load_data(processed_dir: Path, scenario: str, split_names: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    scenario_dir = processed_dir / scenario
    train_path = scenario_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing scenario train split: {train_path}")

    train_df = ensure_features(pd.read_csv(train_path, low_memory=False))
    shared_dir = processed_dir / "shared"
    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in split_names:
        split_path = _split_path(shared_dir, split_name)
        split_frames[split_name] = ensure_features(pd.read_csv(split_path, low_memory=False))
    return train_df, split_frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LightGBM + SHAP on shared splits with baseline vs extended feature set"
    )
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument("--scenario", default="scenario_m_anti_template_feature_regularized")
    parser.add_argument(
        "--splits",
        default="test_iid,test_hard_source,test_hard_cluster",
        help="Comma-separated shared split names.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-shap-rows", type=int, default=6000)
    parser.add_argument(
        "--out-dir",
        default="results/diagnostics_lgbm_shap_split",
    )
    args = parser.parse_args()

    split_names = [part.strip() for part in args.splits.split(",") if part.strip()]
    if not split_names:
        raise ValueError("At least one split must be selected")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, split_frames = _load_data(Path(args.processed_dir), args.scenario, split_names)

    present_features = [feature for feature in NUMERIC_FEATURES if feature in train_df.columns]
    baseline_features = [feature for feature in present_features if feature not in CANDIDATE_FEATURES]
    extended_features = list(present_features)

    if not baseline_features:
        raise RuntimeError("No baseline features available")
    if not extended_features:
        raise RuntimeError("No features available for training")

    setups = {
        "baseline_without_candidates": baseline_features,
        "extended_with_candidates": extended_features,
    }

    metrics_rows: list[dict] = []
    summary_json: dict[str, dict] = {
        "scenario": args.scenario,
        "splits": split_names,
        "candidate_features": CANDIDATE_FEATURES,
        "feature_sets": {
            "baseline_without_candidates": baseline_features,
            "extended_with_candidates": extended_features,
        },
        "results": {},
    }

    for setup_name, features in setups.items():
        model, label_encoder = _train_model(train_df, features)
        summary_json["results"][setup_name] = {}

        for split_name, split_df in split_frames.items():
            metrics, shap_df = _evaluate_split(
                model=model,
                label_encoder=label_encoder,
                split_df=split_df,
                features=features,
                max_shap_rows=args.max_shap_rows,
            )

            metrics_rows.append(
                {
                    "setup": setup_name,
                    "split": split_name,
                    "feature_count": len(features),
                    **metrics,
                }
            )

            shap_csv = out_dir / f"shap_{setup_name}__{split_name}.csv"
            shap_df.to_csv(shap_csv, index=False)
            _plot_top_shap(
                shap_df,
                out_dir / f"shap_{setup_name}__{split_name}.png",
                title=f"{setup_name} | {split_name}",
                top_n=args.top_n,
            )

            summary_json["results"][setup_name][split_name] = {
                **metrics,
                "top_shap": shap_df.head(args.top_n).to_dict(orient="records"),
            }

            print(
                f"{setup_name} :: {split_name} :: f1_macro={metrics['f1_macro']:.4f} "
                f"balanced_acc={metrics['balanced_accuracy']:.4f}"
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)

    delta_rows = []
    for split_name in split_names:
        base_row = metrics_df[
            (metrics_df["setup"] == "baseline_without_candidates") & (metrics_df["split"] == split_name)
        ].iloc[0]
        ext_row = metrics_df[
            (metrics_df["setup"] == "extended_with_candidates") & (metrics_df["split"] == split_name)
        ].iloc[0]
        delta_rows.append(
            {
                "split": split_name,
                "delta_f1_macro": float(ext_row["f1_macro"] - base_row["f1_macro"]),
                "delta_balanced_accuracy": float(ext_row["balanced_accuracy"] - base_row["balanced_accuracy"]),
                "delta_accuracy": float(ext_row["accuracy"] - base_row["accuracy"]),
            }
        )

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "metrics_delta_extended_vs_baseline.csv", index=False)
    summary_json["delta_extended_vs_baseline"] = delta_rows

    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(f"Saved SHAP split diagnostics to: {out_dir}")


if __name__ == "__main__":
    main()
