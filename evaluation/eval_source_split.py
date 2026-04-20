from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import build_model, ensure_features


MODEL_CHOICES = [
    "hybrid_logreg",
    "hybrid_linear_svc_cal",
    "hybrid_sgd_log",
    "hybrid_sgd_hinge",
]


def find_scenarios(processed_dir: Path) -> list[Path]:
    out = []
    for p in sorted(processed_dir.glob("scenario_*")):
        if (p / "train.csv").exists() and (p / "val.csv").exists():
            out.append(p)
    return out


def evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    benign_label: str,
) -> dict:
    labels = sorted(
        set(train_df["label"].astype(str)).union(set(test_df["label"].astype(str)))
    )
    le = LabelEncoder()
    le.fit(labels)

    y_train = le.transform(train_df["label"].astype(str))
    y_test = le.transform(test_df["label"].astype(str))

    model = build_model(model_name)
    model.fit(train_df, y_train)
    pred = model.predict(test_df)
    pred_labels = le.inverse_transform(pred)
    true_labels = test_df["label"].astype(str).to_numpy()

    unique_test = sorted(test_df["label"].astype(str).unique().tolist())
    single_label_holdout = len(unique_test) == 1

    out = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_test, pred, average="weighted", zero_division=0)
        ),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "test_unique_labels": int(len(unique_test)),
        "single_label_holdout": bool(single_label_holdout),
        "test_only_label": unique_test[0] if single_label_holdout else "",
    }

    recalls = precision_recall_fscore_support(
        y_test,
        pred,
        labels=list(range(len(le.classes_))),
        average=None,
        zero_division=0,
    )[1]
    out["per_class_recall"] = {
        str(le.classes_[i]): float(recalls[i]) for i in range(len(le.classes_))
    }

    y_true_bin = (true_labels != benign_label).astype(int)
    y_pred_bin = (pred_labels != benign_label).astype(int)
    out["binary_f1_suspicious"] = float(
        f1_score(y_true_bin, y_pred_bin, average="binary", zero_division=0)
    )
    out["binary_balanced_accuracy"] = float(
        balanced_accuracy_score(y_true_bin, y_pred_bin)
    )
    out["binary_positive_rate_true"] = float(np.mean(y_true_bin))
    out["binary_positive_rate_pred"] = float(np.mean(y_pred_bin))

    if len(np.unique(y_test)) >= 2:
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_test, pred))
    else:
        out["balanced_accuracy"] = np.nan

    return out


def evaluate_scenario_test_holdouts(
    processed_dir: Path,
    scenarios: list[Path],
    models: list[str],
    min_test_rows: int,
    benign_label: str,
) -> tuple[list[dict], list[dict]]:
    shared_test_path = processed_dir / "shared" / "test.csv"
    if not shared_test_path.exists():
        raise FileNotFoundError(f"Missing shared test set: {shared_test_path}")

    test_df = ensure_features(pd.read_csv(shared_test_path, low_memory=False))
    source_counts = test_df["source"].value_counts().to_dict()
    candidate_sources = [
        source for source, n in source_counts.items() if int(n) >= min_test_rows
    ]

    rows: list[dict] = []
    summary_rows: list[dict] = []

    for scenario in scenarios:
        train_path = scenario / "train.csv"
        train_df = ensure_features(pd.read_csv(train_path, low_memory=False))

        for model_name in models:
            baseline = evaluate(
                train_df, test_df, model_name, benign_label=benign_label
            )
            per_source = []

            for source in candidate_sources:
                src_test = test_df[test_df["source"] == source].copy()
                src_train = train_df[train_df["source"] != source].copy()

                if src_train.empty or src_test.empty:
                    continue
                if src_train["label"].nunique() < 2:
                    continue

                metrics = evaluate(
                    src_train,
                    src_test,
                    model_name,
                    benign_label=benign_label,
                )
                row = {
                    "scenario": scenario.name,
                    "model": model_name,
                    "heldout_source": source,
                    **metrics,
                }
                rows.append(row)
                per_source.append(row)
                print(
                    f"scenario={scenario.name:<35} model={model_name:<24} "
                    f"source={source:<20} f1_macro={metrics['f1_macro']:.4f} "
                    f"n_test={metrics['n_test']}"
                )

            if per_source:
                f1_vals = pd.Series([float(r["f1_macro"]) for r in per_source])
                worst_row = min(per_source, key=lambda x: float(x["f1_macro"]))
                summary_rows.append(
                    {
                        "scenario": scenario.name,
                        "model": model_name,
                        "n_sources": int(len(per_source)),
                        "mean_holdout_f1_macro": float(f1_vals.mean()),
                        "std_holdout_f1_macro": float(f1_vals.std(ddof=0)),
                        "worst_holdout_f1_macro": float(f1_vals.min()),
                        "worst_holdout_source": worst_row["heldout_source"],
                        "baseline_test_f1_macro": baseline["f1_macro"],
                        "robust_score": float(
                            f1_vals.mean()
                            - 0.5 * f1_vals.std(ddof=0)
                            - 0.5 * (1.0 - f1_vals.min())
                        ),
                    }
                )

    return rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Source-holdout evaluation")
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument(
        "--mode",
        default="scenario_test",
        choices=["scenario_test"],
        help="Evaluation mode. 'scenario_test' evaluates each scenario train against held-out test sources.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_CHOICES,
        choices=MODEL_CHOICES,
        help="Models to evaluate",
    )
    parser.add_argument("--min-test-rows", type=int, default=150)
    parser.add_argument(
        "--benign-label",
        default="legit",
        help="Label treated as benign for binary suspicious-vs-benign metrics",
    )
    parser.add_argument("--output-csv", default="results/source_split_eval.csv")
    parser.add_argument("--output-json", default="results/source_split_eval.json")
    parser.add_argument(
        "--output-summary-csv",
        default="results/source_split_summary.csv",
        help="Aggregated robustness scores per scenario/model",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    scenarios = find_scenarios(processed_dir)
    if not scenarios:
        raise RuntimeError("No scenario_* directories found")

    rows, summary_rows = evaluate_scenario_test_holdouts(
        processed_dir=processed_dir,
        scenarios=scenarios,
        models=list(args.models),
        min_test_rows=args.min_test_rows,
        benign_label=args.benign_label,
    )

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(
            ["scenario", "model", "f1_macro"],
            ascending=[True, True, True],
        )
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("robust_score", ascending=False)
    summary_csv = Path(args.output_summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)

    report = {
        "mode": args.mode,
        "processed_dir": str(processed_dir),
        "models": list(args.models),
        "min_test_rows": args.min_test_rows,
        "benign_label": args.benign_label,
        "results": rows,
        "summary": summary_rows,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved per-source CSV: {out_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
