from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import (
    INPUT_MODES,
    NUMERIC_FEATURES,
    build_model,
    ensure_features,
    select_model_input,
)


MODEL_CHOICES = [
    "hybrid_logreg",
    "hybrid_linear_svc_cal",
    "hybrid_sgd_log",
    "hybrid_sgd_hinge",
]


def find_scenarios(processed_dir: Path) -> list[Path]:
    out = []
    for p in sorted(processed_dir.glob("scenario_*")):
        if (p / "train.csv").exists():
            out.append(p)
    return out


def load_shared_splits(shared_dir: Path) -> dict[str, pd.DataFrame]:
    candidates = {
        "val_iid": [shared_dir / "val_iid.csv", shared_dir / "val.csv"],
        "test_iid": [shared_dir / "test_iid.csv", shared_dir / "test.csv"],
        "test_hard_source": [shared_dir / "test_hard_source.csv"],
        "test_hard_cluster": [shared_dir / "test_hard_cluster.csv"],
        "test_deployment": [shared_dir / "test_deployment.csv"],
    }
    out: dict[str, pd.DataFrame] = {}
    for split, paths in candidates.items():
        fp = next((p for p in paths if p.exists()), None)
        if fp is None:
            continue
        out[split] = ensure_features(pd.read_csv(fp, low_memory=False))
    if "val_iid" not in out or "test_iid" not in out:
        raise FileNotFoundError(
            "Missing required shared splits: val_iid/val and test_iid/test"
        )
    return out


def evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    input_mode: str,
) -> dict:
    labels = sorted(
        set(train_df["label"].astype(str)).union(set(test_df["label"].astype(str)))
    )
    le = LabelEncoder()
    le.fit(labels)

    y_train = le.transform(train_df["label"].astype(str))
    y_test = le.transform(test_df["label"].astype(str))

    x_train = select_model_input(train_df, input_mode)
    x_test = select_model_input(test_df, input_mode)

    model = build_model(model_name, input_mode=input_mode)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    label_idx = list(range(len(le.classes_)))
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_test, pred, average="weighted", zero_division=0)
        ),
        "rows": int(len(test_df)),
        "classification_report": classification_report(
            y_test,
            pred,
            labels=label_idx,
            target_names=[str(x) for x in le.classes_],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=label_idx).tolist(),
        "labels": [str(x) for x in le.classes_],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate scenario/model on IID, hard, and deployment splits"
    )
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument(
        "--models", nargs="+", default=MODEL_CHOICES, choices=MODEL_CHOICES
    )
    parser.add_argument("--output-csv", default="results/split_suite_results.csv")
    parser.add_argument(
        "--output-summary-csv", default="results/split_suite_summary.csv"
    )
    parser.add_argument("--output-json", default="results/split_suite_results.json")
    parser.add_argument(
        "--input-mode",
        choices=INPUT_MODES,
        default="text_only",
    )
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    shared = load_shared_splits(processed / "shared")

    scenarios = find_scenarios(processed)
    if args.scenarios:
        allowed = set(args.scenarios)
        scenarios = [s for s in scenarios if s.name in allowed]
    if not scenarios:
        raise RuntimeError("No scenarios selected")

    rows = []
    details_dir = Path(args.output_csv).parent / "split_suite_details"
    details_dir.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        train_df = ensure_features(
            pd.read_csv(scenario / "train.csv", low_memory=False)
        )
        for model_name in args.models:
            detail = {
                "scenario": scenario.name,
                "model": model_name,
                "input_mode": args.input_mode,
                "train_rows": int(len(train_df)),
                "train_distribution": train_df["label"].value_counts().to_dict(),
                "splits": {},
            }
            for split_name, split_df in shared.items():
                m = evaluate(train_df, split_df, model_name, input_mode=args.input_mode)
                detail["splits"][split_name] = m
                rows.append(
                    {
                        "scenario": scenario.name,
                        "model": model_name,
                        "input_mode": args.input_mode,
                        "split": split_name,
                        **m,
                    }
                )
                print(
                    f"{scenario.name} {model_name} {split_name} f1_macro={m['f1_macro']:.4f}"
                )

            detail_path = details_dir / f"{scenario.name}__{model_name}.json"
            detail_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")

    out_df = pd.DataFrame(rows)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    summary = out_df.pivot_table(
        index=["scenario", "model"],
        columns="split",
        values=["f1_macro", "balanced_accuracy", "accuracy", "rows"],
        aggfunc="first",
    )
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary_csv = Path(args.output_summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"Saved results CSV: {out_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
