from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import INPUT_MODES, NUMERIC_FEATURES, build_model, ensure_features
from training.train_two_stage import run_two_stage


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


def eval_single_stage(
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

    if input_mode == "text_only":
        x_train = train_df[["text_input"]]
        x_test = test_df[["text_input"]]
    elif input_mode == "features_only":
        x_train = train_df[NUMERIC_FEATURES]
        x_test = test_df[NUMERIC_FEATURES]
    else:
        x_train = train_df[["text_input", *NUMERIC_FEATURES]]
        x_test = test_df[["text_input", *NUMERIC_FEATURES]]

    model = build_model(model_name, input_mode=input_mode)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    report = classification_report(
        y_test,
        pred,
        target_names=list(le.classes_),
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_test, pred, average="weighted", zero_division=0)
        ),
        "classification_report": report,
        "classes": list(le.classes_),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full benchmark: single-stage models and all two-stage model combinations"
    )
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument("--results-dir", default="results/architecture_benchmark")
    parser.add_argument(
        "--benign-labels",
        default="legit",
        help="Comma-separated benign labels for stage1 in two-stage",
    )
    parser.add_argument(
        "--benign-output-label",
        default="legit",
        help="Output label when two-stage stage1 predicts benign",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=MODEL_CHOICES,
    )
    parser.add_argument(
        "--single-models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=None,
        help="Alias for --models (kept for compatibility).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Optional list of scenario directory names (e.g. scenario_g_source_balanced).",
    )
    parser.add_argument(
        "--skip-two-stage",
        action="store_true",
        help="Run only single-stage models.",
    )
    parser.add_argument(
        "--input-mode",
        choices=INPUT_MODES,
        default="text_only",
        help="Model input mode for single-stage runs (recommended: text_only baseline).",
    )
    args = parser.parse_args()

    benign_labels = {x.strip() for x in args.benign_labels.split(",") if x.strip()}

    processed_dir = Path(args.processed_dir)
    scenarios = find_scenarios(processed_dir)
    if args.scenarios:
        allowed = set(args.scenarios)
        scenarios = [s for s in scenarios if s.name in allowed]
    if not scenarios:
        raise RuntimeError("No scenario_* directories found")

    selected_models = args.single_models if args.single_models else args.models

    shared_test_path = processed_dir / "shared" / "test.csv"
    if not shared_test_path.exists():
        raise FileNotFoundError(f"Missing shared test set: {shared_test_path}")

    test_df = ensure_features(pd.read_csv(shared_test_path, low_memory=False))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.results_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    print("Benchmarking architectures")
    print(f"Scenarios: {[s.name for s in scenarios]}")
    print(f"Models: {selected_models}")
    print(f"Input mode: {args.input_mode}")

    for scenario in scenarios:
        train_path = scenario / "train.csv"
        val_path = scenario / "val.csv"

        train_df = ensure_features(pd.read_csv(train_path, low_memory=False))
        val_df = ensure_features(pd.read_csv(val_path, low_memory=False))

        # Single-stage runs
        for model_name in selected_models:
            print(f"[single] {scenario.name} :: {model_name}")
            single = eval_single_stage(
                train_df,
                test_df,
                model_name,
                input_mode=args.input_mode,
            )

            summary_rows.append(
                {
                    "scenario": scenario.name,
                    "architecture": "single_stage",
                    "stage1_model": model_name,
                    "stage2_model": "",
                    "test_accuracy": single["accuracy"],
                    "test_balanced_accuracy": single["balanced_accuracy"],
                    "test_f1_macro": single["f1_macro"],
                    "test_f1_weighted": single["f1_weighted"],
                }
            )

            detail_path = out_dir / f"{scenario.name}__single__{model_name}.json"
            detail_path.write_text(json.dumps(single, indent=2), encoding="utf-8")

        # Two-stage combinations (all pairs)
        if args.skip_two_stage:
            continue

        for stage1_model in selected_models:
            for stage2_model in selected_models:
                print(
                    f"[two-stage] {scenario.name} :: s1={stage1_model} s2={stage2_model}"
                )
                two = run_two_stage(
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    stage1_model_name=stage1_model,
                    stage2_model_name=stage2_model,
                    benign_labels=benign_labels,
                    benign_output_label=args.benign_output_label,
                )

                summary_rows.append(
                    {
                        "scenario": scenario.name,
                        "architecture": "two_stage",
                        "stage1_model": stage1_model,
                        "stage2_model": stage2_model,
                        "test_accuracy": two["overall"]["accuracy"],
                        "test_balanced_accuracy": two["overall"]["balanced_accuracy"],
                        "test_f1_macro": two["overall"]["f1_macro"],
                        "test_f1_weighted": two["overall"]["f1_weighted"],
                        "stage1_f1_suspicious": two["stage1"].get(
                            "f1_suspicious", np.nan
                        ),
                        "stage2_f1_macro": two["stage2"].get("f1_macro", np.nan),
                    }
                )

                detail_path = (
                    out_dir
                    / f"{scenario.name}__two_stage__s1-{stage1_model}__s2-{stage2_model}.json"
                )
                detail_path.write_text(json.dumps(two, indent=2), encoding="utf-8")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        by=["test_f1_macro", "test_f1_weighted"], ascending=False
    )
    summary_csv = out_dir / "benchmark_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    top_txt = out_dir / "top_results.txt"
    with top_txt.open("w", encoding="utf-8") as f:
        f.write("Top benchmark results by test_f1_macro\n")
        f.write("=" * 100 + "\n")
        for _, row in summary_df.head(30).iterrows():
            f.write(
                f"{row['scenario']:<35} {row['architecture']:<12} "
                f"s1={row['stage1_model']:<16} s2={row['stage2_model']:<16} "
                f"macro_f1={row['test_f1_macro']:.4f} weighted_f1={row['test_f1_weighted']:.4f} "
                f"acc={row['test_accuracy']:.4f}\n"
            )

    best = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    best_json = out_dir / "best_overall.json"
    best_json.write_text(json.dumps(best, indent=2), encoding="utf-8")

    print("\nDone")
    print(f"Saved summary: {summary_csv}")
    print(f"Saved top table: {top_txt}")
    print(f"Saved best config: {best_json}")
    if best:
        print("Best overall:")
        print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
