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
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
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
)


def to_binary_labels(labels: pd.Series, benign_labels: set[str]) -> np.ndarray:
    return labels.astype(str).map(lambda x: 0 if x in benign_labels else 1).to_numpy()


def run_two_stage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stage1_model_name: str,
    stage2_model_name: str,
    benign_labels: set[str],
    benign_output_label: str,
    input_mode: str = "text_only",
) -> dict:
    if input_mode == "text_only":
        x_train = train_df[["text_input"]]
        x_val = val_df[["text_input"]]
        x_test = test_df[["text_input"]]
    elif input_mode == "features_only":
        x_train = train_df[NUMERIC_FEATURES]
        x_val = val_df[NUMERIC_FEATURES]
        x_test = test_df[NUMERIC_FEATURES]
    else:
        x_train = train_df[["text_input", *NUMERIC_FEATURES]]
        x_val = val_df[["text_input", *NUMERIC_FEATURES]]
        x_test = test_df[["text_input", *NUMERIC_FEATURES]]

    # Stage 1: benign vs suspicious
    y_train_bin = to_binary_labels(train_df["label"], benign_labels)
    y_val_bin = to_binary_labels(val_df["label"], benign_labels)
    y_test_bin = to_binary_labels(test_df["label"], benign_labels)

    stage1 = build_model(stage1_model_name, input_mode=input_mode)
    stage1.fit(x_train, y_train_bin)

    val_pred_bin = stage1.predict(x_val)
    test_pred_bin = stage1.predict(x_test)

    p1, r1, f11, _ = precision_recall_fscore_support(
        y_test_bin, test_pred_bin, average="binary", zero_division=0
    )
    stage1_metrics = {
        "accuracy": float(accuracy_score(y_test_bin, test_pred_bin)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test_bin, test_pred_bin)),
        "precision_suspicious": float(p1),
        "recall_suspicious": float(r1),
        "f1_suspicious": float(f11),
        "confusion_matrix": confusion_matrix(y_test_bin, test_pred_bin).tolist(),
        "val_f1_suspicious": float(
            f1_score(y_val_bin, val_pred_bin, average="binary", zero_division=0)
        ),
    }

    # Stage 2: suspicious subclass classification
    train_susp = train_df[~train_df["label"].astype(str).isin(benign_labels)].copy()
    test_susp = test_df[~test_df["label"].astype(str).isin(benign_labels)].copy()

    if train_susp.empty:
        raise RuntimeError("No suspicious samples for stage 2 training")

    stage2_classes = sorted(train_susp["label"].astype(str).unique().tolist())
    le2 = LabelEncoder()
    le2.fit(stage2_classes)

    stage2 = build_model(stage2_model_name, input_mode=input_mode)
    y2_train = le2.transform(train_susp["label"].astype(str))
    stage2.fit(train_susp, y2_train)

    # Stage 2 standalone eval on true suspicious subset
    y2_test = (
        le2.transform(test_susp["label"].astype(str))
        if not test_susp.empty
        else np.array([])
    )
    y2_pred = stage2.predict(test_susp) if not test_susp.empty else np.array([])

    stage2_metrics = {}
    if len(y2_test) > 0:
        stage2_metrics = {
            "accuracy": float(accuracy_score(y2_test, y2_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y2_test, y2_pred)),
            "f1_macro": float(
                f1_score(y2_test, y2_pred, average="macro", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y2_test, y2_pred, average="weighted", zero_division=0)
            ),
            "classification_report": classification_report(
                y2_test,
                y2_pred,
                target_names=list(le2.classes_),
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y2_test, y2_pred).tolist(),
            "classes": list(le2.classes_),
        }

    # Gated final prediction on full test set
    final_pred = []
    suspicious_idx = np.where(test_pred_bin == 1)[0]

    if len(suspicious_idx) > 0:
        stage2_pred_for_susp = stage2.predict(x_test.iloc[suspicious_idx])
    else:
        stage2_pred_for_susp = np.array([])

    s_ptr = 0
    for i in range(len(test_df)):
        if test_pred_bin[i] == 0:
            final_pred.append(benign_output_label)
        else:
            pred_label = le2.inverse_transform([int(stage2_pred_for_susp[s_ptr])])[0]
            final_pred.append(pred_label)
            s_ptr += 1

    y_true = test_df["label"].astype(str).to_numpy()
    final_pred = np.array(final_pred)

    # evaluate overall multiclass
    all_classes = sorted(np.unique(np.concatenate([y_true, final_pred])))
    overall = {
        "accuracy": float(accuracy_score(y_true, final_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, final_pred)),
        "f1_macro": float(
            f1_score(y_true, final_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, final_pred, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y_true,
            final_pred,
            labels=all_classes,
            target_names=all_classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, final_pred, labels=all_classes
        ).tolist(),
        "classes": all_classes,
    }

    return {
        "config": {
            "stage1_model": stage1_model_name,
            "stage2_model": stage2_model_name,
            "input_mode": input_mode,
            "benign_labels": sorted(benign_labels),
            "benign_output_label": benign_output_label,
        },
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
        "overall": overall,
        "gating": {
            "predicted_suspicious_count": int((test_pred_bin == 1).sum()),
            "predicted_benign_count": int((test_pred_bin == 0).sum()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate two-stage email classifier"
    )
    parser.add_argument("--scenario", required=True, help="Scenario folder name")
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument(
        "--model",
        default="hybrid_logreg",
        choices=[
            "hybrid_logreg",
            "hybrid_linear_svc_cal",
            "hybrid_sgd_log",
            "hybrid_sgd_hinge",
        ],
        help="Fallback model used for both stages if --stage1-model/--stage2-model are not set.",
    )
    parser.add_argument(
        "--stage1-model",
        default=None,
        choices=[
            "hybrid_logreg",
            "hybrid_linear_svc_cal",
            "hybrid_sgd_log",
            "hybrid_sgd_hinge",
        ],
    )
    parser.add_argument(
        "--stage2-model",
        default=None,
        choices=[
            "hybrid_logreg",
            "hybrid_linear_svc_cal",
            "hybrid_sgd_log",
            "hybrid_sgd_hinge",
        ],
    )
    parser.add_argument(
        "--benign-labels",
        default="legit",
        help="Comma-separated benign labels for stage1 (e.g. legit,newsletter,promotion)",
    )
    parser.add_argument(
        "--benign-output-label",
        default="legit",
        help="Label assigned when stage1 predicts benign",
    )
    parser.add_argument("--results-dir", default="results/two_stage")
    parser.add_argument(
        "--input-mode",
        choices=INPUT_MODES,
        default="text_only",
    )
    args = parser.parse_args()

    benign_labels = {x.strip() for x in args.benign_labels.split(",") if x.strip()}
    stage1_model = args.stage1_model or args.model
    stage2_model = args.stage2_model or args.model

    processed_dir = Path(args.processed_dir)
    scenario_dir = processed_dir / args.scenario
    train_path = scenario_dir / "train.csv"
    val_path = scenario_dir / "val.csv"
    test_path = processed_dir / "shared" / "test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    train_df = ensure_features(pd.read_csv(train_path, low_memory=False))
    val_df = ensure_features(pd.read_csv(val_path, low_memory=False))
    test_df = ensure_features(pd.read_csv(test_path, low_memory=False))

    result = run_two_stage(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        stage1_model_name=stage1_model,
        stage2_model_name=stage2_model,
        benign_labels=benign_labels,
        benign_output_label=args.benign_output_label,
        input_mode=args.input_mode,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir / f"{args.scenario}__s1-{stage1_model}__s2-{stage2_model}__{ts}.json"
    )
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Saved two-stage result: {out_path}")
    print("Overall:")
    print(
        json.dumps(
            {
                "accuracy": result["overall"]["accuracy"],
                "balanced_accuracy": result["overall"]["balanced_accuracy"],
                "f1_macro": result["overall"]["f1_macro"],
                "f1_weighted": result["overall"]["f1_weighted"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
