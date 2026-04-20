from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _normalize_binary_label(v: object) -> int | None:
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in {"", "none", "nan"}:
        return None
    if s in {"0", "legit", "benign", "ham", "safe"}:
        return 0
    if s in {
        "1",
        "malicious",
        "phishing",
        "spam",
        "fraud",
        "financial_fraud",
        "suspicious",
    }:
        return 1
    try:
        num = int(float(s))
        return 0 if num == 0 else 1
    except Exception:
        return None


def load_with_labels(input_csv: Path, metadata_csv: Path | None) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)

    label_cols = ["label", "label_hint", "label_binary", "target"]
    label_source = None
    for col in label_cols:
        if col in df.columns and df[col].notna().any():
            y = df[col].apply(_normalize_binary_label)
            if y.notna().sum() > 0:
                df["y_true"] = y
                label_source = col
                break

    if label_source is None and metadata_csv is not None and metadata_csv.exists():
        meta = pd.read_csv(metadata_csv, low_memory=False)
        if "eml_path" in meta.columns:
            meta = meta.copy()
            meta["_file_key"] = meta["eml_path"].astype(str).map(lambda p: Path(p).name)
            df["_file_key"] = df["file"].astype(str).map(lambda p: Path(p).name)
            if "label_hint" in meta.columns:
                meta_label_col = "metadata_label_hint"
                merged = df.merge(
                    meta[["_file_key", "label_hint"]].rename(
                        columns={"label_hint": meta_label_col}
                    ),
                    on="_file_key",
                    how="left",
                )
                y = merged[meta_label_col].apply(_normalize_binary_label)
                if y.notna().sum() > 0:
                    df = merged
                    df["y_true"] = y
                    label_source = f"metadata.{meta_label_col}"

    if label_source is None:
        raise RuntimeError(
            "Could not find usable ground-truth labels. Provide label/label_hint in input CSV or metadata CSV."
        )

    if "heuristic_score" not in df.columns:
        raise RuntimeError("Input CSV is missing 'heuristic_score' column")

    df = df[df["y_true"].notna()].copy()
    df["y_true"] = df["y_true"].astype(int)
    df["score"] = pd.to_numeric(df["heuristic_score"], errors="coerce").fillna(0.0)
    df["score_proba"] = (df["score"] / 100.0).clip(0.0, 1.0)
    df.attrs["label_source"] = label_source
    return df


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0.0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0.0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "fpr": fpr,
        "fnr": fnr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
    }


def sweep_thresholds(df: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    rows = []
    y_true = df["y_true"].to_numpy(dtype=int)
    score = df["score"].to_numpy(dtype=float)
    score_proba = df["score_proba"].to_numpy(dtype=float)

    for t in thresholds:
        y_pred = (score >= t).astype(int)
        m = binary_metrics(y_true, y_pred, score_proba)
        m["threshold"] = int(t)
        rows.append(m)

    out = pd.DataFrame(rows)
    metric_cols = [
        "threshold",
        "rows",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "mcc",
        "fpr",
        "fnr",
        "tn",
        "fp",
        "fn",
        "tp",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]
    return out[metric_cols]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate heuristic analyzer with binary metrics and threshold sweep"
    )
    parser.add_argument("--input-csv", default="results/eml_model_input.csv")
    parser.add_argument(
        "--metadata-csv", default="dataset/processed/mailhog_messages.csv"
    )
    parser.add_argument("--default-threshold", type=int, default=40)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=[20, 30, 40, 50, 60],
    )
    parser.add_argument(
        "--threshold-range",
        nargs=3,
        type=int,
        metavar=("START", "END", "STEP"),
        default=None,
        help="Optional inclusive threshold range to append, e.g. 20 80 5",
    )
    parser.add_argument(
        "--output-binary-csv", default="results/heuristic_eval_binary.csv"
    )
    parser.add_argument(
        "--output-thresholds-csv",
        default="results/heuristic_eval_thresholds.csv",
    )
    parser.add_argument("--output-json", default="results/heuristic_eval_binary.json")
    parser.add_argument(
        "--output-confusion-csv",
        default="results/heuristic_eval_confusion_matrix.csv",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else None
    df = load_with_labels(input_csv, metadata_csv)

    thresholds = list(args.thresholds)
    if args.threshold_range is not None:
        start, end, step = args.threshold_range
        thresholds.extend(list(range(start, end + 1, step)))
    thresholds = sorted(set(int(x) for x in thresholds))

    y_true = df["y_true"].to_numpy(dtype=int)
    y_score = df["score_proba"].to_numpy(dtype=float)
    y_pred_default = (
        df["score"].to_numpy(dtype=float) >= int(args.default_threshold)
    ).astype(int)

    metrics = binary_metrics(y_true, y_pred_default, y_score)
    metrics["default_threshold"] = int(args.default_threshold)
    metrics["label_source"] = str(df.attrs.get("label_source", "unknown"))

    out_bin_df = pd.DataFrame([metrics])
    out_thr_df = sweep_thresholds(df, thresholds)

    cm = confusion_matrix(y_true, y_pred_default, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["true_benign", "true_malicious"],
        columns=["pred_benign", "pred_malicious"],
    )

    output_binary_csv = Path(args.output_binary_csv)
    output_binary_csv.parent.mkdir(parents=True, exist_ok=True)
    out_bin_df.to_csv(output_binary_csv, index=False)

    output_thresholds_csv = Path(args.output_thresholds_csv)
    output_thresholds_csv.parent.mkdir(parents=True, exist_ok=True)
    out_thr_df.to_csv(output_thresholds_csv, index=False)

    output_confusion_csv = Path(args.output_confusion_csv)
    output_confusion_csv.parent.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(output_confusion_csv)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "binary_metrics": metrics,
                "thresholds": out_thr_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved: {output_binary_csv}")
    print(f"Saved: {output_thresholds_csv}")
    print(f"Saved: {output_confusion_csv}")
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
