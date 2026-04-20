from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import build_model, ensure_features, select_model_input


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict Exp2 campaign using final ML model"
    )
    parser.add_argument(
        "--processed-dir", default="dataset/processed/scenario_m_rebalanced_fraud_spam"
    )
    parser.add_argument(
        "--analyzed-csv", default="results/experiment_2_campaign_model_input.csv"
    )
    parser.add_argument(
        "--seed-csv", default="results/experiment_2/gophish_seed_input_mixed_50_50.csv"
    )
    parser.add_argument(
        "--metadata-csv", default="dataset/processed/mailhog_messages_exp2_campaign.csv"
    )
    parser.add_argument("--model", default="hybrid_logreg")
    parser.add_argument("--input-mode", default="text_plus_features")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=30000,
        help="Optional cap for training rows to reduce memory usage",
    )
    parser.add_argument(
        "--class-scales",
        default="financial_fraud:1.1",
        help="Optional class scales class:scale,class:scale",
    )
    parser.add_argument(
        "--output-pred-csv", default="results/experiment_2_campaign_ml_predictions.csv"
    )
    parser.add_argument(
        "--output-summary-csv", default="results/experiment_2_campaign_ml_summary.csv"
    )
    args = parser.parse_args()

    train_path = Path(args.processed_dir) / "train.csv"
    max_rows = args.max_train_rows if args.max_train_rows > 0 else None
    try:
        train_raw = pd.read_csv(train_path, low_memory=False, nrows=max_rows)
    except Exception:
        train_raw = pd.read_csv(
            train_path,
            nrows=max_rows,
            engine="python",
        )
    train_df = ensure_features(train_raw)
    analyzed = pd.read_csv(args.analyzed_csv, low_memory=False)
    if "label" not in analyzed.columns:
        analyzed["label"] = "unknown"
    analyzed = ensure_features(analyzed)

    metadata = pd.read_csv(args.metadata_csv, low_memory=False)
    seed = pd.read_csv(args.seed_csv, low_memory=False)
    seed["id"] = pd.to_numeric(seed["id"], errors="coerce").astype("Int64")
    seed["attack_type"] = (
        seed["attack_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"spoofing": "phishing"})
    )
    id_to_class = seed.set_index("id")["attack_type"].to_dict()

    pat = re.compile(r"user(\d+)@", re.I)
    file_to_id = {}
    for _, row in metadata.iterrows():
        file_key = Path(str(row.get("eml_path", ""))).name
        m = pat.search(str(row.get("to", "")))
        if file_key and m:
            file_to_id[file_key] = int(m.group(1))

    analyzed["file_key"] = analyzed["file"].astype(str).map(lambda p: Path(p).name)
    analyzed["dataset_id"] = analyzed["file_key"].map(file_to_id)
    analyzed["y_true_multiclass"] = analyzed["dataset_id"].map(id_to_class)
    analyzed = analyzed[analyzed["y_true_multiclass"].notna()].copy()

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"].astype(str))
    model = build_model(args.model, input_mode=args.input_mode)
    model.fit(select_model_input(train_df, args.input_mode), y_train)

    x_eval = select_model_input(analyzed, args.input_mode)
    pred_idx: np.ndarray
    scales = {}
    if args.class_scales.strip() and hasattr(model, "predict_proba"):
        for item in args.class_scales.split(","):
            k, v = item.split(":", 1)
            scales[k.strip()] = float(v.strip())
        proba = model.predict_proba(x_eval)
        classes = [str(c) for c in le.classes_]
        scale_vec = np.array([scales.get(c, 1.0) for c in classes], dtype=float)
        pred_idx = (proba * scale_vec).argmax(axis=1)
    else:
        pred_idx = model.predict(x_eval)

    pred_label = pd.Series(le.inverse_transform(pred_idx), index=analyzed.index)

    y_true = analyzed["y_true_multiclass"].astype(str)
    y_pred = pred_label.astype(str)
    y_true_bin = pd.Series(
        np.where(y_true.eq("legit"), "legit", "suspicious"), index=y_true.index
    )
    y_pred_bin = pd.Series(
        np.where(y_pred.eq("legit"), "legit", "suspicious"), index=y_pred.index
    )

    y_true_bin_num = y_true_bin.eq("suspicious").astype(int)
    y_pred_bin_num = y_pred_bin.eq("suspicious").astype(int)

    summary = pd.DataFrame(
        [
            {
                "rows": int(len(y_true)),
                "bin_accuracy": float(accuracy_score(y_true_bin_num, y_pred_bin_num)),
                "bin_balanced_accuracy": float(
                    balanced_accuracy_score(y_true_bin_num, y_pred_bin_num)
                ),
                "bin_precision": float(
                    precision_score(y_true_bin_num, y_pred_bin_num, zero_division=0)
                ),
                "bin_recall": float(
                    recall_score(y_true_bin_num, y_pred_bin_num, zero_division=0)
                ),
                "bin_f1": float(
                    f1_score(y_true_bin_num, y_pred_bin_num, zero_division=0)
                ),
                "mc_accuracy": float(accuracy_score(y_true, y_pred)),
                "mc_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "mc_f1_macro": float(
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                ),
                "mc_f1_weighted": float(
                    f1_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                "model": args.model,
                "input_mode": args.input_mode,
                "class_scales": args.class_scales,
            }
        ]
    )

    pred_out = analyzed[
        [
            "file",
            "mailhog_id",
            "subject",
            "heuristic_score",
            "dataset_id",
            "y_true_multiclass",
        ]
    ].copy()
    pred_out["ml_pred"] = y_pred.values
    pred_out["y_true_binary"] = y_true_bin
    pred_out["ml_pred_binary"] = y_pred_bin

    Path(args.output_pred_csv).parent.mkdir(parents=True, exist_ok=True)
    pred_out.to_csv(args.output_pred_csv, index=False)
    summary.to_csv(args.output_summary_csv, index=False)

    print(f"Saved predictions: {args.output_pred_csv}")
    print(f"Saved summary: {args.output_summary_csv}")


if __name__ == "__main__":
    main()
