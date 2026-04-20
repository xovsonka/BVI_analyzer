from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import INPUT_MODES, build_model, ensure_features, select_model_input


def parse_configs(values: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Invalid config '{value}'. Use scenario:model format.")
        scenario, model = value.split(":", 1)
        out.append((scenario.strip(), model.strip()))
    return out


def normalize_binary_label(v: object) -> int | None:
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in {"", "none", "nan"}:
        return None
    if s in {"0", "legit", "benign", "ham", "safe"}:
        return 0
    if s in {"1", "malicious", "phishing", "spam", "fraud", "financial_fraud"}:
        return 1
    try:
        return 0 if int(float(s)) == 0 else 1
    except Exception:
        return None


def compute_binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true_np = y_true.to_numpy(dtype=int)
    y_pred_np = y_pred.to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y_true_np)),
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "precision": float(precision_score(y_true_np, y_pred_np, zero_division=0)),
        "recall": float(recall_score(y_true_np, y_pred_np, zero_division=0)),
        "f1": float(f1_score(y_true_np, y_pred_np, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_np, y_pred_np)),
        "mcc": float(matthews_corrcoef(y_true_np, y_pred_np)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def normalize_attack_type(v: object) -> str | None:
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in {"", "none", "nan"}:
        return None
    aliases = {
        "legitimate": "legit",
        "benign": "legit",
        "fraud": "financial_fraud",
    }
    return aliases.get(s, s)


def apply_spoofing_policy(value: str | None, policy: str) -> str | None:
    if value is None:
        return None
    label = str(value).strip().lower()
    if label != "spoofing":
        return label
    if policy == "map_to_phishing":
        return "phishing"
    if policy == "drop":
        return None
    return label


def build_seed_attack_type_map(seed_csv: Path) -> dict[int, str]:
    if not seed_csv.exists():
        return {}
    seed_df = pd.read_csv(seed_csv, low_memory=False)
    if "id" not in seed_df.columns:
        return {}
    if "attack_type" not in seed_df.columns:
        return {}
    seed_df = seed_df.copy()
    seed_df["id"] = pd.to_numeric(seed_df["id"], errors="coerce")
    seed_df["attack_type"] = seed_df["attack_type"].map(normalize_attack_type)
    seed_df = seed_df.dropna(subset=["id", "attack_type"])
    out: dict[int, str] = {}
    for _, row in seed_df.iterrows():
        out[int(row["id"])] = str(row["attack_type"])
    return out


def infer_dataset_id_from_receiver(
    receiver: str, pattern: re.Pattern[str]
) -> int | None:
    m = pattern.search(str(receiver or ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def build_file_to_dataset_id_map(
    metadata_csv: Path, pattern: re.Pattern[str]
) -> dict[str, int]:
    if not metadata_csv.exists():
        return {}
    meta = pd.read_csv(metadata_csv, low_memory=False)
    if "eml_path" not in meta.columns or "to" not in meta.columns:
        return {}

    out: dict[str, int] = {}
    for _, row in meta.iterrows():
        file_key = Path(str(row.get("eml_path", ""))).name
        if not file_key:
            continue
        dataset_id = infer_dataset_id_from_receiver(str(row.get("to", "")), pattern)
        if dataset_id is not None:
            out[file_key] = dataset_id
    return out


def compute_multiclass_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true_s = y_true.astype(str)
    y_pred_s = y_pred.astype(str)
    return {
        "rows_multiclass": int(len(y_true_s)),
        "mc_accuracy": float(accuracy_score(y_true_s, y_pred_s)),
        "mc_balanced_accuracy": float(balanced_accuracy_score(y_true_s, y_pred_s)),
        "mc_f1_macro": float(
            f1_score(y_true_s, y_pred_s, average="macro", zero_division=0)
        ),
        "mc_f1_weighted": float(
            f1_score(y_true_s, y_pred_s, average="weighted", zero_division=0)
        ),
    }


def parse_class_scales(value: str) -> dict[str, float]:
    out: dict[str, float] = {}
    raw = (value or "").strip()
    if not raw:
        return out
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid class scale '{item}'. Use format class:scale,class:scale"
            )
        key, val = item.split(":", 1)
        out[key.strip()] = float(val.strip())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Exp1 campaign with text-only final models"
    )
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument("--input-csv", default="results/eml_model_input.csv")
    parser.add_argument(
        "--ground-truth-seed-csv",
        default="results/experiment_1/gophish_seed_input_all_rich.csv",
        help="Seed CSV containing id->attack_type mapping",
    )
    parser.add_argument(
        "--receiver-id-pattern",
        default=r"user(\d+)@",
        help="Regex for dataset id extraction from receiver",
    )
    parser.add_argument(
        "--metadata-csv",
        default="dataset/processed/mailhog_messages_with_labels.csv",
        help="Optional metadata CSV used to recover recipient ids from eml_path->to",
    )
    parser.add_argument(
        "--spoofing-policy",
        choices=["map_to_phishing", "drop", "keep"],
        default="map_to_phishing",
        help="How to treat spoofing labels during Exp1 multiclass evaluation",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "scenario_l_combined_anti_source:hybrid_linear_svc_cal",
            "scenario_m_anti_template_feature_regularized:hybrid_sgd_hinge",
        ],
        help="List of scenario:model configs",
    )
    parser.add_argument(
        "--input-mode",
        choices=INPUT_MODES,
        default="text_only",
        help="Model input mode for training/inference",
    )
    parser.add_argument(
        "--output-summary-csv",
        default="results/exp1_text_models_binary_eval.csv",
    )
    parser.add_argument(
        "--output-predictions-csv",
        default="results/exp1_text_models_predictions.csv",
    )
    parser.add_argument(
        "--output-json",
        default="results/exp1_text_models_binary_eval.json",
    )
    parser.add_argument(
        "--output-multiclass-per-class-csv",
        default="results/exp1_text_models_multiclass_per_class.csv",
    )
    parser.add_argument(
        "--class-scales",
        default="",
        help="Optional class probability scales, e.g. spam:0.8,financial_fraud:1.1",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    raw_df = pd.read_csv(input_csv, low_memory=False)
    if "label" not in raw_df.columns:
        raw_df["label"] = "unknown"

    seed_map = build_seed_attack_type_map(Path(args.ground_truth_seed_csv))
    id_re = re.compile(args.receiver_id_pattern, flags=re.IGNORECASE)
    file_to_id = build_file_to_dataset_id_map(Path(args.metadata_csv), id_re)
    raw_df["dataset_id_from_receiver"] = raw_df.get(
        "receiver", pd.Series([""] * len(raw_df), index=raw_df.index)
    ).map(lambda x: infer_dataset_id_from_receiver(str(x), id_re))
    if "file" in raw_df.columns and file_to_id:
        missing_id = raw_df["dataset_id_from_receiver"].isna()
        file_ids = raw_df["file"].map(lambda p: file_to_id.get(Path(str(p)).name))
        raw_df.loc[missing_id, "dataset_id_from_receiver"] = file_ids.loc[missing_id]
    raw_df["y_true_multiclass"] = raw_df["dataset_id_from_receiver"].map(seed_map)
    raw_df["y_true_multiclass"] = raw_df["y_true_multiclass"].map(
        lambda x: apply_spoofing_policy(
            normalize_attack_type(x),
            args.spoofing_policy,
        )
    )

    exp1_df = ensure_features(raw_df)
    exp1_df["y_true"] = pd.Series(
        [None] * len(exp1_df), index=exp1_df.index, dtype="object"
    )
    exp1_df["y_true"] = exp1_df.get("label_hint", pd.Series([None] * len(exp1_df))).map(
        normalize_binary_label
    )
    missing_binary = exp1_df["y_true"].isna() & exp1_df["y_true_multiclass"].notna()
    exp1_df.loc[missing_binary, "y_true"] = exp1_df.loc[
        missing_binary, "y_true_multiclass"
    ].map(lambda s: 0 if str(s).strip().lower() == "legit" else 1)

    processed_dir = Path(args.processed_dir)
    configs = parse_configs(args.configs)
    class_scales = parse_class_scales(args.class_scales)

    summary_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    per_class_rows: list[dict] = []

    for scenario, model_name in configs:
        train_path = processed_dir / scenario / "train.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing train file: {train_path}")

        train_df = ensure_features(pd.read_csv(train_path, low_memory=False))
        le = LabelEncoder()
        y_train = le.fit_transform(train_df["label"].astype(str))

        model = build_model(model_name, input_mode=args.input_mode)
        x_train = select_model_input(train_df, input_mode=args.input_mode)
        model.fit(x_train, y_train)

        x_exp1 = select_model_input(exp1_df, input_mode=args.input_mode)
        if class_scales and hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_exp1)
            class_names = [str(c) for c in le.classes_]
            scales = np.array(
                [class_scales.get(name, 1.0) for name in class_names], dtype=float
            )
            pred_idx = (proba / scales).argmax(axis=1)
        else:
            pred_idx = model.predict(x_exp1)
        pred_label = pd.Series(le.inverse_transform(pred_idx), index=exp1_df.index)
        pred_binary = pred_label.map(
            lambda s: 0 if str(s).strip().lower() == "legit" else 1
        )

        eval_mask = exp1_df["y_true"].notna()
        y_true = exp1_df.loc[eval_mask, "y_true"].astype(int)
        y_pred = pred_binary.loc[eval_mask].astype(int)

        metrics = compute_binary_metrics(y_true, y_pred)

        multi_mask = exp1_df["y_true_multiclass"].notna()
        y_true_mc = exp1_df.loc[multi_mask, "y_true_multiclass"].astype(str)
        y_pred_mc = pred_label.loc[multi_mask].map(
            lambda x: apply_spoofing_policy(str(x), args.spoofing_policy) or ""
        )
        valid_mc = y_pred_mc.astype(str).str.strip() != ""
        y_true_mc = y_true_mc.loc[valid_mc]
        y_pred_mc = y_pred_mc.loc[valid_mc].astype(str)
        mc_metrics = compute_multiclass_metrics(y_true_mc, y_pred_mc)
        report = classification_report(
            y_true_mc, y_pred_mc, output_dict=True, zero_division=0
        )
        for cls_name, cls_values in report.items():
            if cls_name in {"accuracy", "macro avg", "weighted avg"}:
                continue
            if not isinstance(cls_values, dict):
                continue
            per_class_rows.append(
                {
                    "model_config": f"{scenario}:{model_name}",
                    "class": cls_name,
                    "precision": float(cls_values.get("precision", 0.0)),
                    "recall": float(cls_values.get("recall", 0.0)),
                    "f1": float(cls_values.get("f1-score", 0.0)),
                    "support": int(cls_values.get("support", 0)),
                }
            )

        metrics.update(
            {
                "scenario": scenario,
                "model": model_name,
                "model_config": f"{scenario}:{model_name}",
                "input_mode": args.input_mode,
                "rows_total": int(len(exp1_df)),
                "rows_labeled": int(eval_mask.sum()),
                "pred_suspicious_rate": float(pred_binary.mean()),
                "pred_legit_count": int((pred_binary == 0).sum()),
                "pred_suspicious_count": int((pred_binary == 1).sum()),
            }
        )
        metrics.update(mc_metrics)
        summary_rows.append(metrics)

        pred_df = pd.DataFrame(
            {
                "model_config": f"{scenario}:{model_name}",
                "file": exp1_df.get("file", ""),
                "mailhog_id": exp1_df.get("mailhog_id", ""),
                "subject": exp1_df.get("subject", ""),
                "label_hint": exp1_df.get("label_hint", ""),
                "y_true_multiclass": exp1_df.get("y_true_multiclass", ""),
                "pred_label": pred_label,
                "pred_binary": pred_binary,
            }
        )
        prediction_frames.append(pred_df)

        print(
            f"{scenario:<44} {model_name:<22} "
            f"bin_f1={metrics['f1']:.4f} bin_bal_acc={metrics['balanced_accuracy']:.4f} "
            f"mc_f1={metrics['mc_f1_macro']:.4f} rows_labeled={metrics['rows_labeled']}"
        )

    out_summary = pd.DataFrame(summary_rows)
    out_pred = pd.concat(prediction_frames, ignore_index=True)

    output_summary_csv = Path(args.output_summary_csv)
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary.to_csv(output_summary_csv, index=False)

    output_predictions_csv = Path(args.output_predictions_csv)
    output_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    out_pred.to_csv(output_predictions_csv, index=False)

    output_per_class_csv = Path(args.output_multiclass_per_class_csv)
    output_per_class_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_class_rows).to_csv(output_per_class_csv, index=False)

    payload = {
        "configs": [f"{s}:{m}" for s, m in configs],
        "summary": summary_rows,
        "output_summary_csv": str(output_summary_csv),
        "output_predictions_csv": str(output_predictions_csv),
        "output_multiclass_per_class_csv": str(output_per_class_csv),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved summary CSV: {output_summary_csv}")
    print(f"Saved predictions CSV: {output_predictions_csv}")
    print(f"Saved multiclass per-class CSV: {output_per_class_csv}")
    print(f"Saved summary JSON: {output_json}")


if __name__ == "__main__":
    main()
