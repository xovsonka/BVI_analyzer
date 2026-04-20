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
    f1_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import build_model, ensure_features, select_model_input


CLASS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "phishing": ("verify", "mailbox", "login", "password", "account", "security"),
    "spoofing": ("microsoft", "paypal", "apple", "google", "brand", "alert"),
    "financial_fraud": ("invoice", "payment", "billing", "bank", "transfer", "due"),
    "spam": ("offer", "reward", "promo", "discount", "selected", "claim"),
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


def infer_dataset_id(receiver: str, pattern: re.Pattern[str]) -> int | None:
    m = pattern.search(str(receiver or ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def build_seed_attack_map(seed_csv: Path) -> dict[int, str]:
    if not seed_csv.exists():
        return {}
    df = pd.read_csv(seed_csv, low_memory=False)
    if "id" not in df.columns or "attack_type" not in df.columns:
        return {}
    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["attack_type"] = df["attack_type"].map(normalize_attack_type)
    df = df.dropna(subset=["id", "attack_type"])
    out: dict[int, str] = {}
    for _, row in df.iterrows():
        out[int(row["id"])] = str(row["attack_type"])
    return out


def build_file_to_dataset_id(
    metadata_csv: Path, pattern: re.Pattern[str]
) -> dict[str, int]:
    if not metadata_csv.exists():
        return {}
    df = pd.read_csv(metadata_csv, low_memory=False)
    if "eml_path" not in df.columns or "to" not in df.columns:
        return {}
    out: dict[str, int] = {}
    for _, row in df.iterrows():
        key = Path(str(row.get("eml_path", ""))).name
        if not key:
            continue
        idx = infer_dataset_id(str(row.get("to", "")), pattern)
        if idx is not None:
            out[key] = idx
    return out


def load_exp1_with_multiclass_gt(
    exp1_csv: Path,
    seed_csv: Path,
    metadata_csv: Path,
    receiver_pattern: str,
    spoofing_policy: str,
) -> pd.DataFrame:
    df = pd.read_csv(exp1_csv, low_memory=False)
    if "label" not in df.columns:
        df["label"] = "unknown"

    seed_map = build_seed_attack_map(seed_csv)
    id_re = re.compile(receiver_pattern, flags=re.IGNORECASE)
    file_to_id = build_file_to_dataset_id(metadata_csv, id_re)

    receiver_series = df.get("receiver", pd.Series([""] * len(df), index=df.index))
    dataset_id = receiver_series.map(lambda x: infer_dataset_id(str(x), id_re))
    if "file" in df.columns and file_to_id:
        file_id = df["file"].map(lambda x: file_to_id.get(Path(str(x)).name))
        dataset_id = dataset_id.where(dataset_id.notna(), file_id)

    df["dataset_id"] = dataset_id
    df["y_true_multiclass"] = (
        df["dataset_id"]
        .map(seed_map)
        .map(lambda x: apply_spoofing_policy(normalize_attack_type(x), spoofing_policy))
    )
    df["y_true_binary"] = df["y_true_multiclass"].map(
        lambda s: 0
        if str(s).strip().lower() == "legit"
        else 1
        if isinstance(s, str)
        else np.nan
    )
    return ensure_features(df)


def rebalance_multiclass(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    work = df.copy()
    grouped = {label: part for label, part in work.groupby(work["label"].astype(str))}
    if not grouped:
        return work
    target = max(len(part) for part in grouped.values())
    blocks = []
    for label, part in grouped.items():
        if len(part) >= target:
            blocks.append(part.sample(n=target, random_state=seed))
            continue
        take = part.sample(n=target, replace=True, random_state=seed)
        take = take.copy()
        take["scenario_n_aug"] = "rebalance_oversample"
        blocks.append(take)
    out = pd.concat(blocks, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out["scenario_n_stage"] = "step1_balanced"
    return out


def count_keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    low = str(text or "").lower()
    return sum(1 for kw in keywords if kw in low)


def add_hard_negatives(df: pd.DataFrame, multiplier: float, seed: int) -> pd.DataFrame:
    if multiplier <= 1.0:
        out = df.copy()
        out["scenario_n_stage"] = "step1_balanced"
        return out

    rows = []
    for idx, row in df.iterrows():
        label = str(row.get("label", "")).strip().lower()
        if label == "legit":
            continue
        text = str(row.get("text_input", ""))
        other_hits = 0
        for other, kws in CLASS_KEYWORDS.items():
            if other == label:
                continue
            other_hits += int(count_keyword_hits(text, kws) > 0)
        if other_hits > 0:
            rows.append(idx)

    if not rows:
        out = df.copy()
        out["scenario_n_stage"] = "step1_balanced"
        return out

    hard_df = df.loc[rows].copy()
    extra_n = int(len(hard_df) * (multiplier - 1.0))
    if extra_n <= 0:
        out = df.copy()
        out["scenario_n_stage"] = "step1_balanced"
        return out

    extra = hard_df.sample(n=extra_n, replace=True, random_state=seed).copy()
    extra["scenario_n_aug"] = "hard_negative_upsample"
    out = pd.concat([df.copy(), extra], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out["scenario_n_stage"] = "step1_balanced_hardneg"
    return out


def evaluate_binary(y_true_mc: pd.Series, y_pred_mc: pd.Series) -> dict[str, float]:
    y_true = y_true_mc.map(lambda s: 0 if str(s).lower() == "legit" else 1).to_numpy(
        dtype=int
    )
    y_pred = y_pred_mc.map(lambda s: 0 if str(s).lower() == "legit" else 1).to_numpy(
        dtype=int
    )
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    tpr = rec
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "bin_accuracy": float(accuracy_score(y_true, y_pred)),
        "bin_balanced_accuracy": float((tpr + tnr) / 2.0),
        "bin_precision": float(prec),
        "bin_recall": float(rec),
        "bin_f1": float(f1),
        "bin_mcc": float(matthews_corrcoef(y_true, y_pred)),
        "bin_tn": tn,
        "bin_fp": fp,
        "bin_fn": fn,
        "bin_tp": tp,
    }


def evaluate_multiclass(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true_s = y_true.astype(str)
    y_pred_s = y_pred.astype(str)
    return {
        "mc_accuracy": float(accuracy_score(y_true_s, y_pred_s)),
        "mc_balanced_accuracy": float(balanced_accuracy_score(y_true_s, y_pred_s)),
        "mc_f1_macro": float(
            f1_score(y_true_s, y_pred_s, average="macro", zero_division=0)
        ),
        "mc_f1_weighted": float(
            f1_score(y_true_s, y_pred_s, average="weighted", zero_division=0)
        ),
    }


def tune_class_scales(
    model,
    x_val: pd.DataFrame,
    y_val_idx: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    proba = model.predict_proba(x_val)
    scales = {label: 1.0 for label in labels}
    candidates = [0.7, 0.85, 1.0, 1.15, 1.3]

    def score(sc: dict[str, float]) -> float:
        scale_vec = np.array([sc[lbl] for lbl in labels], dtype=float)
        adjusted = proba / scale_vec
        pred = adjusted.argmax(axis=1)
        return float(f1_score(y_val_idx, pred, average="macro", zero_division=0))

    best = score(scales)
    improved = True
    while improved:
        improved = False
        for label in labels:
            local_best = scales[label]
            local_score = best
            for c in candidates:
                trial = dict(scales)
                trial[label] = c
                s = score(trial)
                if s > local_score:
                    local_score = s
                    local_best = c
            if local_best != scales[label]:
                scales[label] = local_best
                best = local_score
                improved = True
    return scales


def predict_with_scales(
    model, x: pd.DataFrame, labels: list[str], scales: dict[str, float]
) -> np.ndarray:
    proba = model.predict_proba(x)
    scale_vec = np.array([scales[lbl] for lbl in labels], dtype=float)
    pred_idx = (proba / scale_vec).argmax(axis=1)
    return pred_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scenario N: balanced multiclass + hard negatives + per-class thresholds + two-stage"
    )
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument(
        "--base-scenario", default="scenario_m_anti_template_feature_regularized"
    )
    parser.add_argument("--exp1-input-csv", default="results/eml_model_input.csv")
    parser.add_argument(
        "--seed-csv", default="results/experiment_1/gophish_seed_input_all_rich.csv"
    )
    parser.add_argument(
        "--metadata-csv", default="dataset/processed/mailhog_messages_with_labels.csv"
    )
    parser.add_argument("--receiver-id-pattern", default=r"user(\d+)@")
    parser.add_argument(
        "--spoofing-policy",
        choices=["map_to_phishing", "drop", "keep"],
        default="map_to_phishing",
    )
    parser.add_argument(
        "--model", default="hybrid_logreg", choices=["hybrid_logreg", "hybrid_sgd_log"]
    )
    parser.add_argument("--hardneg-multiplier", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-summary-csv",
        default="results/exp1_scenario_n_summary.csv",
    )
    parser.add_argument(
        "--output-per-class-csv",
        default="results/exp1_scenario_n_per_class.csv",
    )
    parser.add_argument(
        "--output-predictions-csv",
        default="results/exp1_scenario_n_predictions.csv",
    )
    parser.add_argument(
        "--output-json",
        default="results/exp1_scenario_n_summary.json",
    )
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    base_dir = processed / args.base_scenario
    train_path = base_dir / "train.csv"
    val_path = base_dir / "val.csv"
    test_path = base_dir / "test.csv"
    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required base scenario file: {p}")

    train_df = ensure_features(pd.read_csv(train_path, low_memory=False))
    val_df = ensure_features(pd.read_csv(val_path, low_memory=False))
    test_df = ensure_features(pd.read_csv(test_path, low_memory=False))

    # Step 1: balanced multiclass + hard negatives
    train_n = rebalance_multiclass(train_df, seed=args.seed)
    train_n = add_hard_negatives(
        train_n, multiplier=args.hardneg_multiplier, seed=args.seed
    )

    base_suffix = args.base_scenario.replace("scenario_", "").replace("_", "-")
    scenario_n_dir = processed / f"scenario_n_multiclass_hardneg_from_{base_suffix}"
    scenario_n_dir.mkdir(parents=True, exist_ok=True)
    train_n.to_csv(scenario_n_dir / "train.csv", index=False)
    val_df.to_csv(scenario_n_dir / "val.csv", index=False)
    test_df.to_csv(scenario_n_dir / "test.csv", index=False)

    labels = sorted(
        set(train_n["label"].astype(str)) | set(val_df["label"].astype(str))
    )
    le = LabelEncoder()
    le.fit(labels)

    y_train = le.transform(train_n["label"].astype(str))
    x_train = select_model_input(train_n, input_mode="text_only")

    single = build_model(args.model, input_mode="text_only")
    single.fit(x_train, y_train)

    x_val = select_model_input(val_df, input_mode="text_only")
    y_val = le.transform(val_df["label"].astype(str))

    # Step 2: per-class thresholds (implemented as class probability scales)
    if not hasattr(single, "predict_proba"):
        raise RuntimeError(
            f"Model {args.model} does not support predict_proba for threshold tuning"
        )
    class_labels = [str(x) for x in le.classes_]
    scales = tune_class_scales(single, x_val, y_val, class_labels)

    # Exp1 inference set + multiclass GT
    exp1_df = load_exp1_with_multiclass_gt(
        exp1_csv=Path(args.exp1_input_csv),
        seed_csv=Path(args.seed_csv),
        metadata_csv=Path(args.metadata_csv),
        receiver_pattern=args.receiver_id_pattern,
        spoofing_policy=args.spoofing_policy,
    )
    eval_mask = exp1_df["y_true_multiclass"].notna()
    eval_df = exp1_df.loc[eval_mask].copy()
    if eval_df.empty:
        raise RuntimeError("No Exp1 rows with multiclass ground truth found")

    x_eval = select_model_input(eval_df, input_mode="text_only")
    y_true_mc = eval_df["y_true_multiclass"].astype(str)

    # Baseline single-stage on Scenario N training
    pred_base_idx = single.predict(x_eval)
    pred_base = pd.Series(le.inverse_transform(pred_base_idx), index=eval_df.index).map(
        lambda x: apply_spoofing_policy(str(x), args.spoofing_policy) or ""
    )

    # Threshold-scaled single-stage
    pred_thr_idx = predict_with_scales(single, x_eval, class_labels, scales)
    pred_thr = pd.Series(le.inverse_transform(pred_thr_idx), index=eval_df.index).map(
        lambda x: apply_spoofing_policy(str(x), args.spoofing_policy) or ""
    )

    # Step 3: two-stage (binary gate + suspicious subclass)
    train_n_bin = train_n.copy()
    train_n_bin["label_bin"] = (
        train_n_bin["label"].astype(str).map(lambda s: 0 if s == "legit" else 1)
    )
    stage1 = build_model(args.model, input_mode="text_only")
    stage1.fit(
        select_model_input(train_n_bin, "text_only"),
        train_n_bin["label_bin"].to_numpy(),
    )

    susp_train = train_n[train_n["label"].astype(str) != "legit"].copy()
    le2 = LabelEncoder()
    le2.fit(sorted(susp_train["label"].astype(str).unique().tolist()))
    stage2 = build_model(args.model, input_mode="text_only")
    stage2.fit(
        select_model_input(susp_train, "text_only"),
        le2.transform(susp_train["label"].astype(str)),
    )

    gate_pred = stage1.predict(x_eval)
    pred_two = []
    if (gate_pred == 1).any():
        idx_susp = np.where(gate_pred == 1)[0]
        x_susp = x_eval.iloc[idx_susp]
        stage2_pred = stage2.predict(x_susp)
        stage2_labels = le2.inverse_transform(stage2_pred)
        ptr = 0
        for i in range(len(gate_pred)):
            if gate_pred[i] == 0:
                pred_two.append("legit")
            else:
                pred_two.append(str(stage2_labels[ptr]))
                ptr += 1
    else:
        pred_two = ["legit"] * len(gate_pred)
    pred_two_s = pd.Series(pred_two, index=eval_df.index).map(
        lambda x: apply_spoofing_policy(str(x), args.spoofing_policy) or ""
    )

    methods = {
        "step1_single_stage": pred_base,
        "step2_threshold_scaled": pred_thr,
        "step3_two_stage": pred_two_s,
    }

    summary_rows = []
    per_class_rows = []
    pred_rows = []
    for method, pred in methods.items():
        valid = pred.astype(str).str.strip() != ""
        y_true_m = y_true_mc.loc[valid]
        pred_m = pred.loc[valid].astype(str)
        mc = evaluate_multiclass(y_true_m, pred_m)
        bn = evaluate_binary(y_true_m, pred_m)
        summary_rows.append(
            {
                "method": method,
                "base_scenario": args.base_scenario,
                "model": args.model,
                "rows_eval": int(valid.sum()),
                **mc,
                **bn,
            }
        )

        rep = classification_report(y_true_m, pred_m, output_dict=True, zero_division=0)
        for cls_name, cls_values in rep.items():
            if cls_name in {"accuracy", "macro avg", "weighted avg"} or not isinstance(
                cls_values, dict
            ):
                continue
            per_class_rows.append(
                {
                    "method": method,
                    "class": cls_name,
                    "precision": float(cls_values.get("precision", 0.0)),
                    "recall": float(cls_values.get("recall", 0.0)),
                    "f1": float(cls_values.get("f1-score", 0.0)),
                    "support": int(cls_values.get("support", 0)),
                }
            )

        tmp = pd.DataFrame(
            {
                "method": method,
                "file": eval_df.get("file", ""),
                "mailhog_id": eval_df.get("mailhog_id", ""),
                "subject": eval_df.get("subject", ""),
                "y_true_multiclass": y_true_mc,
                "pred_label": pred,
            }
        )
        pred_rows.append(tmp)

    out_summary = pd.DataFrame(summary_rows)
    out_per_class = pd.DataFrame(per_class_rows)
    out_pred = pd.concat(pred_rows, ignore_index=True)

    output_summary_csv = Path(args.output_summary_csv)
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary.to_csv(output_summary_csv, index=False)

    output_per_class_csv = Path(args.output_per_class_csv)
    output_per_class_csv.parent.mkdir(parents=True, exist_ok=True)
    out_per_class.to_csv(output_per_class_csv, index=False)

    output_predictions_csv = Path(args.output_predictions_csv)
    output_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    out_pred.to_csv(output_predictions_csv, index=False)

    payload = {
        "base_scenario": args.base_scenario,
        "model": args.model,
        "hardneg_multiplier": args.hardneg_multiplier,
        "class_scales": scales,
        "summary": summary_rows,
        "scenario_n_dir": str(scenario_n_dir),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved Scenario N train/val/test to: {scenario_n_dir}")
    print(f"Saved summary CSV: {output_summary_csv}")
    print(f"Saved per-class CSV: {output_per_class_csv}")
    print(f"Saved predictions CSV: {output_predictions_csv}")
    print(f"Saved summary JSON: {output_json}")
    print("Top methods by multiclass F1:")
    print(
        out_summary.sort_values(
            ["mc_f1_macro", "mc_balanced_accuracy"], ascending=False
        )[
            ["method", "mc_f1_macro", "mc_balanced_accuracy", "mc_accuracy", "bin_f1"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
