from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


MULTICLASS_LABELS = ["legit", "phishing", "spam", "financial_fraud"]
RETENTION_SPECS = [
    (
        "display_name_spoof",
        "expected_display_name_spoof_flag",
        lambda df: _num(df, "display_name_spoof_flag").gt(0),
    ),
    (
        "external_domain_url",
        "expected_external_domain_url",
        lambda df: _num(df, "external_domain_url_count").gt(0),
    ),
    (
        "sender_url_mismatch",
        "expected_sender_url_mismatch",
        lambda df: _num(df, "sender_url_domain_mismatch_ratio").gt(0),
    ),
    (
        "suspicious_tld",
        "expected_suspicious_tld",
        lambda df: _num(df, "suspicious_tld_count").gt(0),
    ),
]


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _str(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=str)
    return df[col].fillna(default).astype(str)


def _risky_url_signal(df: pd.DataFrame) -> pd.Series:
    return (
        (_num(df, "external_domain_url_count") > 0)
        | (_num(df, "suspicious_tld_count") > 0)
        | (_num(df, "obfuscated_url_count") > 0)
        | (_num(df, "shortener_url_count") > 0)
        | (_num(df, "ip_url_count") > 0)
        | (_num(df, "credential_url_count") > 0)
    )


def heuristic_binary(df: pd.DataFrame, threshold: int = 20) -> pd.Series:
    score = _num(df, "heuristic_score")
    pred = pd.Series(["legit"] * len(df), index=df.index, dtype=str)
    pred.loc[score >= threshold] = "suspicious"
    return pred


def _binary_confusion(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    y_true_num = (y_true.astype(str).str.lower() == "suspicious").astype(int)
    y_pred_num = (y_pred.astype(str).str.lower() == "suspicious").astype(int)
    return {
        "tn": int(((y_true_num == 0) & (y_pred_num == 0)).sum()),
        "fp": int(((y_true_num == 0) & (y_pred_num == 1)).sum()),
        "fn": int(((y_true_num == 1) & (y_pred_num == 0)).sum()),
        "tp": int(((y_true_num == 1) & (y_pred_num == 1)).sum()),
    }


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true_num = (y_true.astype(str).str.lower() == "suspicious").astype(int)
    y_pred_num = (y_pred.astype(str).str.lower() == "suspicious").astype(int)
    if len(y_true_num) == 0:
        return {
            "bin_accuracy": 0.0,
            "bin_balanced_accuracy": 0.0,
            "bin_precision": 0.0,
            "bin_recall": 0.0,
            "bin_f1": 0.0,
        }
    bal_acc = (
        float((y_true_num == y_pred_num).mean())
        if y_true_num.nunique() < 2
        else float(balanced_accuracy_score(y_true_num, y_pred_num))
    )
    return {
        "bin_accuracy": float(accuracy_score(y_true_num, y_pred_num)),
        "bin_balanced_accuracy": bal_acc,
        "bin_precision": float(
            precision_score(y_true_num, y_pred_num, zero_division=0)
        ),
        "bin_recall": float(recall_score(y_true_num, y_pred_num, zero_division=0)),
        "bin_f1": float(f1_score(y_true_num, y_pred_num, zero_division=0)),
    }


def multiclass_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "mc_accuracy": 0.0,
            "mc_balanced_accuracy": 0.0,
            "mc_f1_macro": 0.0,
            "mc_f1_weighted": 0.0,
        }
    bal_acc = (
        float((y_true.astype(str) == y_pred.astype(str)).mean())
        if y_true.astype(str).nunique() < 2
        else float(balanced_accuracy_score(y_true, y_pred))
    )
    return {
        "mc_accuracy": float(accuracy_score(y_true, y_pred)),
        "mc_balanced_accuracy": bal_acc,
        "mc_f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "mc_f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def build_multiclass_reports(
    y_true: pd.Series,
    y_pred: pd.Series,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    report = classification_report(
        y_true,
        y_pred,
        labels=MULTICLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    per_class_rows = []
    for label in MULTICLASS_LABELS:
        stats = report.get(label, {})
        per_class_rows.append(
            {
                "method": method,
                "label": label,
                "precision": float(stats.get("precision", 0.0)),
                "recall": float(stats.get("recall", 0.0)),
                "f1": float(stats.get("f1-score", 0.0)),
                "support": int(stats.get("support", 0)),
            }
        )

    conf = confusion_matrix(y_true, y_pred, labels=MULTICLASS_LABELS)
    conf_long = []
    for i, true_label in enumerate(MULTICLASS_LABELS):
        for j, pred_label in enumerate(MULTICLASS_LABELS):
            conf_long.append(
                {
                    "method": method,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": int(conf[i][j]),
                }
            )

    summary = pd.DataFrame([{"method": method, **multiclass_metrics(y_true, y_pred)}])
    return summary, pd.DataFrame(per_class_rows), pd.DataFrame(conf_long)


def _threshold_sweep(
    score: pd.Series,
    y_true_binary: pd.Series,
    start: int,
    end: int,
    step: int,
) -> pd.DataFrame:
    rows = []
    for threshold in range(start, end + 1, step):
        pred = np.where(score >= threshold, "suspicious", "legit")
        rows.append(
            {
                "threshold": threshold,
                **binary_metrics(
                    y_true_binary, pd.Series(pred, index=y_true_binary.index)
                ),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        best_f1 = float(out["bin_f1"].max())
        best_bal = float(out["bin_balanced_accuracy"].max())
        out["is_best_f1"] = out["bin_f1"].eq(best_f1).astype(int)
        out["is_best_balanced_accuracy"] = (
            out["bin_balanced_accuracy"].eq(best_bal).astype(int)
        )
    return out


def _extract_dataset_id(df: pd.DataFrame) -> pd.Series:
    if "dataset_id" in df.columns:
        dataset_id = pd.to_numeric(df["dataset_id"], errors="coerce")
    else:
        dataset_id = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
    receiver_id = pd.to_numeric(
        _str(df, "receiver").str.extract(r"user(\d+)@", expand=False),
        errors="coerce",
    )
    dataset_id = dataset_id.where(dataset_id.notna(), receiver_id)
    return dataset_id.astype("Int64")


def _merge_optional_metadata(
    merged: pd.DataFrame,
    seed_csv: str,
    expected_csv: str,
) -> pd.DataFrame:
    out = merged.copy()
    out["dataset_id"] = _extract_dataset_id(out)

    seed_path = Path(seed_csv)
    if seed_path.exists():
        seed = pd.read_csv(seed_path, low_memory=False)
        if "id" in seed.columns:
            seed["id"] = pd.to_numeric(seed["id"], errors="coerce").astype("Int64")
            keep = [
                c
                for c in [
                    "id",
                    "attack_type",
                    "generation_kind",
                    "generation_family",
                    "label",
                ]
                if c in seed.columns
            ]
            out = out.merge(
                seed[keep],
                left_on="dataset_id",
                right_on="id",
                how="left",
                suffixes=("", "_seed"),
            )
            if "id" in out.columns:
                out = out.drop(columns=["id"])
    else:
        print(f"[WARN] Seed CSV not found for slice metadata: {seed_path}")

    expected_path = Path(expected_csv)
    if expected_path.exists():
        expected = pd.read_csv(expected_path, low_memory=False)
        if "id" in expected.columns:
            expected["id"] = pd.to_numeric(expected["id"], errors="coerce").astype(
                "Int64"
            )
            out = out.merge(
                expected,
                left_on="dataset_id",
                right_on="id",
                how="left",
                suffixes=("", "_expected"),
            )
            if "id" in out.columns:
                out = out.drop(columns=["id"])
    else:
        print(f"[WARN] Expected CSV not found for retention analysis: {expected_path}")

    return out


def _build_retention_table(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for indicator, expected_col, detected_fn in RETENTION_SPECS:
        if expected_col not in merged.columns:
            continue
        exp = _num(merged, expected_col).astype(int)
        det = detected_fn(merged).astype(int)
        tp = int(((exp == 1) & (det == 1)).sum())
        fn = int(((exp == 1) & (det == 0)).sum())
        rows.append(
            {
                "indicator": indicator,
                "injected_positive": int(exp.sum()),
                "detected_positive": int(det.sum()),
                "tp": tp,
                "fn": fn,
                "injected_to_detected_retention_rate": (
                    float(tp / (tp + fn)) if (tp + fn) else 0.0
                ),
                "note": "Track B injected metadata (not strict per-indicator ground truth)",
            }
        )
    return pd.DataFrame(rows)


def _slice_metrics(
    merged: pd.DataFrame,
    methods: list[tuple[str, str]],
) -> pd.DataFrame:
    rows = []
    candidate_slice_cols = [
        "attack_type",
        "generation_kind",
        "generation_family",
        "expected_display_name_spoof_flag",
        "expected_external_domain_url",
        "expected_sender_url_mismatch",
        "expected_suspicious_tld",
    ]
    available = [c for c in candidate_slice_cols if c in merged.columns]
    for slice_col in available:
        for slice_value, subset in merged.groupby(slice_col, dropna=False):
            if subset.empty:
                continue
            slice_value_str = str(slice_value)
            for method_name, col in methods:
                rows.append(
                    {
                        "task": "binary",
                        "slice_column": slice_col,
                        "slice_value": slice_value_str,
                        "method": method_name,
                        "rows": int(len(subset)),
                        **binary_metrics(
                            subset["y_true_binary"], subset[col].astype(str)
                        ),
                    }
                )
            rows.append(
                {
                    "task": "multiclass_ml_only",
                    "slice_column": slice_col,
                    "slice_value": slice_value_str,
                    "method": "ml_only",
                    "rows": int(len(subset)),
                    **multiclass_metrics(
                        subset["y_true_multiclass"].astype(str),
                        subset["ml_pred"].astype(str),
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Experiment 2 campaign report tables"
    )
    parser.add_argument(
        "--analyzed-csv", default="results/experiment_2_campaign_model_input.csv"
    )
    parser.add_argument(
        "--ml-pred-csv", default="results/experiment_2_campaign_ml_predictions.csv"
    )
    parser.add_argument(
        "--seed-csv",
        default="results/experiment_2/gophish_seed_input_header_focused_50_50.csv",
    )
    parser.add_argument(
        "--expected-csv",
        default="results/experiment_2/gophish_seed_input_header_focused_expected.csv",
    )
    parser.add_argument("--heur-threshold", type=int, default=20)
    parser.add_argument("--hybrid-high-threshold", type=int, default=35)
    parser.add_argument("--hybrid-low-threshold", type=int, default=8)
    parser.add_argument("--threshold-sweep-start", type=int, default=0)
    parser.add_argument("--threshold-sweep-end", type=int, default=80)
    parser.add_argument("--threshold-sweep-step", type=int, default=5)
    parser.add_argument(
        "--output-summary-csv",
        default="results/experiment_2_campaign_indicator_summary.csv",
    )
    parser.add_argument(
        "--output-method-compare-csv",
        default="results/experiment_2_campaign_method_compare.csv",
    )
    parser.add_argument(
        "--output-qualitative-csv",
        default="results/experiment_2_campaign_qualitative_examples.csv",
    )
    args = parser.parse_args()

    analyzed = pd.read_csv(args.analyzed_csv, low_memory=False)
    pred = pd.read_csv(args.ml_pred_csv, low_memory=False)

    analyzed = analyzed.copy()
    pred = pred.copy()
    analyzed["file_key"] = analyzed["file"].astype(str).map(lambda p: Path(p).name)
    pred["file_key"] = pred["file"].astype(str).map(lambda p: Path(p).name)

    pred_cols = [
        c
        for c in [
            "file_key",
            "dataset_id",
            "ml_pred",
            "y_true_multiclass",
            "y_true_binary",
            "ml_pred_binary",
        ]
        if c in pred.columns
    ]
    merged = analyzed.merge(pred[pred_cols], on="file_key", how="inner")
    merged["dataset_id"] = _extract_dataset_id(merged)
    merged["y_true_multiclass"] = _str(merged, "y_true_multiclass").str.lower()
    merged["ml_pred"] = _str(merged, "ml_pred").str.lower()
    if "y_true_binary" not in merged.columns:
        merged["y_true_binary"] = np.where(
            merged["y_true_multiclass"].eq("legit"), "legit", "suspicious"
        )
    else:
        merged["y_true_binary"] = _str(merged, "y_true_binary").str.lower()
    if "ml_pred_binary" not in merged.columns:
        merged["ml_pred_binary"] = np.where(
            merged["ml_pred"].eq("legit"), "legit", "suspicious"
        )
    else:
        merged["ml_pred_binary"] = _str(merged, "ml_pred_binary").str.lower()

    merged = _merge_optional_metadata(merged, args.seed_csv, args.expected_csv)

    score = _num(merged, "heuristic_score")
    merged["heur_pred_binary"] = heuristic_binary(merged, threshold=args.heur_threshold)
    risky_url_signal = _risky_url_signal(merged)
    strong_header_signal = (
        (_num(merged, "from_reply_mismatch") > 0)
        | (_num(merged, "from_return_path_mismatch") > 0)
        | (_num(merged, "display_name_spoof_flag") > 0)
        | (_num(merged, "received_anomaly_flag") > 0)
        | risky_url_signal
    )

    hybrid = merged["ml_pred_binary"].astype(str).copy()
    force_suspicious = (score >= args.hybrid_high_threshold) & (hybrid == "legit")
    hybrid.loc[force_suspicious] = "suspicious"
    relax_to_legit = (
        (score < args.hybrid_low_threshold)
        & (hybrid != "legit")
        & (~strong_header_signal)
    )
    hybrid.loc[relax_to_legit] = "legit"
    merged["hybrid_pred_binary"] = hybrid

    indicator_summary = pd.DataFrame(
        [
            {
                "rows": int(len(merged)),
                "from_reply_mismatch_count": int(
                    _num(merged, "from_reply_mismatch").sum()
                ),
                "from_return_path_mismatch_count": int(
                    _num(merged, "from_return_path_mismatch").sum()
                ),
                "display_name_spoof_count": int(
                    _num(merged, "display_name_spoof_flag").sum()
                ),
                "received_anomaly_count": int(
                    _num(merged, "received_anomaly_flag").sum()
                ),
                "risky_url_messages": int(risky_url_signal.sum()),
                "external_domain_url_count": int(
                    _num(merged, "external_domain_url_count").sum()
                ),
                "suspicious_tld_count": int(_num(merged, "suspicious_tld_count").sum()),
                "obfuscated_url_count": int(_num(merged, "obfuscated_url_count").sum()),
                "shortener_url_count": int(_num(merged, "shortener_url_count").sum()),
                "ip_url_count": int(_num(merged, "ip_url_count").sum()),
                "credential_url_count": int(_num(merged, "credential_url_count").sum()),
                "message_id_domain_mismatch_count": int(
                    _num(merged, "message_id_domain_mismatch").sum()
                ),
                "trusted_domain_only_count": int(
                    _num(merged, "trusted_domain_only").sum()
                ),
            }
        ]
    )

    mean_scores = (
        merged.assign(heuristic_score=score)
        .groupby("y_true_multiclass", as_index=False)["heuristic_score"]
        .mean()
        .rename(columns={"heuristic_score": "mean_heuristic_score"})
    )

    methods = [
        ("heuristic_only", "heur_pred_binary"),
        ("ml_only", "ml_pred_binary"),
        ("heuristic_plus_ml", "hybrid_pred_binary"),
    ]
    method_rows = []
    binary_confusion_rows = []
    for method_name, col in methods:
        metrics = binary_metrics(merged["y_true_binary"], merged[col].astype(str))
        method_rows.append({"method": method_name, "rows": int(len(merged)), **metrics})
        binary_confusion_rows.append(
            {
                "method": method_name,
                **_binary_confusion(merged["y_true_binary"], merged[col].astype(str)),
            }
        )
    method_compare = pd.DataFrame(method_rows)
    binary_confusion = pd.DataFrame(binary_confusion_rows)

    mc_summary, mc_per_class, mc_confusion = build_multiclass_reports(
        merged["y_true_multiclass"].astype(str),
        merged["ml_pred"].astype(str),
        method="ml_only",
    )
    threshold_sweep = _threshold_sweep(
        score,
        merged["y_true_binary"],
        start=args.threshold_sweep_start,
        end=args.threshold_sweep_end,
        step=args.threshold_sweep_step,
    )
    retention_df = _build_retention_table(merged)
    slice_df = _slice_metrics(merged, methods)

    merged["is_ml_correct_binary"] = (
        merged["ml_pred_binary"].astype(str) == merged["y_true_binary"].astype(str)
    ).astype(int)
    merged["abs_score_center"] = (score - args.heur_threshold).abs()
    sample_good = (
        merged[merged["is_ml_correct_binary"] == 1]
        .sort_values(["abs_score_center", "heuristic_score"], ascending=[False, False])
        .head(4)
    )
    sample_bad = (
        merged[merged["is_ml_correct_binary"] == 0]
        .sort_values(["abs_score_center", "heuristic_score"], ascending=[False, False])
        .head(4)
    )
    qual = pd.concat([sample_good, sample_bad], ignore_index=True)

    def _comment(row: pd.Series) -> str:
        reasons = []
        if int(_num(pd.DataFrame([row]), "from_reply_mismatch").iloc[0]) == 1:
            reasons.append("from/reply mismatch")
        if int(_num(pd.DataFrame([row]), "from_return_path_mismatch").iloc[0]) == 1:
            reasons.append("from/return-path mismatch")
        if int(_num(pd.DataFrame([row]), "display_name_spoof_flag").iloc[0]) == 1:
            reasons.append("display-name spoof")
        if int(_num(pd.DataFrame([row]), "received_anomaly_flag").iloc[0]) == 1:
            reasons.append("received anomaly")
        if str(row.get("spf_result", "none")).lower() == "fail":
            reasons.append("SPF fail")
        if str(row.get("dkim_result", "none")).lower() == "fail":
            reasons.append("DKIM fail")
        if str(row.get("dmarc_result", "none")).lower() == "fail":
            reasons.append("DMARC fail")
        if bool(_risky_url_signal(pd.DataFrame([row])).iloc[0]):
            reasons.append("risky URL")
        if not reasons:
            return "No strong header/auth signal"
        return ", ".join(reasons[:3])

    qual["comment"] = qual.apply(_comment, axis=1)
    qual_cols = [
        "file",
        "subject",
        "attack_type",
        "generation_kind",
        "generation_family",
        "y_true_multiclass",
        "y_true_binary",
        "heuristic_score",
        "heur_pred_binary",
        "ml_pred",
        "ml_pred_binary",
        "hybrid_pred_binary",
        "from_reply_mismatch",
        "from_return_path_mismatch",
        "display_name_spoof_flag",
        "received_anomaly_flag",
        "spf_result",
        "dkim_result",
        "dmarc_result",
        "is_ml_correct_binary",
        "comment",
    ]
    qual = qual[[c for c in qual_cols if c in qual.columns]].copy()

    output_base = Path(args.output_summary_csv)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    indicator_summary.to_csv(args.output_summary_csv, index=False)
    mean_scores.to_csv(
        output_base.with_name("experiment_2_campaign_mean_score_by_class.csv"),
        index=False,
    )
    method_compare.to_csv(args.output_method_compare_csv, index=False)
    binary_confusion.to_csv(
        output_base.with_name("experiment_2_campaign_binary_confusion_counts.csv"),
        index=False,
    )
    mc_summary.to_csv(
        output_base.with_name("experiment_2_campaign_multiclass_summary.csv"),
        index=False,
    )
    mc_per_class.to_csv(
        output_base.with_name("experiment_2_campaign_multiclass_per_class.csv"),
        index=False,
    )
    mc_confusion.to_csv(
        output_base.with_name("experiment_2_campaign_multiclass_confusion_long.csv"),
        index=False,
    )
    threshold_sweep.to_csv(
        output_base.with_name("experiment_2_campaign_threshold_sweep.csv"),
        index=False,
    )
    if not retention_df.empty:
        retention_df.to_csv(
            output_base.with_name(
                "experiment_2_campaign_injected_indicator_retention.csv"
            ),
            index=False,
        )
    if not slice_df.empty:
        slice_df.to_csv(
            output_base.with_name("experiment_2_campaign_slice_metrics.csv"),
            index=False,
        )
    qual.to_csv(args.output_qualitative_csv, index=False)

    print(f"Saved indicator summary: {args.output_summary_csv}")
    print(f"Saved method comparison: {args.output_method_compare_csv}")
    print(f"Saved qualitative examples: {args.output_qualitative_csv}")
    print(
        "Saved mean score by class:",
        output_base.with_name("experiment_2_campaign_mean_score_by_class.csv"),
    )
    print(
        "Saved binary confusion counts:",
        output_base.with_name("experiment_2_campaign_binary_confusion_counts.csv"),
    )
    print(
        "Saved multiclass summary:",
        output_base.with_name("experiment_2_campaign_multiclass_summary.csv"),
    )
    print(
        "Saved multiclass per-class table:",
        output_base.with_name("experiment_2_campaign_multiclass_per_class.csv"),
    )
    print(
        "Saved multiclass confusion:",
        output_base.with_name("experiment_2_campaign_multiclass_confusion_long.csv"),
    )
    print(
        "Saved threshold sweep:",
        output_base.with_name("experiment_2_campaign_threshold_sweep.csv"),
    )
    if not retention_df.empty:
        print(
            "Saved injected-indicator retention:",
            output_base.with_name(
                "experiment_2_campaign_injected_indicator_retention.csv"
            ),
        )
    if not slice_df.empty:
        print(
            "Saved slice metrics:",
            output_base.with_name("experiment_2_campaign_slice_metrics.csv"),
        )


if __name__ == "__main__":
    main()
