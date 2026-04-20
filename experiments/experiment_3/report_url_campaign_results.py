from __future__ import annotations

import argparse
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


URL_INDICATORS = [
    "ip_url",
    "shortener_url",
    "anchor_mismatch",
    "suspicious_tld",
    "brand_typosquat",
    "obfuscated_url",
]
MULTICLASS_LABELS = ["legit", "phishing", "spam", "financial_fraud"]


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _str(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=str)
    return df[col].fillna(default).astype(str)


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true_num = y_true.astype(str).str.lower().eq("suspicious").astype(int)
    y_pred_num = y_pred.astype(str).str.lower().eq("suspicious").astype(int)
    if len(y_true_num) == 0:
        return {
            "rows": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    bal_acc = (
        float((y_true_num == y_pred_num).mean())
        if y_true_num.nunique() < 2
        else float(balanced_accuracy_score(y_true_num, y_pred_num))
    )
    return {
        "rows": int(len(y_true_num)),
        "accuracy": float(accuracy_score(y_true_num, y_pred_num)),
        "balanced_accuracy": bal_acc,
        "precision": float(precision_score(y_true_num, y_pred_num, zero_division=0)),
        "recall": float(recall_score(y_true_num, y_pred_num, zero_division=0)),
        "f1": float(f1_score(y_true_num, y_pred_num, zero_division=0)),
    }


def _multiclass_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
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


def _indicator_presence(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ip_url"] = _num(df, "ip_url_count").gt(0).astype(int)
    out["shortener_url"] = _num(df, "shortener_url_count").gt(0).astype(int)
    out["anchor_mismatch"] = _num(df, "mismatched_anchor_count").gt(0).astype(int)
    out["suspicious_tld"] = _num(df, "suspicious_tld_count").gt(0).astype(int)
    out["brand_typosquat"] = _num(df, "brand_typosquat_flag").gt(0).astype(int)
    out["obfuscated_url"] = _num(df, "obfuscated_url_count").gt(0).astype(int)
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


def _build_multiclass_reports(
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
    conf_rows = []
    for i, true_label in enumerate(MULTICLASS_LABELS):
        for j, pred_label in enumerate(MULTICLASS_LABELS):
            conf_rows.append(
                {
                    "method": method,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": int(conf[i][j]),
                }
            )

    summary = pd.DataFrame([{"method": method, **_multiclass_metrics(y_true, y_pred)}])
    return summary, pd.DataFrame(per_class_rows), pd.DataFrame(conf_rows)


def _threshold_sweep(
    score: pd.Series,
    y_true_binary: pd.Series,
    start: int,
    end: int,
    step: int,
) -> pd.DataFrame:
    rows = []
    for threshold in range(start, end + 1, step):
        pred = pd.Series(
            np.where(score >= threshold, "suspicious", "legit"),
            index=y_true_binary.index,
        )
        rows.append({"threshold": threshold, **_binary_metrics(y_true_binary, pred)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["is_best_f1"] = out["f1"].eq(float(out["f1"].max())).astype(int)
        out["is_best_balanced_accuracy"] = (
            out["balanced_accuracy"]
            .eq(float(out["balanced_accuracy"].max()))
            .astype(int)
        )
    return out


def _merge_seed_metadata(merged: pd.DataFrame, seed_csv: str) -> pd.DataFrame:
    out = merged.copy()
    out["dataset_id"] = _extract_dataset_id(out)
    seed_path = Path(seed_csv)
    if not seed_path.exists():
        print(f"[WARN] Seed CSV not found for slice metadata: {seed_path}")
        return out

    seed = pd.read_csv(seed_path, low_memory=False)
    if "id" not in seed.columns:
        return out
    seed["id"] = pd.to_numeric(seed["id"], errors="coerce").astype("Int64")
    keep = [
        c
        for c in [
            "id",
            "attack_type",
            "generation_kind",
            "generation_family",
        ]
        if c in seed.columns
    ]
    out = out.merge(seed[keep], left_on="dataset_id", right_on="id", how="left")
    if "id" in out.columns:
        out = out.drop(columns=["id"])
    return out


def _slice_metrics(
    merged: pd.DataFrame,
    methods: list[tuple[str, pd.Series]],
) -> pd.DataFrame:
    rows = []
    candidate_slice_cols = [
        "attack_type",
        "generation_kind",
        "generation_family",
        "injection_mode",
        "url_indicator_bucket",
    ]
    available = [c for c in candidate_slice_cols if c in merged.columns]
    for slice_col in available:
        for slice_value, subset in merged.groupby(slice_col, dropna=False):
            if subset.empty:
                continue
            for method_name, pred_values in methods:
                rows.append(
                    {
                        "task": "binary",
                        "slice_column": slice_col,
                        "slice_value": str(slice_value),
                        "method": method_name,
                        **_binary_metrics(
                            subset["y_true_binary"], pred_values.loc[subset.index]
                        ),
                    }
                )
            rows.append(
                {
                    "task": "multiclass_ml_only",
                    "slice_column": slice_col,
                    "slice_value": str(slice_value),
                    "method": "ml_only",
                    "rows": int(len(subset)),
                    **_multiclass_metrics(
                        subset["y_true_multiclass"].astype(str),
                        subset["ml_pred"].astype(str),
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Experiment 3 Track B URL-focused report tables"
    )
    parser.add_argument(
        "--analyzed-csv", default="results/experiment_3_campaign_model_input.csv"
    )
    parser.add_argument(
        "--ml-pred-csv", default="results/experiment_3_campaign_ml_predictions.csv"
    )
    parser.add_argument(
        "--expected-csv",
        default="results/experiment_3/gophish_seed_input_url_focused_expected.csv",
    )
    parser.add_argument(
        "--seed-csv",
        default="results/experiment_3/gophish_seed_input_url_focused_50_50.csv",
    )
    parser.add_argument("--heur-threshold", type=int, default=20)
    parser.add_argument("--hybrid-high-threshold", type=int, default=35)
    parser.add_argument("--hybrid-low-threshold", type=int, default=8)
    parser.add_argument("--threshold-sweep-start", type=int, default=0)
    parser.add_argument("--threshold-sweep-end", type=int, default=80)
    parser.add_argument("--threshold-sweep-step", type=int, default=5)
    parser.add_argument(
        "--output-indicator-summary-csv",
        default="results/experiment_3_campaign_url_indicator_summary.csv",
    )
    parser.add_argument(
        "--output-method-compare-csv",
        default="results/experiment_3_campaign_method_compare.csv",
    )
    parser.add_argument(
        "--output-correlation-csv",
        default="results/experiment_3_campaign_url_score_correlation.csv",
    )
    parser.add_argument(
        "--output-error-impact-csv",
        default="results/experiment_3_campaign_url_error_impact.csv",
    )
    parser.add_argument(
        "--output-confusion-csv",
        default="results/experiment_3_campaign_binary_confusion_counts.csv",
    )
    parser.add_argument(
        "--output-qualitative-csv",
        default="results/experiment_3_campaign_qualitative_examples.csv",
    )
    args = parser.parse_args()

    analyzed = pd.read_csv(args.analyzed_csv, low_memory=False)
    pred = pd.read_csv(args.ml_pred_csv, low_memory=False)
    expected = pd.read_csv(args.expected_csv, low_memory=False)

    analyzed["file_key"] = analyzed["file"].astype(str).map(lambda p: Path(p).name)
    pred["file_key"] = pred["file"].astype(str).map(lambda p: Path(p).name)

    pred_cols = [
        c
        for c in [
            "file_key",
            "dataset_id",
            "ml_pred",
            "ml_pred_binary",
            "y_true_binary",
            "y_true_multiclass",
        ]
        if c in pred.columns
    ]
    merged = analyzed.merge(pred[pred_cols], on="file_key", how="inner")
    merged["dataset_id"] = _extract_dataset_id(merged)
    merged["ml_pred"] = _str(merged, "ml_pred").str.lower()
    merged["ml_pred_binary"] = _str(merged, "ml_pred_binary").str.lower()
    merged["y_true_binary"] = _str(merged, "y_true_binary").str.lower()
    merged["y_true_multiclass"] = _str(merged, "y_true_multiclass").str.lower()
    merged = _merge_seed_metadata(merged, args.seed_csv)

    indicator_presence = _indicator_presence(merged)
    score = _num(merged, "heuristic_score")
    heur_pred = pd.Series(
        np.where(score >= args.heur_threshold, "suspicious", "legit"),
        index=merged.index,
    )

    strong_url_signal = indicator_presence.sum(axis=1).gt(0)
    hybrid = pd.Series(merged["ml_pred_binary"].astype(str), index=merged.index)
    hybrid.loc[(score >= args.hybrid_high_threshold) & (hybrid == "legit")] = (
        "suspicious"
    )
    hybrid.loc[
        (score < args.hybrid_low_threshold)
        & (hybrid != "legit")
        & (~strong_url_signal),
    ] = "legit"

    method_rows = []
    confusion_rows = []
    methods = [
        ("heuristic_only", heur_pred),
        ("ml_only", merged["ml_pred_binary"].astype(str)),
        ("heuristic_plus_ml", hybrid.astype(str)),
    ]
    for method, pred_values in methods:
        method_rows.append(
            {"method": method, **_binary_metrics(merged["y_true_binary"], pred_values)}
        )
        y_true_num = merged["y_true_binary"].astype(str).eq("suspicious").astype(int)
        y_pred_num = pred_values.astype(str).eq("suspicious").astype(int)
        confusion_rows.append(
            {
                "method": method,
                "tn": int(((y_true_num == 0) & (y_pred_num == 0)).sum()),
                "fp": int(((y_true_num == 0) & (y_pred_num == 1)).sum()),
                "fn": int(((y_true_num == 1) & (y_pred_num == 0)).sum()),
                "tp": int(((y_true_num == 1) & (y_pred_num == 1)).sum()),
            }
        )
    method_compare = pd.DataFrame(method_rows)
    confusion_df = pd.DataFrame(confusion_rows)

    indicator_summary = pd.DataFrame(
        [
            {
                "rows": int(len(merged)),
                "ip_url_messages": int(indicator_presence["ip_url"].sum()),
                "shortener_messages": int(indicator_presence["shortener_url"].sum()),
                "anchor_mismatch_messages": int(
                    indicator_presence["anchor_mismatch"].sum()
                ),
                "suspicious_tld_messages": int(
                    indicator_presence["suspicious_tld"].sum()
                ),
                "brand_typosquat_messages": int(
                    indicator_presence["brand_typosquat"].sum()
                ),
                "obfuscated_url_messages": int(
                    indicator_presence["obfuscated_url"].sum()
                ),
                "mean_heuristic_score": float(score.mean()),
            }
        ]
    )

    expected_map = expected.copy()
    expected_map["id"] = pd.to_numeric(expected_map["id"], errors="coerce").astype(
        "Int64"
    )
    merged = merged.merge(
        expected_map,
        left_on="dataset_id",
        right_on="id",
        how="left",
        suffixes=("", "_expected"),
    )
    if "id" in merged.columns:
        merged = merged.drop(columns=["id"])

    retention_rows = []
    for indicator in URL_INDICATORS:
        expected_col = f"expected_{indicator}"
        if expected_col not in merged.columns:
            continue
        exp = pd.to_numeric(merged[expected_col], errors="coerce").fillna(0).astype(int)
        det = indicator_presence[indicator].astype(int)
        tp = int(((exp == 1) & (det == 1)).sum())
        fn = int(((exp == 1) & (det == 0)).sum())
        retention_rows.append(
            {
                "indicator": indicator,
                "injected_positive": int(exp.sum()),
                "detected_positive": int(det.sum()),
                "tp": tp,
                "fn": fn,
                "injected_to_detected_retention_rate": tp / (tp + fn)
                if (tp + fn)
                else 0.0,
                "note": "Track B injected metadata (not strict per-indicator ground truth)",
            }
        )
    retention_df = pd.DataFrame(retention_rows)

    corr_rows = []
    for indicator in URL_INDICATORS:
        if (
            pd.Series(score).nunique(dropna=False) <= 1
            or indicator_presence[indicator].nunique(dropna=False) <= 1
        ):
            corr = 0.0
        else:
            corr = float(
                pd.Series(score).corr(indicator_presence[indicator], method="spearman")
            )
        corr_rows.append({"indicator": indicator, "spearman_corr_with_score": corr})
    correlation_df = pd.DataFrame(corr_rows)

    y_true_bin_num = merged["y_true_binary"].astype(str).eq("suspicious").astype(int)
    ml_pred_bin_num = merged["ml_pred_binary"].astype(str).eq("suspicious").astype(int)
    heur_pred_bin_num = heur_pred.eq("suspicious").astype(int)

    error_rows = []
    for indicator in URL_INDICATORS:
        present = indicator_presence[indicator].eq(1)
        absent = ~present
        ml_err = (ml_pred_bin_num != y_true_bin_num).astype(int)
        heur_err = (heur_pred_bin_num != y_true_bin_num).astype(int)
        error_rows.append(
            {
                "indicator": indicator,
                "rows_present": int(present.sum()),
                "rows_absent": int(absent.sum()),
                "ml_error_rate_present": float(ml_err[present].mean())
                if present.any()
                else 0.0,
                "ml_error_rate_absent": float(ml_err[absent].mean())
                if absent.any()
                else 0.0,
                "heur_error_rate_present": float(heur_err[present].mean())
                if present.any()
                else 0.0,
                "heur_error_rate_absent": float(heur_err[absent].mean())
                if absent.any()
                else 0.0,
            }
        )
    error_df = pd.DataFrame(error_rows)

    mc_summary, mc_per_class, mc_confusion = _build_multiclass_reports(
        merged["y_true_multiclass"].astype(str),
        merged["ml_pred"].astype(str),
        method="ml_only",
    )
    threshold_df = _threshold_sweep(
        score,
        merged["y_true_binary"],
        start=args.threshold_sweep_start,
        end=args.threshold_sweep_end,
        step=args.threshold_sweep_step,
    )

    merged["heur_pred_binary"] = heur_pred.astype(str)
    merged["hybrid_pred_binary"] = hybrid.astype(str)
    merged["is_ml_correct_binary"] = (
        merged["ml_pred_binary"].astype(str) == merged["y_true_binary"].astype(str)
    ).astype(int)
    merged["url_indicator_count"] = indicator_presence.sum(axis=1)
    merged["url_indicator_bucket"] = merged["url_indicator_count"].map(
        lambda n: "0" if int(n) == 0 else ("1" if int(n) == 1 else "2+")
    )
    merged["abs_score_center"] = (score - args.heur_threshold).abs()
    slice_df = _slice_metrics(merged, methods)

    qual = pd.concat(
        [
            merged[merged["is_ml_correct_binary"] == 1]
            .sort_values(
                ["url_indicator_count", "heuristic_score"], ascending=[False, False]
            )
            .head(4),
            merged[merged["is_ml_correct_binary"] == 0]
            .sort_values(
                ["url_indicator_count", "heuristic_score"], ascending=[False, False]
            )
            .head(4),
        ],
        ignore_index=True,
    )

    out_base = Path(args.output_indicator_summary_csv)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    indicator_summary.to_csv(args.output_indicator_summary_csv, index=False)
    method_compare.to_csv(args.output_method_compare_csv, index=False)
    confusion_df.to_csv(args.output_confusion_csv, index=False)
    correlation_df.to_csv(args.output_correlation_csv, index=False)
    error_df.to_csv(args.output_error_impact_csv, index=False)
    mc_summary.to_csv(
        out_base.with_name("experiment_3_campaign_multiclass_summary.csv"), index=False
    )
    mc_per_class.to_csv(
        out_base.with_name("experiment_3_campaign_multiclass_per_class.csv"),
        index=False,
    )
    mc_confusion.to_csv(
        out_base.with_name("experiment_3_campaign_multiclass_confusion_long.csv"),
        index=False,
    )
    threshold_df.to_csv(
        out_base.with_name("experiment_3_campaign_threshold_sweep.csv"), index=False
    )
    if not retention_df.empty:
        retention_df.to_csv(
            out_base.with_name(
                "experiment_3_campaign_injected_indicator_retention.csv"
            ),
            index=False,
        )
    if not slice_df.empty:
        slice_df.to_csv(
            out_base.with_name("experiment_3_campaign_slice_metrics.csv"), index=False
        )
    qual_cols = [
        "file",
        "subject",
        "attack_type",
        "generation_kind",
        "generation_family",
        "heuristic_score",
        "heuristic_reasons",
        "y_true_binary",
        "y_true_multiclass",
        "heur_pred_binary",
        "ml_pred",
        "ml_pred_binary",
        "hybrid_pred_binary",
        "is_ml_correct_binary",
        "url_indicator_count",
    ]
    qual[[c for c in qual_cols if c in qual.columns]].to_csv(
        args.output_qualitative_csv, index=False
    )

    print(f"Saved indicator summary: {args.output_indicator_summary_csv}")
    print(f"Saved method comparison: {args.output_method_compare_csv}")
    print(f"Saved confusion counts: {args.output_confusion_csv}")
    print(f"Saved score correlation: {args.output_correlation_csv}")
    print(f"Saved error impact: {args.output_error_impact_csv}")
    print(f"Saved qualitative examples: {args.output_qualitative_csv}")
    print(
        "Saved multiclass summary:",
        out_base.with_name("experiment_3_campaign_multiclass_summary.csv"),
    )
    print(
        "Saved multiclass per-class table:",
        out_base.with_name("experiment_3_campaign_multiclass_per_class.csv"),
    )
    print(
        "Saved multiclass confusion:",
        out_base.with_name("experiment_3_campaign_multiclass_confusion_long.csv"),
    )
    print(
        "Saved threshold sweep:",
        out_base.with_name("experiment_3_campaign_threshold_sweep.csv"),
    )
    if not retention_df.empty:
        print(
            "Saved injected-indicator retention:",
            out_base.with_name(
                "experiment_3_campaign_injected_indicator_retention.csv"
            ),
        )
    if not slice_df.empty:
        print(
            "Saved slice metrics:",
            out_base.with_name("experiment_3_campaign_slice_metrics.csv"),
        )


if __name__ == "__main__":
    main()
