from __future__ import annotations

import argparse
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


INDICATORS = [
    ("ip_url", "expected_ip_url", "detected_ip_url"),
    ("shortener_url", "expected_shortener_url", "detected_shortener_url"),
    ("anchor_mismatch", "expected_anchor_mismatch", "detected_anchor_mismatch"),
    ("suspicious_tld", "expected_suspicious_tld", "detected_suspicious_tld"),
    ("brand_typosquat", "expected_brand_typosquat", "detected_brand_typosquat"),
    ("obfuscated_url", "expected_obfuscated_url", "detected_obfuscated_url"),
]

URL_REASON_TOKENS = {
    "ip_url",
    "shortener_url",
    "anchor_mismatch",
    "suspicious_tld",
    "brand_typosquat",
    "obfuscated_url",
}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([0] * len(df), index=df.index, dtype=int)
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


def _str(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=str)
    return df[col].astype(str)


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true_num = y_true.astype(str).str.lower().eq("suspicious").astype(int)
    y_pred_num = y_pred.astype(str).str.lower().eq("suspicious").astype(int)
    return {
        "rows": int(len(y_true_num)),
        "accuracy": float(accuracy_score(y_true_num, y_pred_num)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_num, y_pred_num)),
        "precision": float(precision_score(y_true_num, y_pred_num, zero_division=0)),
        "recall": float(recall_score(y_true_num, y_pred_num, zero_division=0)),
        "f1": float(f1_score(y_true_num, y_pred_num, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Experiment 3 URL indicators (Track A)"
    )
    parser.add_argument(
        "--analyzed-csv",
        default="results/experiment_3_url_model_input.csv",
    )
    parser.add_argument(
        "--expected-csv",
        default="dataset/experiment_3/url_expected.csv",
    )
    parser.add_argument("--heur-threshold", type=int, default=20)
    parser.add_argument("--threshold-sweep-start", type=int, default=0)
    parser.add_argument("--threshold-sweep-end", type=int, default=80)
    parser.add_argument("--threshold-sweep-step", type=int, default=1)
    parser.add_argument(
        "--output-detail-csv",
        default="results/experiment_3_url_indicator_detail.csv",
    )
    parser.add_argument(
        "--output-indicator-summary-csv",
        default="results/experiment_3_url_indicator_summary.csv",
    )
    parser.add_argument(
        "--output-binary-summary-csv",
        default="results/experiment_3_url_binary_summary.csv",
    )
    parser.add_argument(
        "--output-explainability-csv",
        default="results/experiment_3_url_explainability.csv",
    )
    parser.add_argument(
        "--output-score-by-case-csv",
        default="results/experiment_3_url_score_by_case.csv",
    )
    parser.add_argument(
        "--output-score-by-label-csv",
        default="results/experiment_3_url_score_by_label.csv",
    )
    parser.add_argument(
        "--output-summary-txt",
        default="results/experiment_3_url_summary.txt",
    )
    parser.add_argument(
        "--output-threshold-sweep-csv",
        default="results/experiment_3_url_threshold_sweep.csv",
    )
    args = parser.parse_args()

    analyzed = pd.read_csv(args.analyzed_csv, low_memory=False)
    expected = pd.read_csv(args.expected_csv, low_memory=False)

    analyzed = analyzed.copy()
    expected = expected.copy()
    analyzed["file_key"] = analyzed["file"].astype(str).map(lambda p: Path(p).name)
    expected["file_key"] = expected["file"].astype(str).map(lambda p: Path(p).name)

    merged = expected.merge(analyzed, on="file_key", how="left", suffixes=("_exp", ""))

    merged["detected_ip_url"] = _num(merged, "ip_url_count").gt(0).astype(int)
    merged["detected_shortener_url"] = (
        _num(merged, "shortener_url_count").gt(0).astype(int)
    )
    merged["detected_anchor_mismatch"] = (
        _num(merged, "mismatched_anchor_count").gt(0).astype(int)
    )
    merged["detected_suspicious_tld"] = (
        _num(merged, "suspicious_tld_count").gt(0).astype(int)
    )
    merged["detected_brand_typosquat"] = (
        _num(merged, "brand_typosquat_flag").gt(0).astype(int)
    )
    merged["detected_obfuscated_url"] = (
        _num(merged, "obfuscated_url_count").gt(0).astype(int)
    )
    if merged["detected_obfuscated_url"].sum() == 0:
        merged["detected_obfuscated_url"] = (
            _str(merged, "heuristic_reasons").str.contains(
                "obfuscated_url", regex=False
            )
        ).astype(int)

    score = _num(merged, "heuristic_score")
    merged["detected_binary_label"] = "legit"
    merged.loc[score >= args.heur_threshold, "detected_binary_label"] = "suspicious"

    detail_rows: list[dict] = []
    indicator_rows: list[dict] = []
    for indicator_name, exp_col, det_col in INDICATORS:
        exp_val = _num(merged, exp_col)
        det_val = _num(merged, det_col)

        tp = int(((exp_val == 1) & (det_val == 1)).sum())
        tn = int(((exp_val == 0) & (det_val == 0)).sum())
        fp = int(((exp_val == 0) & (det_val == 1)).sum())
        fn = int(((exp_val == 1) & (det_val == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = float((exp_val == det_val).mean()) if len(exp_val) else 0.0
        indicator_rows.append(
            {
                "indicator": indicator_name,
                "rows": int(len(exp_val)),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )

        for _, row in merged.iterrows():
            detail_rows.append(
                {
                    "file": row.get("file_exp", row.get("file", "")),
                    "case_id": row.get("case_id", ""),
                    "indicator": indicator_name,
                    "expected_indicator": int(row.get(exp_col, 0) or 0),
                    "detected_indicator": int(row.get(det_col, 0) or 0),
                    "correct": int(
                        int(row.get(exp_col, 0) or 0) == int(row.get(det_col, 0) or 0)
                    ),
                }
            )

    binary_summary = pd.DataFrame(
        [
            {
                "task": "heuristic_binary_legit_vs_suspicious",
                **_binary_metrics(
                    _str(merged, "expected_binary_label", "legit"),
                    _str(merged, "detected_binary_label", "legit"),
                ),
            }
        ]
    )

    merged["detected_reasons"] = _str(merged, "heuristic_reasons", "")

    def expected_reason_list(value: str) -> list[str]:
        text = (value or "").strip()
        if text.lower() == "nan":
            return []
        if not text:
            return []
        return [item.strip() for item in text.split("|") if item.strip()]

    explain_rows: list[dict] = []
    for _, row in merged.iterrows():
        expected_reasons = expected_reason_list(str(row.get("expected_reasons", "")))
        detected_reasons = {
            item.strip()
            for item in str(row.get("detected_reasons", "")).split(",")
            if item.strip()
        }
        detected_url_reasons = {r for r in detected_reasons if r in URL_REASON_TOKENS}
        expected_set = set(expected_reasons)
        matched = [
            reason for reason in expected_reasons if reason in detected_url_reasons
        ]
        unexpected_url_reasons = sorted(detected_url_reasons - expected_set)
        unexpected_non_url_reasons = sorted(detected_reasons - detected_url_reasons)
        expected_count = len(expected_reasons)
        match_count = len(matched)
        coverage = 1.0 if expected_count == 0 else float(match_count / expected_count)
        exact_match = int(expected_set == detected_url_reasons)
        partial_match = int(
            expected_count > 0 and match_count >= max(1, expected_count // 2)
        )
        explain_rows.append(
            {
                "file": row.get("file_exp", row.get("file", "")),
                "case_id": row.get("case_id", ""),
                "expected_reasons": "|".join(expected_reasons),
                "detected_reasons": ",".join(sorted(detected_reasons)),
                "detected_url_reasons": ",".join(sorted(detected_url_reasons)),
                "unexpected_url_reasons": ",".join(unexpected_url_reasons),
                "unexpected_non_url_reasons": ",".join(unexpected_non_url_reasons),
                "expected_reason_count": expected_count,
                "matched_reason_count": match_count,
                "unexpected_url_reason_count": len(unexpected_url_reasons),
                "unexpected_non_url_reason_count": len(unexpected_non_url_reasons),
                "reason_coverage": coverage,
                "any_expected_reason_matched": int(
                    (expected_count == 0) or (match_count >= 1)
                ),
                "all_expected_reasons_matched": int(
                    (expected_count == 0) or (match_count == expected_count)
                ),
                "exact_reason_match": int((expected_count == 0) or bool(exact_match)),
                "strict_exact_reason_match": int(bool(exact_match)),
                "partial_reason_match": int(
                    (expected_count == 0) or bool(partial_match)
                ),
            }
        )

    explain_df = pd.DataFrame(explain_rows)
    explain_summary = pd.DataFrame(
        [
            {
                "rows": int(len(explain_df)),
                "any_expected_reason_match_rate": float(
                    explain_df["any_expected_reason_matched"].mean()
                ),
                "all_expected_reasons_match_rate": float(
                    explain_df["all_expected_reasons_matched"].mean()
                ),
                "exact_reason_match_rate": float(
                    explain_df["exact_reason_match"].mean()
                ),
                "strict_exact_reason_match_rate": float(
                    explain_df["strict_exact_reason_match"].mean()
                ),
                "partial_reason_match_rate": float(
                    explain_df["partial_reason_match"].mean()
                ),
                "mean_reason_coverage": float(explain_df["reason_coverage"].mean()),
                "mean_unexpected_url_reason_count": float(
                    explain_df["unexpected_url_reason_count"].mean()
                ),
                "mean_unexpected_non_url_reason_count": float(
                    explain_df["unexpected_non_url_reason_count"].mean()
                ),
                "missing_reason_rate": float(
                    1.0 - explain_df["all_expected_reasons_matched"].mean()
                ),
            }
        ]
    )

    threshold_rows = []
    expected_binary = _str(merged, "expected_binary_label", "legit")
    for threshold in range(
        args.threshold_sweep_start,
        args.threshold_sweep_end + 1,
        args.threshold_sweep_step,
    ):
        pred = pd.Series(
            np.where(score >= threshold, "suspicious", "legit"), index=merged.index
        )
        threshold_rows.append(
            {
                "threshold": threshold,
                **_binary_metrics(expected_binary, pred),
            }
        )
    threshold_df = pd.DataFrame(threshold_rows)
    if not threshold_df.empty:
        threshold_df["is_best_f1"] = (
            threshold_df["f1"].eq(float(threshold_df["f1"].max())).astype(int)
        )
        threshold_df["is_best_balanced_accuracy"] = (
            threshold_df["balanced_accuracy"]
            .eq(float(threshold_df["balanced_accuracy"].max()))
            .astype(int)
        )

    score_by_case = (
        merged.assign(heuristic_score=score)
        .groupby("case_id", as_index=False)
        .agg(
            rows=("heuristic_score", "count"),
            mean=("heuristic_score", "mean"),
            std=("heuristic_score", "std"),
            min=("heuristic_score", "min"),
            max=("heuristic_score", "max"),
        )
    )

    score_by_label = (
        merged.assign(heuristic_score=score)
        .groupby("expected_binary_label", as_index=False)
        .agg(
            rows=("heuristic_score", "count"),
            mean=("heuristic_score", "mean"),
            std=("heuristic_score", "std"),
            min=("heuristic_score", "min"),
            max=("heuristic_score", "max"),
        )
    )

    indicator_df = pd.DataFrame(indicator_rows).sort_values("indicator")
    detail_df = pd.DataFrame(detail_rows)

    Path(args.output_detail_csv).parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(args.output_detail_csv, index=False)
    indicator_df.to_csv(args.output_indicator_summary_csv, index=False)
    binary_summary.to_csv(args.output_binary_summary_csv, index=False)
    explain_df.to_csv(args.output_explainability_csv, index=False)
    explain_summary.to_csv(
        Path(args.output_explainability_csv).with_name(
            "experiment_3_url_explainability_summary.csv"
        ),
        index=False,
    )
    score_by_case.to_csv(args.output_score_by_case_csv, index=False)
    score_by_label.to_csv(args.output_score_by_label_csv, index=False)
    threshold_df.to_csv(args.output_threshold_sweep_csv, index=False)

    with Path(args.output_summary_txt).open("w", encoding="utf-8") as handle:
        handle.write("Experiment 3 URL indicator summary\n")
        handle.write("=" * 72 + "\n")
        handle.write("[Per-indicator]\n")
        handle.write(indicator_df.to_string(index=False))
        handle.write("\n\n[Binary legit vs suspicious]\n")
        handle.write(binary_summary.to_string(index=False))
        if not threshold_df.empty:
            handle.write("\n\n[Threshold sweep]\n")
            handle.write(
                threshold_df[
                    [
                        "threshold",
                        "accuracy",
                        "balanced_accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "is_best_f1",
                        "is_best_balanced_accuracy",
                    ]
                ].to_string(index=False)
            )
        handle.write("\n\n[Explainability]\n")
        handle.write(explain_summary.to_string(index=False))
        handle.write("\n")

    print(f"Saved detail table: {args.output_detail_csv}")
    print(f"Saved indicator summary: {args.output_indicator_summary_csv}")
    print(f"Saved binary summary: {args.output_binary_summary_csv}")
    print(f"Saved explainability table: {args.output_explainability_csv}")
    print(
        "Saved explainability summary:",
        Path(args.output_explainability_csv).with_name(
            "experiment_3_url_explainability_summary.csv"
        ),
    )
    print(f"Saved score by case: {args.output_score_by_case_csv}")
    print(f"Saved score by binary label: {args.output_score_by_label_csv}")
    print(f"Saved threshold sweep: {args.output_threshold_sweep_csv}")
    print(f"Saved summary text: {args.output_summary_txt}")


if __name__ == "__main__":
    main()
