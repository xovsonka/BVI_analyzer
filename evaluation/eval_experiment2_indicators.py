from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


INDICATORS = [
    (
        "from_reply_mismatch",
        "expected_from_reply_mismatch",
        "detected_from_reply_mismatch",
    ),
    (
        "from_return_path_mismatch",
        "expected_from_return_path_mismatch",
        "detected_from_return_path_mismatch",
    ),
    ("spf_fail", "expected_spf_fail", "detected_spf_fail"),
    ("dkim_fail", "expected_dkim_fail", "detected_dkim_fail"),
    ("dmarc_fail", "expected_dmarc_fail", "detected_dmarc_fail"),
    (
        "received_anomaly",
        "expected_received_anomaly",
        "detected_received_anomaly",
    ),
    (
        "message_id_domain_mismatch",
        "expected_message_id_domain_mismatch",
        "detected_message_id_domain_mismatch",
    ),
    (
        "display_name_spoof_flag",
        "expected_display_name_spoof_flag",
        "detected_display_name_spoof_flag",
    ),
    (
        "trusted_domain_only",
        "expected_trusted_domain_only",
        "detected_trusted_domain_only",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Experiment 2 header/auth indicator detection"
    )
    parser.add_argument(
        "--analyzed-csv",
        default="results/experiment_2_header_auth_model_input.csv",
        help="CSV produced by analysis/analyze_campaign.py",
    )
    parser.add_argument(
        "--expected-csv",
        default="dataset/experiment_2/header_auth_expected.csv",
        help="Expected indicator table from generator",
    )
    parser.add_argument(
        "--output-detail-csv",
        default="results/experiment_2_indicator_detail.csv",
    )
    parser.add_argument(
        "--output-summary-csv",
        default="results/experiment_2_indicator_summary.csv",
    )
    parser.add_argument(
        "--output-summary-txt",
        default="results/experiment_2_indicator_summary.txt",
    )
    args = parser.parse_args()

    analyzed = pd.read_csv(args.analyzed_csv, low_memory=False)
    expected = pd.read_csv(args.expected_csv, low_memory=False)

    analyzed = analyzed.copy()
    analyzed["file_key"] = analyzed["file"].astype(str).map(lambda p: Path(p).name)
    expected = expected.copy()
    expected["file_key"] = expected["file"].astype(str).map(lambda p: Path(p).name)

    merged = expected.merge(analyzed, on="file_key", how="left", suffixes=("_exp", ""))

    def _int_col(name: str) -> pd.Series:
        if name not in merged.columns:
            return pd.Series([0] * len(merged), index=merged.index, dtype=int)
        return pd.to_numeric(merged[name], errors="coerce").fillna(0).astype(int)

    def _str_col(name: str, default: str = "none") -> pd.Series:
        if name not in merged.columns:
            return pd.Series([default] * len(merged), index=merged.index, dtype=str)
        return merged[name].astype(str)

    merged["detected_from_reply_mismatch"] = _int_col("from_reply_mismatch")
    merged["detected_from_return_path_mismatch"] = _int_col("from_return_path_mismatch")
    merged["detected_spf_fail"] = (
        _str_col("spf_result").str.lower().eq("fail").astype(int)
    )
    merged["detected_dkim_fail"] = (
        _str_col("dkim_result").str.lower().eq("fail").astype(int)
    )
    merged["detected_dmarc_fail"] = (
        _str_col("dmarc_result").str.lower().eq("fail").astype(int)
    )
    merged["detected_received_anomaly"] = _int_col("received_anomaly_flag")
    merged["detected_message_id_domain_mismatch"] = _int_col(
        "message_id_domain_mismatch"
    )
    merged["detected_display_name_spoof_flag"] = _int_col("display_name_spoof_flag")
    merged["detected_trusted_domain_only"] = _int_col("trusted_domain_only")

    detail_rows: list[dict] = []
    summary_rows: list[dict] = []

    for indicator_name, expected_col, detected_col in INDICATORS:
        exp_val = (
            pd.to_numeric(merged[expected_col], errors="coerce").fillna(0).astype(int)
        )
        det_val = (
            pd.to_numeric(merged[detected_col], errors="coerce").fillna(0).astype(int)
        )

        correct = (exp_val == det_val).astype(int)
        tp = int(((exp_val == 1) & (det_val == 1)).sum())
        tn = int(((exp_val == 0) & (det_val == 0)).sum())
        fp = int(((exp_val == 0) & (det_val == 1)).sum())
        fn = int(((exp_val == 1) & (det_val == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = float(correct.mean()) if len(correct) else 0.0

        summary_rows.append(
            {
                "indicator": indicator_name,
                "rows": int(len(correct)),
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
                    "expected_indicator": int(row.get(expected_col, 0) or 0),
                    "detected_indicator": int(row.get(detected_col, 0) or 0),
                    "correct": int(
                        int(row.get(expected_col, 0) or 0)
                        == int(row.get(detected_col, 0) or 0)
                    ),
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("indicator")

    Path(args.output_detail_csv).parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(args.output_detail_csv, index=False)
    summary_df.to_csv(args.output_summary_csv, index=False)

    with Path(args.output_summary_txt).open("w", encoding="utf-8") as handle:
        handle.write("Experiment 2 indicator detection summary\n")
        handle.write("=" * 72 + "\n")
        handle.write(summary_df.to_string(index=False))
        handle.write("\n")

    print(f"Saved detail table: {args.output_detail_csv}")
    print(f"Saved summary table: {args.output_summary_csv}")
    print(f"Saved summary text: {args.output_summary_txt}")


if __name__ == "__main__":
    main()
