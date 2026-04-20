from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


INDICATOR_MAP = [
    ("spf_fail", "our_spf_fail", "phishtool_spf_fail"),
    ("dkim_fail", "our_dkim_fail", "phishtool_dkim_fail"),
    ("dmarc_fail", "our_dmarc_fail", "phishtool_dmarc_fail"),
    (
        "from_reply_mismatch",
        "our_from_reply_mismatch",
        "phishtool_from_reply_mismatch",
    ),
    (
        "from_return_path_mismatch",
        "our_from_return_path_mismatch",
        "phishtool_from_return_path_mismatch",
    ),
    (
        "message_id_domain_mismatch",
        "our_message_id_domain_mismatch",
        "phishtool_message_id_domain_mismatch",
    ),
    ("received_anomaly", "our_received_anomaly", "phishtool_received_anomaly"),
    ("display_name_spoof", "our_display_name_spoof", "phishtool_display_name_spoof"),
    ("ip_url", "our_ip_url", "phishtool_ip_url"),
    ("shortener_url", "our_shortener_url", "phishtool_shortener_url"),
    ("anchor_mismatch", "our_anchor_mismatch", "phishtool_anchor_mismatch"),
    ("suspicious_tld", "our_suspicious_tld", "phishtool_suspicious_tld"),
    ("brand_typosquat", "our_brand_typosquat", "phishtool_brand_typosquat"),
    ("obfuscated_url", "our_obfuscated_url", "phishtool_obfuscated_url"),
    ("risky_attachment", "our_risky_attachment", "phishtool_risky_attachment"),
]


def _as_bool01(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    true_tokens = {"1", "true", "yes", "y", "t"}
    false_tokens = {"0", "false", "no", "n", "f", "", "nan", "none", "<na>"}

    out = pd.Series([pd.NA] * len(s), index=s.index, dtype="Int64")
    out.loc[s.isin(true_tokens)] = 1
    out.loc[s.isin(false_tokens)] = 0
    numeric = pd.to_numeric(s, errors="coerce")
    out.loc[numeric.notna()] = (numeric.loc[numeric.notna()] > 0).astype(int)
    return out


def _decision_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    yt = y_true.astype(str).str.lower().eq("suspicious").astype(int)
    yp = y_pred.astype(str).str.lower().eq("suspicious").astype(int)
    return {
        "rows": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Exp6 vs PhishTool comparison"
    )
    parser.add_argument(
        "--our-ioc-csv",
        default="results/experiment_6/exp6_our_ioc_table.csv",
    )
    parser.add_argument(
        "--phishtool-csv",
        default="results/experiment_6/exp6_phishtool_filled.csv",
        help="Filled template with phishtool_* columns",
    )
    parser.add_argument(
        "--output-overlap-long-csv",
        default="results/experiment_6/exp6_ioc_overlap_long.csv",
    )
    parser.add_argument(
        "--output-overlap-summary-csv",
        default="results/experiment_6/exp6_ioc_overlap_summary.csv",
    )
    parser.add_argument(
        "--output-capability-summary-csv",
        default="results/experiment_6/exp6_capability_summary.csv",
    )
    parser.add_argument(
        "--output-decision-compare-csv",
        default="results/experiment_6/exp6_decision_compare.csv",
    )
    parser.add_argument(
        "--output-decision-summary-csv",
        default="results/experiment_6/exp6_decision_summary.csv",
    )
    parser.add_argument(
        "--output-case-study-csv",
        default="results/experiment_6/exp6_case_studies.csv",
    )
    args = parser.parse_args()

    our = pd.read_csv(args.our_ioc_csv, low_memory=False)
    ph = pd.read_csv(args.phishtool_csv, low_memory=False)

    merged = our.merge(ph, on="exp6_id", how="left", suffixes=("", "_ph"))

    overlap_rows = []
    summary_rows = []
    capability_rows = []

    for indicator, our_col, ph_col in INDICATOR_MAP:
        if ph_col not in merged.columns:
            print(f"[WARN] Missing column in PhishTool file: {ph_col}")
            continue
        our_val = pd.to_numeric(merged[our_col], errors="coerce").fillna(0).astype(int)
        ph_val = _as_bool01(merged[ph_col])
        valid = ph_val.notna()
        if not valid.any():
            print(f"[WARN] No filled rows for indicator: {indicator}")
            continue

        comp = pd.DataFrame(
            {
                "exp6_id": merged["exp6_id"],
                "benchmark_source": merged["benchmark_source"],
                "benchmark_bucket": merged["benchmark_bucket"],
                "indicator": indicator,
                "our_detected": our_val,
                "phishtool_detected": ph_val,
            }
        )
        comp = comp[valid].copy()
        comp["match"] = (comp["our_detected"] == comp["phishtool_detected"]).astype(int)
        overlap_rows.append(comp)

        tp = int(
            ((comp["our_detected"] == 1) & (comp["phishtool_detected"] == 1)).sum()
        )
        tn = int(
            ((comp["our_detected"] == 0) & (comp["phishtool_detected"] == 0)).sum()
        )
        fp = int(
            ((comp["our_detected"] == 1) & (comp["phishtool_detected"] == 0)).sum()
        )
        fn = int(
            ((comp["our_detected"] == 0) & (comp["phishtool_detected"] == 1)).sum()
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        acc = float((comp["our_detected"] == comp["phishtool_detected"]).mean())

        summary_rows.append(
            {
                "indicator": indicator,
                "rows": int(len(comp)),
                "match_rate": acc,
                "precision_vs_phishtool": precision,
                "recall_vs_phishtool": recall,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )
        capability_rows.append(
            {
                "indicator": indicator,
                "our_coverage_rate": float(comp["our_detected"].mean()),
                "phishtool_coverage_rate": float(comp["phishtool_detected"].mean()),
                "overlap_match_rate": acc,
            }
        )

    overlap_df = (
        pd.concat(overlap_rows, ignore_index=True) if overlap_rows else pd.DataFrame()
    )
    summary_df = pd.DataFrame(summary_rows)
    capability_df = pd.DataFrame(capability_rows)

    # Decision-level comparison
    our_decision = merged["our_pred_binary_ml"].astype(str).str.lower()
    missing_ml = our_decision.isin(["", "nan", "none", "<na>"])
    our_decision.loc[missing_ml] = (
        merged.loc[missing_ml, "our_pred_binary_heur"].astype(str).str.lower()
    )

    ph_decision = merged.get("phishtool_pred_binary", pd.Series([""] * len(merged)))
    ph_decision = (
        ph_decision.astype(str).str.lower().replace({"nan": "", "none": "", "<na>": ""})
    )
    valid_decision = ph_decision.ne("")

    decision_df = merged[
        [
            "exp6_id",
            "benchmark_layer",
            "benchmark_source",
            "benchmark_bucket",
            "y_true_binary",
        ]
    ].copy()
    decision_df["our_pred_binary"] = our_decision
    decision_df["phishtool_pred_binary"] = ph_decision
    decision_df = decision_df[valid_decision].copy()
    decision_df["agreement"] = (
        decision_df["our_pred_binary"] == decision_df["phishtool_pred_binary"]
    ).astype(int)

    if not decision_df.empty:
        decision_summary = pd.DataFrame(
            [
                {
                    "rows": int(len(decision_df)),
                    "agreement_rate": float(decision_df["agreement"].mean()),
                    **{
                        f"our_vs_truth_{k}": v
                        for k, v in _decision_metrics(
                            decision_df["y_true_binary"], decision_df["our_pred_binary"]
                        ).items()
                    },
                    **{
                        f"phishtool_vs_truth_{k}": v
                        for k, v in _decision_metrics(
                            decision_df["y_true_binary"],
                            decision_df["phishtool_pred_binary"],
                        ).items()
                    },
                }
            ]
        )
    else:
        decision_summary = pd.DataFrame(
            [
                {
                    "rows": 0,
                    "agreement_rate": 0.0,
                }
            ]
        )

    case_studies = pd.DataFrame()
    if not overlap_df.empty:
        mismatch_counts = (
            overlap_df.assign(mismatch=(1 - overlap_df["match"]))
            .groupby("exp6_id", as_index=False)["mismatch"]
            .sum()
            .rename(columns={"mismatch": "ioc_mismatch_count"})
        )
        case_studies = merged.merge(mismatch_counts, on="exp6_id", how="left")
        case_studies["ioc_mismatch_count"] = (
            pd.to_numeric(case_studies["ioc_mismatch_count"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        case_studies = case_studies.sort_values(
            ["ioc_mismatch_count", "our_heuristic_score"], ascending=[False, False]
        ).head(8)
        case_studies = case_studies[
            [
                "exp6_id",
                "benchmark_source",
                "benchmark_bucket",
                "subject",
                "y_true_binary",
                "our_pred_binary_heur",
                "our_pred_binary_ml",
                "ioc_mismatch_count",
                "our_heuristic_reasons",
            ]
        ]

    out_dir = Path(args.output_overlap_long_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    overlap_df.to_csv(args.output_overlap_long_csv, index=False)
    summary_df.to_csv(args.output_overlap_summary_csv, index=False)
    capability_df.to_csv(args.output_capability_summary_csv, index=False)
    decision_df.to_csv(args.output_decision_compare_csv, index=False)
    decision_summary.to_csv(args.output_decision_summary_csv, index=False)
    case_studies.to_csv(args.output_case_study_csv, index=False)

    print(f"Saved IOC overlap long: {args.output_overlap_long_csv}")
    print(f"Saved IOC overlap summary: {args.output_overlap_summary_csv}")
    print(f"Saved capability summary: {args.output_capability_summary_csv}")
    print(f"Saved decision compare: {args.output_decision_compare_csv}")
    print(f"Saved decision summary: {args.output_decision_summary_csv}")
    print(f"Saved case studies: {args.output_case_study_csv}")


if __name__ == "__main__":
    main()
