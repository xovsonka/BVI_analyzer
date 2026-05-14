from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _file_key(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda x: Path(x).name)


def _resolve_file(path_str: str) -> Path:
    raw = Path(str(path_str))
    if raw.exists():
        return raw
    joined = PROJECT_ROOT / raw
    if joined.exists():
        return joined
    return raw


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        return df.head(0).copy()
    if len(df) >= n:
        return df.sample(n=n, random_state=seed)
    print(
        f"[WARN] Requested {n} rows, only {len(df)} available; sampling with replacement"
    )
    return df.sample(n=n, random_state=seed, replace=True)


def _load_campaign(
    analyzed_csv: Path, pred_csv: Path, source_name: str
) -> pd.DataFrame:
    analyzed = pd.read_csv(analyzed_csv, low_memory=False)
    pred = pd.read_csv(pred_csv, low_memory=False)

    analyzed = analyzed.copy()
    pred = pred.copy()
    analyzed["file_key"] = _file_key(analyzed["file"])
    pred["file_key"] = _file_key(pred["file"])

    keep = [
        "file_key",
        "y_true_multiclass",
        "y_true_binary",
        "ml_pred",
        "ml_pred_binary",
    ]
    pred = pred[keep].drop_duplicates(subset=["file_key"])
    merged = analyzed.merge(pred, on="file_key", how="inner")
    merged["benchmark_source"] = source_name
    merged["benchmark_layer"] = "realistic_campaign"
    return merged


def _load_exp2_tracka(analyzed_csv: Path, expected_csv: Path) -> pd.DataFrame:
    analyzed = pd.read_csv(analyzed_csv, low_memory=False)
    expected = pd.read_csv(expected_csv, low_memory=False)

    analyzed["file_key"] = _file_key(analyzed["file"])
    expected["file_key"] = _file_key(expected["file"])

    cols = [
        "expected_from_reply_mismatch",
        "expected_from_return_path_mismatch",
        "expected_spf_fail",
        "expected_dkim_fail",
        "expected_dmarc_fail",
        "expected_received_anomaly",
        "expected_message_id_domain_mismatch",
        "expected_display_name_spoof_flag",
    ]
    exp = expected[["file_key", "case_id", *cols]].copy()
    for c in cols:
        exp[c] = pd.to_numeric(exp[c], errors="coerce").fillna(0).astype(int)
    exp["is_header_heavy"] = exp[cols].sum(axis=1) > 0
    exp["y_true_binary"] = "legit"
    exp.loc[exp["is_header_heavy"], "y_true_binary"] = "suspicious"

    merged = analyzed.merge(exp, on="file_key", how="inner")
    merged["benchmark_source"] = "exp2_track_a"
    merged["benchmark_layer"] = "controlled_benchmark"
    merged["y_true_multiclass"] = pd.NA
    return merged


def _load_exp3_tracka(analyzed_csv: Path, expected_csv: Path) -> pd.DataFrame:
    analyzed = pd.read_csv(analyzed_csv, low_memory=False)
    expected = pd.read_csv(expected_csv, low_memory=False)

    analyzed["file_key"] = _file_key(analyzed["file"])
    expected["file_key"] = _file_key(expected["file"])

    exp = expected[["file_key", "case_id", "expected_binary_label"]].copy()
    exp["y_true_binary"] = exp["expected_binary_label"].astype(str).str.lower()
    merged = analyzed.merge(exp, on="file_key", how="inner")
    merged["benchmark_source"] = "exp3_track_a"
    merged["benchmark_layer"] = "controlled_benchmark"
    merged["y_true_multiclass"] = pd.NA
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Experiment 6 representative EML set"
    )
    parser.add_argument(
        "--exp1-analyzed-csv",
        default="results/experiment_5/exp1_campaign_model_input.csv",
    )
    parser.add_argument(
        "--exp1-pred-csv",
        default="results/retuned_exp1234/semantic_cta_eval/exp1_spamboost_ps160_fs160_pred.csv",
    )
    parser.add_argument(
        "--exp2-b-analyzed-csv",
        default="results/eperiment_2_results/experiment_2_campaign_model_input.csv",
    )
    parser.add_argument(
        "--exp2-b-pred-csv",
        default="results/retuned_exp1234/semantic_cta_eval/exp2_spamboost_ps130_fs160_pred.csv",
    )
    parser.add_argument(
        "--exp3-b-analyzed-csv",
        default="results/experiment_3_result/experiment_3_campaign_model_input.csv",
    )
    parser.add_argument(
        "--exp3-b-pred-csv",
        default="results/retuned_exp1234/semantic_cta_eval/exp3_ps160_fs160_pred.csv",
    )
    parser.add_argument(
        "--exp2-a-analyzed-csv",
        default="results/eperiment_2_results/experiment_2_header_auth_model_input.csv",
    )
    parser.add_argument(
        "--exp2-a-expected-csv",
        default="dataset/experiment_2/header_auth_expected.csv",
    )
    parser.add_argument(
        "--exp3-a-analyzed-csv",
        default="results/experiment_3_result/experiment_3_url_model_input.csv",
    )
    parser.add_argument(
        "--exp3-a-expected-csv",
        default="dataset/experiment_3/url_expected.csv",
    )
    parser.add_argument("--n-legit", type=int, default=15)
    parser.add_argument("--n-phishing", type=int, default=15)
    parser.add_argument("--n-spam", type=int, default=10)
    parser.add_argument("--n-financial-fraud", type=int, default=10)
    parser.add_argument("--n-header-heavy", type=int, default=10)
    parser.add_argument("--n-url-heavy", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-selection-csv",
        default="results/experiment_6/exp6_selection.csv",
    )
    parser.add_argument(
        "--output-summary-csv",
        default="results/experiment_6/exp6_selection_summary.csv",
    )
    parser.add_argument(
        "--output-phishtool-template-csv",
        default="results/experiment_6/exp6_phishtool_template.csv",
    )
    parser.add_argument(
        "--output-eml-dir",
        default="dataset/experiment_6/eml_selected",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing *.eml files in output directory before copy",
    )
    parser.add_argument("--copy-eml", action="store_true")
    args = parser.parse_args()

    exp1 = _load_campaign(
        Path(args.exp1_analyzed_csv),
        Path(args.exp1_pred_csv),
        "exp1_campaign",
    )
    exp2b = _load_campaign(
        Path(args.exp2_b_analyzed_csv),
        Path(args.exp2_b_pred_csv),
        "exp2_track_b",
    )
    exp3b = _load_campaign(
        Path(args.exp3_b_analyzed_csv),
        Path(args.exp3_b_pred_csv),
        "exp3_track_b",
    )
    campaign_all = pd.concat([exp1, exp2b, exp3b], ignore_index=True)
    campaign_all["y_true_multiclass"] = (
        campaign_all["y_true_multiclass"].astype(str).str.lower()
    )

    exp2a = _load_exp2_tracka(
        Path(args.exp2_a_analyzed_csv), Path(args.exp2_a_expected_csv)
    )
    exp3a = _load_exp3_tracka(
        Path(args.exp3_a_analyzed_csv), Path(args.exp3_a_expected_csv)
    )

    picked = []
    class_targets = [
        ("legit", args.n_legit),
        ("phishing", args.n_phishing),
        ("spam", args.n_spam),
        ("financial_fraud", args.n_financial_fraud),
    ]
    for i, (label, n) in enumerate(class_targets):
        subset = campaign_all[campaign_all["y_true_multiclass"] == label].copy()
        part = _sample(subset, n, args.seed + i)
        part["benchmark_bucket"] = label
        picked.append(part)

    header_subset = exp2a[exp2a["is_header_heavy"]].copy()
    header_part = _sample(header_subset, args.n_header_heavy, args.seed + 100)
    header_part["benchmark_bucket"] = "spoofing_header_heavy"
    picked.append(header_part)

    url_part = _sample(
        exp3a[exp3a["y_true_binary"].astype(str).str.lower().eq("suspicious")].copy(),
        args.n_url_heavy,
        args.seed + 101,
    )
    url_part["benchmark_bucket"] = "url_heavy"
    picked.append(url_part)

    out = pd.concat(picked, ignore_index=True)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    out["exp6_id"] = [f"exp6_{i + 1:03d}" for i in range(len(out))]

    out["file"] = out["file"].astype(str)
    out["resolved_file"] = out["file"].map(_resolve_file)
    out["file_exists"] = (
        out["resolved_file"].map(lambda p: Path(p).exists()).astype(int)
    )

    eml_dir = Path(args.output_eml_dir)
    eml_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        for old in eml_dir.glob("*.eml"):
            old.unlink()
    copied_paths = []
    for _, row in out.iterrows():
        src = Path(str(row["resolved_file"]))
        dst = eml_dir / f"{row['exp6_id']}__{src.name}"
        if args.copy_eml and src.exists():
            shutil.copy2(src, dst)
        copied_paths.append(str(dst if args.copy_eml else src))
    out["exp6_eml_path"] = copied_paths

    sel_cols = [
        "exp6_id",
        "benchmark_layer",
        "benchmark_source",
        "benchmark_bucket",
        "file",
        "exp6_eml_path",
        "subject",
        "sender",
        "receiver",
        "y_true_binary",
        "y_true_multiclass",
        "heuristic_score",
        "heuristic_reasons",
        "file_exists",
    ]
    out_final = out[sel_cols].copy()

    out_path = Path(args.output_selection_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_final.to_csv(out_path, index=False)

    summary = (
        out_final.groupby(["benchmark_layer", "benchmark_bucket"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    summary.to_csv(Path(args.output_summary_csv), index=False)

    phishtool_template = out_final[
        [
            "exp6_id",
            "benchmark_layer",
            "benchmark_source",
            "benchmark_bucket",
            "exp6_eml_path",
            "y_true_binary",
            "y_true_multiclass",
        ]
    ].copy()
    for c in [
        "phishtool_spf_fail",
        "phishtool_dkim_fail",
        "phishtool_dmarc_fail",
        "phishtool_from_reply_mismatch",
        "phishtool_from_return_path_mismatch",
        "phishtool_message_id_domain_mismatch",
        "phishtool_received_anomaly",
        "phishtool_display_name_spoof",
        "phishtool_ip_url",
        "phishtool_shortener_url",
        "phishtool_anchor_mismatch",
        "phishtool_suspicious_tld",
        "phishtool_brand_typosquat",
        "phishtool_obfuscated_url",
        "phishtool_risky_attachment",
        "phishtool_pred_binary",
        "analyst_notes",
    ]:
        phishtool_template[c] = ""
    phishtool_template.to_csv(Path(args.output_phishtool_template_csv), index=False)

    print(f"Saved selection: {out_path} ({len(out_final)} rows)")
    print(f"Saved summary: {args.output_summary_csv}")
    print(f"Saved PhishTool template: {args.output_phishtool_template_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
