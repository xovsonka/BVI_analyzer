from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyze_campaign import compute_heuristic_score


def _file_key(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda x: Path(x).name)


def _binary_from_multiclass(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.lower()
    return out.where(out.eq("legit"), "suspicious")


def _load_campaign_source(
    analyzed_csv: Path,
    pred_csv: Path,
    source_name: str,
) -> pd.DataFrame:
    analyzed = pd.read_csv(analyzed_csv, low_memory=False)
    pred = pd.read_csv(pred_csv, low_memory=False)

    analyzed = analyzed.copy()
    pred = pred.copy()
    analyzed["file_key"] = _file_key(analyzed["file"])
    pred["file_key"] = _file_key(pred["file"])

    keep_cols = [
        "file_key",
        "y_true_multiclass",
        "y_true_binary",
    ]
    pred_small = pred[keep_cols].drop_duplicates(subset=["file_key"])

    merged = analyzed.merge(pred_small, on="file_key", how="inner")
    merged["benchmark_group"] = "realistic_campaign"
    merged["benchmark_source"] = source_name
    merged["has_multiclass"] = 1
    return merged


def _load_exp2_tracka(analyzed_csv: Path, expected_csv: Path) -> pd.DataFrame:
    analyzed = pd.read_csv(analyzed_csv, low_memory=False)
    expected = pd.read_csv(expected_csv, low_memory=False)

    analyzed = analyzed.copy()
    expected = expected.copy()
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
    exp["y_true_binary"] = "legit"
    exp.loc[exp[cols].sum(axis=1) > 0, "y_true_binary"] = "suspicious"
    exp["y_true_multiclass"] = pd.NA

    merged = analyzed.merge(
        exp[["file_key", "case_id", "y_true_binary", "y_true_multiclass"]],
        on="file_key",
        how="inner",
    )
    merged["benchmark_group"] = "controlled_benchmark"
    merged["benchmark_source"] = "exp2_track_a"
    merged["has_multiclass"] = 0
    return merged


def _load_exp3_tracka(analyzed_csv: Path, expected_csv: Path) -> pd.DataFrame:
    analyzed = pd.read_csv(analyzed_csv, low_memory=False)
    expected = pd.read_csv(expected_csv, low_memory=False)

    analyzed = analyzed.copy()
    expected = expected.copy()
    analyzed["file_key"] = _file_key(analyzed["file"])
    expected["file_key"] = _file_key(expected["file"])

    exp = expected[["file_key", "case_id", "expected_binary_label"]].copy()
    exp["y_true_binary"] = exp["expected_binary_label"].astype(str).str.lower()
    exp["y_true_multiclass"] = pd.NA

    merged = analyzed.merge(
        exp[["file_key", "case_id", "y_true_binary", "y_true_multiclass"]],
        on="file_key",
        how="inner",
    )
    merged["benchmark_group"] = "controlled_benchmark"
    merged["benchmark_source"] = "exp3_track_a"
    merged["has_multiclass"] = 0
    return merged


def _load_deployment_sample(
    test_csv: Path, sample_rows: int, seed: int
) -> pd.DataFrame:
    df = pd.read_csv(test_csv, low_memory=False)
    if sample_rows > 0 and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=seed)
    df = df.reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        features = row.to_dict()
        heuristic = compute_heuristic_score(features)
        out = row.to_dict()
        out["heuristic_score"] = int(heuristic.get("heuristic_score", 0))
        out["heuristic_reasons"] = heuristic.get("heuristic_reasons", "")
        out["file"] = f"offline_deployment_row_{len(rows) + 1}.eml"
        out["text_input"] = str(row.get("text", ""))
        out["y_true_multiclass"] = str(row.get("label", "")).strip().lower()
        out["y_true_binary"] = (
            "legit" if out["y_true_multiclass"] == "legit" else "suspicious"
        )
        out["benchmark_group"] = "offline_generalization"
        out["benchmark_source"] = "test_deployment_sample"
        out["has_multiclass"] = 1
        rows.append(out)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build unified benchmark dataset for Experiment 5"
    )
    parser.add_argument(
        "--exp1-analyzed-csv",
        default="results/experiment_5/exp1_campaign_model_input.csv",
    )
    parser.add_argument(
        "--exp1-pred-csv",
        default="results/experiment_5/exp1_campaign_ml_predictions.csv",
    )
    parser.add_argument(
        "--exp2-analyzed-csv",
        default="results/eperiment_2_results/experiment_2_campaign_model_input.csv",
    )
    parser.add_argument(
        "--exp2-pred-csv",
        default="results/eperiment_2_results/experiment_2_campaign_ml_predictions.csv",
    )
    parser.add_argument(
        "--exp3-analyzed-csv",
        default="results/experiment_3_result/experiment_3_campaign_model_input.csv",
    )
    parser.add_argument(
        "--exp3-pred-csv",
        default="results/experiment_3_result/experiment_3_campaign_ml_predictions.csv",
    )
    parser.add_argument(
        "--exp2-tracka-analyzed-csv",
        default="results/eperiment_2_results/experiment_2_header_auth_model_input.csv",
    )
    parser.add_argument(
        "--exp2-tracka-expected-csv",
        default="dataset/experiment_2/header_auth_expected.csv",
    )
    parser.add_argument(
        "--exp3-tracka-analyzed-csv",
        default="results/experiment_3_result/experiment_3_url_model_input.csv",
    )
    parser.add_argument(
        "--exp3-tracka-expected-csv",
        default="dataset/experiment_3/url_expected.csv",
    )
    parser.add_argument(
        "--deployment-test-csv",
        default="dataset/processed/shared/test_deployment.csv",
    )
    parser.add_argument(
        "--deployment-sample-rows",
        type=int,
        default=800,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-csv",
        default="results/experiment_5/exp5_unified_benchmark.csv",
    )
    parser.add_argument(
        "--output-summary-csv",
        default="results/experiment_5/exp5_unified_benchmark_summary.csv",
    )
    args = parser.parse_args()

    frames = [
        _load_campaign_source(
            Path(args.exp1_analyzed_csv),
            Path(args.exp1_pred_csv),
            "exp1_campaign",
        ),
        _load_campaign_source(
            Path(args.exp2_analyzed_csv),
            Path(args.exp2_pred_csv),
            "exp2_track_b",
        ),
        _load_campaign_source(
            Path(args.exp3_analyzed_csv),
            Path(args.exp3_pred_csv),
            "exp3_track_b",
        ),
        _load_exp2_tracka(
            Path(args.exp2_tracka_analyzed_csv),
            Path(args.exp2_tracka_expected_csv),
        ),
        _load_exp3_tracka(
            Path(args.exp3_tracka_analyzed_csv),
            Path(args.exp3_tracka_expected_csv),
        ),
        _load_deployment_sample(
            Path(args.deployment_test_csv),
            sample_rows=args.deployment_sample_rows,
            seed=args.seed,
        ),
    ]

    unified = pd.concat(frames, ignore_index=True, sort=False)
    unified["y_true_binary"] = unified["y_true_binary"].astype(str).str.lower()
    unified["y_true_multiclass"] = (
        unified["y_true_multiclass"].astype(str).str.lower().replace({"nan": pd.NA})
    )
    unified["has_multiclass"] = unified["has_multiclass"].fillna(0).astype(int)
    unified["heuristic_score"] = (
        pd.to_numeric(unified.get("heuristic_score", 0), errors="coerce")
        .fillna(0)
        .astype(float)
    )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    unified.to_csv(out_path, index=False)

    summary = (
        unified.groupby(["benchmark_group", "benchmark_source"], dropna=False)
        .agg(
            rows=("benchmark_source", "count"),
            multiclass_rows=("has_multiclass", "sum"),
            mean_heuristic_score=("heuristic_score", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(Path(args.output_summary_csv), index=False)

    print(f"Saved unified benchmark: {out_path} ({len(unified)} rows)")
    print(f"Saved benchmark summary: {args.output_summary_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
