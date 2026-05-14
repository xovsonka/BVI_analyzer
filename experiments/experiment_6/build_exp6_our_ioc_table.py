from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _file_key(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda x: Path(x).name)


def _orig_file_key(series: pd.Series) -> pd.Series:
    def _strip(name: str) -> str:
        base = Path(str(name)).name
        return base.split("__", 1)[1] if "__" in base else base
    return series.astype(str).map(_strip)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _str(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=str)
    return df[col].astype(str)


def _load_source(analyzed_csv: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(analyzed_csv, low_memory=False)
    df["file_key"] = _file_key(df["file"])
    df["benchmark_source"] = source_name
    return df


def _load_predictions(pred_csv: Path, source_name: str) -> pd.DataFrame:
    if not pred_csv.exists():
        return pd.DataFrame(columns=["file_key", "ml_pred", "ml_pred_binary"])
    df = pd.read_csv(pred_csv, low_memory=False)
    df["file_key"] = _file_key(df["file"])
    keep = [c for c in ["file_key", "ml_pred", "ml_pred_binary"] if c in df.columns]
    out = df[keep].drop_duplicates(subset=["file_key"]).copy()
    out["benchmark_source"] = source_name
    return out


def _bool_from_col(df: pd.DataFrame, col: str, threshold: float = 0) -> pd.Series:
    return _num(df, col).gt(threshold).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build normalized our-tool IOC table for Exp6"
    )
    parser.add_argument(
        "--selection-csv",
        default="results/experiment_6/exp6_selection.csv",
    )
    parser.add_argument(
        "--exp1-analyzed-csv",
        default="results/experiment_5/exp1_campaign_model_input.csv",
    )
    parser.add_argument(
        "--exp2-b-analyzed-csv",
        default="results/eperiment_2_results/experiment_2_campaign_model_input.csv",
    )
    parser.add_argument(
        "--exp3-b-analyzed-csv",
        default="results/experiment_3_result/experiment_3_campaign_model_input.csv",
    )
    parser.add_argument(
        "--exp2-a-analyzed-csv",
        default="results/eperiment_2_results/experiment_2_header_auth_model_input.csv",
    )
    parser.add_argument(
        "--exp3-a-analyzed-csv",
        default="results/experiment_3_result/experiment_3_url_model_input.csv",
    )
    parser.add_argument(
        "--exp1-pred-csv",
        default="results/retuned_exp1234/semantic_cta_eval/exp1_spamboost_ps160_fs160_pred.csv",
    )
    parser.add_argument(
        "--exp2-b-pred-csv",
        default="results/retuned_exp1234/semantic_cta_eval/exp2_spamboost_ps130_fs160_pred.csv",
    )
    parser.add_argument(
        "--exp3-b-pred-csv",
        default="results/retuned_exp1234/semantic_cta_eval/exp3_ps160_fs160_pred.csv",
    )
    parser.add_argument("--heur-threshold", type=int, default=20)
    parser.add_argument(
        "--output-csv",
        default="results/experiment_6/exp6_our_ioc_table.csv",
    )
    args = parser.parse_args()

    selection = pd.read_csv(args.selection_csv, low_memory=False)
    selection["file_key"] = _file_key(selection["file"])
    selection["orig_file_key"] = _orig_file_key(selection["file"])

    analyzed = pd.concat(
        [
            _load_source(Path(args.exp1_analyzed_csv), "exp1_campaign"),
            _load_source(Path(args.exp2_b_analyzed_csv), "exp2_track_b"),
            _load_source(Path(args.exp3_b_analyzed_csv), "exp3_track_b"),
            _load_source(Path(args.exp2_a_analyzed_csv), "exp2_track_a"),
            _load_source(Path(args.exp3_a_analyzed_csv), "exp3_track_a"),
        ],
        ignore_index=True,
        sort=False,
    )

    preds = pd.concat(
        [
            _load_predictions(Path(args.exp1_pred_csv), "exp1_campaign"),
            _load_predictions(Path(args.exp2_b_pred_csv), "exp2_track_b"),
            _load_predictions(Path(args.exp3_b_pred_csv), "exp3_track_b"),
        ],
        ignore_index=True,
        sort=False,
    )

    merged = selection.merge(
        analyzed,
        left_on=["orig_file_key", "benchmark_source"],
        right_on=["file_key", "benchmark_source"],
        how="left",
        suffixes=("", "_an"),
    )
    merged = merged.merge(
        preds,
        left_on=["orig_file_key", "benchmark_source"],
        right_on=["file_key", "benchmark_source"],
        how="left",
        suffixes=("", "_pred"),
    )

    out = pd.DataFrame()
    out["exp6_id"] = _str(merged, "exp6_id")
    out["benchmark_layer"] = _str(merged, "benchmark_layer")
    out["benchmark_source"] = _str(merged, "benchmark_source")
    out["benchmark_bucket"] = _str(merged, "benchmark_bucket")
    out["file"] = _str(merged, "file")
    out["exp6_eml_path"] = _str(merged, "exp6_eml_path")
    out["subject"] = _str(merged, "subject")
    out["y_true_binary"] = _str(merged, "y_true_binary").str.lower()
    out["y_true_multiclass"] = (
        _str(merged, "y_true_multiclass").str.lower().replace({"nan": pd.NA})
    )

    out["our_spf_fail"] = _str(merged, "spf_result").str.lower().eq("fail").astype(int)
    out["our_dkim_fail"] = (
        _str(merged, "dkim_result").str.lower().eq("fail").astype(int)
    )
    out["our_dmarc_fail"] = (
        _str(merged, "dmarc_result").str.lower().eq("fail").astype(int)
    )
    out["our_from_reply_mismatch"] = _bool_from_col(merged, "from_reply_mismatch")
    out["our_from_return_path_mismatch"] = _bool_from_col(
        merged, "from_return_path_mismatch"
    )
    out["our_message_id_domain_mismatch"] = _bool_from_col(
        merged, "message_id_domain_mismatch"
    )
    out["our_received_anomaly"] = _bool_from_col(merged, "received_anomaly_flag")
    out["our_display_name_spoof"] = _bool_from_col(merged, "display_name_spoof_flag")
    out["our_ip_url"] = _bool_from_col(merged, "ip_url_count")
    out["our_shortener_url"] = _bool_from_col(merged, "shortener_url_count")
    out["our_anchor_mismatch"] = _bool_from_col(merged, "mismatched_anchor_count")
    out["our_suspicious_tld"] = _bool_from_col(merged, "suspicious_tld_count")
    out["our_brand_typosquat"] = _bool_from_col(merged, "brand_typosquat_flag")
    obf = _bool_from_col(merged, "obfuscated_url_count")
    if int(obf.sum()) == 0:
        obf = (
            _str(merged, "heuristic_reasons")
            .str.contains("obfuscated_url", regex=False)
            .astype(int)
        )
    out["our_obfuscated_url"] = obf
    out["our_risky_attachment"] = _bool_from_col(merged, "risky_attachment_ext_count")

    out["our_heuristic_score"] = _num(merged, "heuristic_score").astype(float)
    out["our_heuristic_reasons"] = _str(merged, "heuristic_reasons")
    out["our_pred_binary_heur"] = "legit"
    out.loc[
        out["our_heuristic_score"] >= args.heur_threshold, "our_pred_binary_heur"
    ] = "suspicious"

    out["our_pred_multiclass_ml"] = _str(merged, "ml_pred").replace({"nan": ""})
    out["our_pred_binary_ml"] = _str(merged, "ml_pred_binary").replace({"nan": ""})
    missing_ml_bin = out["our_pred_binary_ml"].eq("") & out[
        "our_pred_multiclass_ml"
    ].ne("")
    out.loc[missing_ml_bin, "our_pred_binary_ml"] = out.loc[
        missing_ml_bin, "our_pred_multiclass_ml"
    ].where(out.loc[missing_ml_bin, "our_pred_multiclass_ml"].eq("legit"), "suspicious")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved our IOC table: {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
