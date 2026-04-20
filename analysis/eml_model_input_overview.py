from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS_DEFAULT = [
    "heuristic_score",
    "url_count",
    "ip_url_count",
    "shortener_url_count",
    "unique_domain_count",
    "urgent_keyword_count",
    "credential_keyword_count",
    "threat_keyword_count",
    "financial_keyword_count",
    "subject_len",
    "text_len",
    "exclamation_count",
    "uppercase_ratio",
    "digit_ratio",
    "attachment_count",
    "risky_attachment_ext_count",
    "received_hops_count",
]


def format_series_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.DataFrame()

    stats = df[existing].describe().T.reset_index().rename(columns={"index": "feature"})
    keep = ["feature", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    return stats[keep]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick overview for model-ready EML CSV"
    )
    parser.add_argument(
        "--input", default="results/eml_model_input.csv", help="Input CSV path"
    )
    parser.add_argument(
        "--output-summary",
        default="results/eml_model_input_overview.txt",
        help="Output text summary path",
    )
    parser.add_argument(
        "--output-stats-csv",
        default="results/eml_model_input_feature_stats.csv",
        help="Output numeric stats CSV path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    risk_counts = (
        df["risk_level"].value_counts(dropna=False)
        if "risk_level" in df.columns
        else pd.Series(dtype=int)
    )
    top_reasons = (
        (
            df["heuristic_reasons"]
            .fillna("")
            .str.split(",")
            .explode()
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(15)
        )
        if "heuristic_reasons" in df.columns
        else pd.Series(dtype=int)
    )

    feature_stats = format_series_stats(df, NUMERIC_COLUMNS_DEFAULT)

    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with output_summary.open("w", encoding="utf-8") as f:
        f.write("EML Model Input Overview\n")
        f.write("=" * 80 + "\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Columns: {len(df.columns)}\n\n")

        f.write("Columns:\n")
        for col in df.columns:
            f.write(f"- {col}\n")

        f.write("\nRisk Level Distribution:\n")
        if len(risk_counts) > 0:
            f.write(risk_counts.to_string() + "\n")
        else:
            f.write("(column 'risk_level' not found)\n")

        f.write("\nTop Heuristic Reasons:\n")
        if len(top_reasons) > 0:
            f.write(top_reasons.to_string() + "\n")
        else:
            f.write("(column 'heuristic_reasons' not found or empty)\n")

        f.write("\nNumeric Feature Stats:\n")
        if not feature_stats.empty:
            f.write(feature_stats.to_string(index=False) + "\n")
        else:
            f.write("(no configured numeric columns found)\n")

        f.write("\nSample Rows (first 5):\n")
        f.write(df.head(5).to_string(index=False) + "\n")

    output_stats_csv = Path(args.output_stats_csv)
    output_stats_csv.parent.mkdir(parents=True, exist_ok=True)
    if not feature_stats.empty:
        feature_stats.to_csv(output_stats_csv, index=False)

    print(f"Saved summary: {output_summary}")
    if not feature_stats.empty:
        print(f"Saved feature stats: {output_stats_csv}")
    print("Done")


if __name__ == "__main__":
    main()
