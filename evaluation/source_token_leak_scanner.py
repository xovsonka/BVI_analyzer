from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import ensure_features


DATASET_LEAK_PATTERNS = {
    "token_ceas": r"\bceas[_\- ]?0?8\b",
    "token_fraud_email": r"\bfraud[_\- ]?email\b",
    "token_phishing_email": r"\bphishing[_\- ]?email\b",
    "token_spamassassin": r"\bspamassassin\b",
    "token_enron": r"\benron\b",
    "token_nazario": r"\bnazario\b",
    "token_nigerian_fraud": r"\bnigerian[_\- ]?fraud\b",
    "token_export_marker": r"\b(?:export|annotation|ground[_\- ]?truth|label\s*=)\b",
}

HEADER_ARTIFACT_PATTERNS = {
    "hdr_x_spam_status": r"(?im)^x-spam-status\s*:",
    "hdr_x_spam_flag": r"(?im)^x-spam-flag\s*:",
    "hdr_x_mailer": r"(?im)^x-mailer\s*:",
    "hdr_received": r"(?im)^received\s*:",
    "hdr_return_path": r"(?im)^return-path\s*:",
    "hdr_auth_results": r"(?im)^authentication-results\s*:",
    "hdr_dkim_signature": r"(?im)^dkim-signature\s*:",
    "hdr_x_gateway": r"(?im)^x-(?:proofpoint|mimecast|ironport|amavis|barracuda)[^:]*:",
}


def scan_one(
    df: pd.DataFrame, split_name: str, text_col: str = "text"
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    examples: list[dict] = []

    text_series = df[text_col].fillna("").astype(str)
    total = max(1, len(df))

    for group_name, patterns in [
        ("source_tokens", DATASET_LEAK_PATTERNS),
        ("header_artifacts", HEADER_ARTIFACT_PATTERNS),
    ]:
        for key, pattern in patterns.items():
            rgx = re.compile(pattern)
            mask = text_series.str.contains(rgx, regex=True)
            hit_rows = int(mask.sum())
            hit_pct = float(hit_rows / total)

            rows.append(
                {
                    "split": split_name,
                    "group": group_name,
                    "pattern_key": key,
                    "rows": int(len(df)),
                    "hit_rows": hit_rows,
                    "hit_pct": hit_pct,
                }
            )

            if hit_rows > 0:
                sample_idx = df[mask].head(3).index.tolist()
                for idx in sample_idx:
                    preview = re.sub(r"\s+", " ", str(text_series.loc[idx]))[:220]
                    examples.append(
                        {
                            "split": split_name,
                            "group": group_name,
                            "pattern_key": key,
                            "row_index": int(idx),
                            "label": str(df.loc[idx].get("label", "")),
                            "source": str(df.loc[idx].get("source", "")),
                            "preview": preview,
                        }
                    )

    return rows, examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan text for source-token and header-artifact leakage"
    )
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument("--scenario", default="scenario_g_source_balanced")
    parser.add_argument("--output-csv", default="results/source_token_leak_scan.csv")
    parser.add_argument("--output-json", default="results/source_token_leak_scan.json")
    parser.add_argument(
        "--output-examples-csv",
        default="results/source_token_leak_examples.csv",
    )
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    scenario_dir = processed / args.scenario
    files = {
        "train": scenario_dir / "train.csv",
        "val": scenario_dir / "val.csv",
        "test": scenario_dir / "test.csv",
        "shared_test": processed / "shared" / "test.csv",
    }

    for p in files.values():
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    all_rows: list[dict] = []
    all_examples: list[dict] = []
    for split_name, path in files.items():
        df = ensure_features(pd.read_csv(path, low_memory=False))
        rows, examples = scan_one(df, split_name=split_name, text_col="text")
        all_rows.extend(rows)
        all_examples.extend(examples)

    out_df = pd.DataFrame(all_rows).sort_values(["group", "pattern_key", "split"])
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    ex_df = pd.DataFrame(all_examples)
    out_ex = Path(args.output_examples_csv)
    out_ex.parent.mkdir(parents=True, exist_ok=True)
    ex_df.to_csv(out_ex, index=False)

    summary = (
        out_df.groupby(["group", "pattern_key"])["hit_pct"]
        .max()
        .reset_index()
        .sort_values("hit_pct", ascending=False)
    )
    red_flags = summary[summary["hit_pct"] >= 0.01].to_dict(orient="records")

    payload = {
        "scenario": args.scenario,
        "rows_scanned": int(len(out_df)),
        "max_hit_pct_by_pattern": summary.to_dict(orient="records"),
        "red_flags_threshold": 0.01,
        "red_flags": red_flags,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved CSV: {out_csv}")
    print(f"Saved examples: {out_ex}")
    print(f"Saved JSON: {out_json}")
    if red_flags:
        print(f"RED FLAGS: {len(red_flags)} patterns with >=1% hits")


if __name__ == "__main__":
    main()
