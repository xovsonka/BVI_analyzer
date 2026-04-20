from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


INDICATOR_COLUMNS = [
    "from_reply_mismatch",
    "from_return_path_mismatch",
    "sender_receiver_domain_mismatch",
    "ip_url_count",
    "shortener_url_count",
    "obfuscated_url_count",
    "mismatched_anchor_count",
    "suspicious_tld_count",
    "brand_typosquat_flag",
    "credential_keyword_count",
    "urgent_keyword_count",
    "threat_keyword_count",
    "financial_keyword_count",
    "risky_attachment_ext_count",
    "display_name_spoof_flag",
    "received_anomaly_flag",
    "external_domain_url_count",
    "sender_url_domain_mismatch_ratio",
    "credential_keyword_density",
    "urgent_keyword_density",
    "unicode_mixed_script_flag",
]

PROFILE_FEATURES = [
    "url_count",
    "credential_keyword_count",
    "mismatched_anchor_count",
    "risky_attachment_ext_count",
    "from_reply_mismatch",
    "from_return_path_mismatch",
    "external_domain_url_count",
    "sender_url_domain_mismatch_ratio",
    "credential_keyword_density",
    "urgent_keyword_density",
    "unicode_mixed_script_flag",
]


def normalize_binary_label(v: object) -> int | None:
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in {"", "none", "nan"}:
        return None
    if s in {"0", "legit", "benign", "ham", "safe"}:
        return 0
    if s in {
        "1",
        "malicious",
        "phishing",
        "spam",
        "fraud",
        "financial_fraud",
        "suspicious",
    }:
        return 1
    try:
        num = int(float(s))
        return 0 if num == 0 else 1
    except Exception:
        return None


def explode_reasons(df: pd.DataFrame) -> pd.Series:
    if "heuristic_reasons" not in df.columns:
        return pd.Series(dtype="string")
    return (
        df["heuristic_reasons"]
        .fillna("")
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )


def to_flag(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([0] * len(df), index=df.index)
    series = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return (series > 0).astype(int)


def build_report(df: pd.DataFrame) -> dict:
    n_rows = len(df)

    risk_distribution = (
        df["risk_level"].value_counts(dropna=False).to_dict()
        if "risk_level" in df.columns
        else {}
    )

    reason_series = explode_reasons(df)
    reason_counts = reason_series.value_counts().to_dict()

    if "heuristic_score" in df.columns and not reason_series.empty:
        reason_score_df = pd.DataFrame(
            {
                "reason": reason_series,
                "score": df.loc[reason_series.index, "heuristic_score"].values,
            }
        )
        avg_score_by_reason = (
            reason_score_df.groupby("reason")["score"]
            .mean()
            .sort_values(ascending=False)
            .round(3)
            .to_dict()
        )
    else:
        avg_score_by_reason = {}

    indicator_prevalence = {}
    for col in INDICATOR_COLUMNS:
        if col in df.columns:
            flagged = int(to_flag(df, col).sum())
            indicator_prevalence[col] = {
                "flagged_count": flagged,
                "flagged_pct": round((flagged / n_rows * 100.0), 3) if n_rows else 0.0,
            }

    top_indicators = dict(
        sorted(
            indicator_prevalence.items(),
            key=lambda x: x[1]["flagged_count"],
            reverse=True,
        )[:12]
    )

    # Pair co-occurrence for top reason keys
    top_reason_keys = list(
        pd.Series(reason_counts).sort_values(ascending=False).head(10).index
    )
    co_occurrence = {}
    if top_reason_keys and "heuristic_reasons" in df.columns:
        reason_sets = (
            df["heuristic_reasons"]
            .fillna("")
            .astype(str)
            .apply(lambda s: set([x.strip() for x in s.split(",") if x.strip()]))
        )
        for i, a in enumerate(top_reason_keys):
            for b in top_reason_keys[i + 1 :]:
                pair = f"{a} + {b}"
                count = int(reason_sets.apply(lambda s: int(a in s and b in s)).sum())
                if count > 0:
                    co_occurrence[pair] = count

    score_stats = {}
    if "heuristic_score" in df.columns:
        s = pd.to_numeric(df["heuristic_score"], errors="coerce").dropna()
        if len(s) > 0:
            score_stats = {
                "mean": float(round(s.mean(), 3)),
                "std": float(round(s.std(), 3)) if len(s) > 1 else 0.0,
                "min": float(s.min()),
                "p25": float(round(s.quantile(0.25), 3)),
                "median": float(round(s.quantile(0.5), 3)),
                "p75": float(round(s.quantile(0.75), 3)),
                "max": float(s.max()),
            }

    class_col = None
    for c in ["label", "label_hint"]:
        if c in df.columns and df[c].notna().any():
            class_col = c
            break

    coverage_by_class = {}
    per_class_feature_profile = {}
    separability_legit_vs_malicious = {}
    if class_col is not None:
        class_df = df.copy()
        class_df["_class"] = class_df[class_col].astype(str).fillna("unknown")

        for cls, part in class_df.groupby("_class"):
            cls_cov = {}
            for col in INDICATOR_COLUMNS:
                if col in part.columns:
                    flagged = int(to_flag(part, col).sum())
                    cls_cov[col] = {
                        "flagged_count": flagged,
                        "flagged_pct": round((flagged / max(1, len(part))) * 100.0, 3),
                    }
            coverage_by_class[str(cls)] = cls_cov

        for feat in PROFILE_FEATURES:
            if feat not in class_df.columns:
                continue
            numeric = pd.to_numeric(class_df[feat], errors="coerce")
            tmp = class_df.assign(_feat=numeric)
            agg = (
                tmp.groupby("_class")["_feat"]
                .agg(
                    mean="mean",
                    median="median",
                    p25=lambda s: s.quantile(0.25),
                    p75=lambda s: s.quantile(0.75),
                )
                .fillna(0.0)
                .round(4)
            )
            per_class_feature_profile[feat] = {
                str(k): {kk: float(vv) for kk, vv in row.items()}
                for k, row in agg.to_dict(orient="index").items()
            }

        if "label" in df.columns or "label_hint" in df.columns:
            y_col = "label" if "label" in df.columns else "label_hint"
            y_bin = df[y_col].apply(normalize_binary_label)
            tmp = df.copy()
            tmp["_y_bin"] = y_bin
            tmp = tmp[tmp["_y_bin"].notna()].copy()
            if not tmp.empty:
                tmp["_y_bin"] = tmp["_y_bin"].astype(int)
                for feat in PROFILE_FEATURES:
                    if feat not in tmp.columns:
                        continue
                    s = pd.to_numeric(tmp[feat], errors="coerce").fillna(0.0)
                    legit = s[tmp["_y_bin"] == 0]
                    malicious = s[tmp["_y_bin"] == 1]
                    if len(legit) == 0 or len(malicious) == 0:
                        continue
                    separability_legit_vs_malicious[feat] = {
                        "mean_legit": float(round(legit.mean(), 4)),
                        "mean_malicious": float(round(malicious.mean(), 4)),
                        "median_legit": float(round(legit.median(), 4)),
                        "median_malicious": float(round(malicious.median(), 4)),
                        "delta_mean": float(round(malicious.mean() - legit.mean(), 4)),
                    }

    return {
        "rows": n_rows,
        "risk_distribution": risk_distribution,
        "score_stats": score_stats,
        "reason_counts": reason_counts,
        "avg_score_by_reason": avg_score_by_reason,
        "indicator_prevalence": indicator_prevalence,
        "top_indicators": top_indicators,
        "reason_co_occurrence": co_occurrence,
        "coverage_by_class": coverage_by_class,
        "per_class_feature_profile": per_class_feature_profile,
        "separability_legit_vs_malicious": separability_legit_vs_malicious,
    }


def write_text_summary(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("Heuristic Analyzer Statistics\n")
        f.write("=" * 80 + "\n")
        f.write(f"Rows analyzed: {report['rows']}\n\n")

        f.write("Risk distribution:\n")
        for k, v in report["risk_distribution"].items():
            f.write(f"- {k}: {v}\n")

        f.write("\nHeuristic score stats:\n")
        for k, v in report["score_stats"].items():
            f.write(f"- {k}: {v}\n")

        f.write("\nTop reason counts:\n")
        for k, v in sorted(
            report["reason_counts"].items(), key=lambda x: x[1], reverse=True
        )[:20]:
            f.write(f"- {k}: {v}\n")

        f.write("\nTop indicators by prevalence:\n")
        for k, v in report["top_indicators"].items():
            f.write(f"- {k}: {v['flagged_count']} ({v['flagged_pct']}%)\n")

        f.write("\nMost common reason co-occurrence:\n")
        for k, v in sorted(
            report["reason_co_occurrence"].items(), key=lambda x: x[1], reverse=True
        )[:20]:
            f.write(f"- {k}: {v}\n")

        if report.get("separability_legit_vs_malicious"):
            f.write("\nFeature separability (legit vs malicious):\n")
            for feat, vals in sorted(
                report["separability_legit_vs_malicious"].items(),
                key=lambda x: abs(x[1].get("delta_mean", 0.0)),
                reverse=True,
            )[:15]:
                f.write(
                    f"- {feat}: legit_mean={vals['mean_legit']}, malicious_mean={vals['mean_malicious']}, delta={vals['delta_mean']}\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate heuristic analyzer statistics report"
    )
    parser.add_argument(
        "--input", default="results/eml_model_input.csv", help="Input model-ready CSV"
    )
    parser.add_argument(
        "--output-json",
        default="results/heuristic_stats.json",
        help="Output JSON report",
    )
    parser.add_argument(
        "--output-txt",
        default="results/heuristic_stats.txt",
        help="Output text summary",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    report = build_report(df)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    output_txt = Path(args.output_txt)
    write_text_summary(report, output_txt)

    print(f"Saved JSON: {output_json}")
    print(f"Saved TXT: {output_txt}")


if __name__ == "__main__":
    main()
