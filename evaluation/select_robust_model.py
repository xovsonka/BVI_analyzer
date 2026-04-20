from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def latest_benchmark_csv(root: Path) -> Path:
    candidates = sorted(
        root.glob("*/benchmark_summary.csv"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(f"No benchmark_summary.csv found under: {root}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select robust scenario/model using benchmark + source-holdout summary"
    )
    parser.add_argument(
        "--benchmark-csv",
        default="",
        help="Path to benchmark_summary.csv (if omitted, latest file under results/architecture_benchmark is used)",
    )
    parser.add_argument(
        "--source-summary-csv",
        default="results/source_split_summary.csv",
        help="Path to source holdout summary CSV from eval_source_split.py",
    )
    parser.add_argument(
        "--architecture",
        default="single_stage",
        choices=["single_stage", "two_stage", "all"],
        help="Architecture subset for selection",
    )
    parser.add_argument(
        "--selection-mode",
        default="hierarchy",
        choices=["hierarchy", "weighted", "final_rule"],
        help="'hierarchy' = mean_holdout -> worst_holdout -> shared_test; 'weighted' = weighted score",
    )
    parser.add_argument(
        "--split-suite-summary-csv",
        default="results/split_suite_summary.csv",
        help="Summary CSV from eval_split_suite.py for val_iid/test_iid/hard/deployment selection rule",
    )
    parser.add_argument(
        "--min-hard-f1",
        type=float,
        default=0.0,
        help="Optional gate threshold for both hard split macro F1 in final_rule mode",
    )
    parser.add_argument("--w-overall", type=float, default=0.40)
    parser.add_argument("--w-mean-holdout", type=float, default=0.30)
    parser.add_argument("--w-worst-holdout", type=float, default=0.20)
    parser.add_argument("--w-std-penalty", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-csv", default="results/robust_model_selection.csv")
    parser.add_argument("--output-json", default="results/robust_model_selection.json")
    args = parser.parse_args()

    if args.benchmark_csv:
        benchmark_csv = Path(args.benchmark_csv)
    else:
        benchmark_csv = latest_benchmark_csv(Path("results/architecture_benchmark"))
    if not benchmark_csv.exists():
        raise FileNotFoundError(f"Missing benchmark CSV: {benchmark_csv}")

    benchmark = pd.read_csv(benchmark_csv)
    if benchmark.empty:
        raise RuntimeError("Benchmark summary is empty")

    if args.architecture != "all":
        benchmark = benchmark[benchmark["architecture"] == args.architecture].copy()

    benchmark = benchmark.copy()
    benchmark["model"] = benchmark["stage1_model"].astype(str)

    source_summary_path = Path(args.source_summary_csv)
    if source_summary_path.exists():
        source_summary = pd.read_csv(source_summary_path)
    else:
        source_summary = pd.DataFrame(
            columns=[
                "scenario",
                "model",
                "mean_holdout_f1_macro",
                "worst_holdout_f1_macro",
                "std_holdout_f1_macro",
            ]
        )

    merged = benchmark.merge(
        source_summary[
            [
                "scenario",
                "model",
                "mean_holdout_f1_macro",
                "worst_holdout_f1_macro",
                "std_holdout_f1_macro",
            ]
        ],
        on=["scenario", "model"],
        how="left",
    )

    for col in [
        "test_f1_macro",
        "mean_holdout_f1_macro",
        "worst_holdout_f1_macro",
        "std_holdout_f1_macro",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["mean_holdout_f1_macro"] = merged["mean_holdout_f1_macro"].fillna(
        merged["test_f1_macro"]
    )
    merged["worst_holdout_f1_macro"] = merged["worst_holdout_f1_macro"].fillna(
        merged["test_f1_macro"]
    )
    merged["std_holdout_f1_macro"] = merged["std_holdout_f1_macro"].fillna(0.0)

    if args.selection_mode == "final_rule":
        split_suite_path = Path(args.split_suite_summary_csv)
        if not split_suite_path.exists():
            raise FileNotFoundError(
                f"Missing split suite summary CSV for final_rule mode: {split_suite_path}"
            )
        suite = pd.read_csv(split_suite_path)

        needed = [
            "scenario",
            "model",
            "f1_macro_val_iid",
            "f1_macro_test_iid",
            "f1_macro_test_hard_source",
            "f1_macro_test_hard_cluster",
            "f1_macro_test_deployment",
        ]
        missing_cols = [c for c in needed if c not in suite.columns]
        if missing_cols:
            raise ValueError(
                "split suite summary missing columns: " + ", ".join(missing_cols)
            )

        for c in needed[2:]:
            suite[c] = pd.to_numeric(suite[c], errors="coerce")

        ranked = suite.merge(
            benchmark[["scenario", "model", "architecture"]].drop_duplicates(),
            on=["scenario", "model"],
            how="left",
        )
        if args.architecture != "all":
            ranked = ranked[ranked["architecture"] == args.architecture].copy()

        ranked["hard_gate"] = ranked[
            [
                "f1_macro_test_hard_source",
                "f1_macro_test_hard_cluster",
            ]
        ].min(axis=1)
        if args.min_hard_f1 > 0:
            ranked = ranked[ranked["hard_gate"] >= args.min_hard_f1].copy()

        ranked["robust_selection_score"] = (
            ranked["f1_macro_val_iid"] * 1_000_000.0
            + ranked["hard_gate"] * 1_000.0
            + ranked["f1_macro_test_iid"]
            + ranked["f1_macro_test_deployment"] * 0.001
        )
        ranked = ranked.sort_values(
            [
                "f1_macro_val_iid",
                "hard_gate",
                "f1_macro_test_iid",
                "f1_macro_test_deployment",
            ],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    elif args.selection_mode == "weighted":
        merged["robust_selection_score"] = (
            args.w_overall * merged["test_f1_macro"]
            + args.w_mean_holdout * merged["mean_holdout_f1_macro"]
            + args.w_worst_holdout * merged["worst_holdout_f1_macro"]
            - args.w_std_penalty * merged["std_holdout_f1_macro"]
        )
        ranked = merged.sort_values(
            ["robust_selection_score", "test_f1_macro"],
            ascending=False,
        ).reset_index(drop=True)
    else:
        merged["robust_selection_score"] = (
            merged["mean_holdout_f1_macro"] * 1_000_000.0
            + merged["worst_holdout_f1_macro"] * 1_000.0
            + merged["test_f1_macro"]
        )
        ranked = merged.sort_values(
            [
                "mean_holdout_f1_macro",
                "worst_holdout_f1_macro",
                "test_f1_macro",
                "std_holdout_f1_macro",
            ],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_csv, index=False)

    best = ranked.iloc[0].to_dict() if not ranked.empty else {}
    payload = {
        "benchmark_csv": str(benchmark_csv),
        "source_summary_csv": str(source_summary_path),
        "split_suite_summary_csv": args.split_suite_summary_csv,
        "selection_mode": args.selection_mode,
        "weights": {
            "w_overall": args.w_overall,
            "w_mean_holdout": args.w_mean_holdout,
            "w_worst_holdout": args.w_worst_holdout,
            "w_std_penalty": args.w_std_penalty,
        },
        "best": best,
        "top": ranked.head(max(1, args.top_k)).to_dict(orient="records"),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved ranking CSV: {out_csv}")
    print(f"Saved summary JSON: {out_json}")
    if best:
        print("Best robust config:")
        print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
