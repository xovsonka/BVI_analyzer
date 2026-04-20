from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_top_models(split_summary: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        "scenario",
        "model",
        "f1_macro_val_iid",
        "f1_macro_test_iid",
        "f1_macro_test_hard_source",
        "f1_macro_test_hard_cluster",
        "f1_macro_test_deployment",
    ]
    keep = split_summary[cols].copy()
    keep["label"] = keep["scenario"] + "\n" + keep["model"]
    keep = keep.sort_values("f1_macro_val_iid", ascending=False).head(10)

    x = range(len(keep))
    plt.figure(figsize=(14, 7))
    plt.plot(x, keep["f1_macro_val_iid"], marker="o", label="val_iid")
    plt.plot(x, keep["f1_macro_test_iid"], marker="o", label="test_iid")
    plt.plot(x, keep["f1_macro_test_hard_source"], marker="o", label="hard_source")
    plt.plot(x, keep["f1_macro_test_hard_cluster"], marker="o", label="hard_cluster")
    plt.plot(x, keep["f1_macro_test_deployment"], marker="o", label="deployment")
    plt.xticks(list(x), keep["label"], rotation=30, ha="right")
    plt.ylabel("Macro F1")
    plt.title("Top configurations across evaluation splits")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "top_models_across_splits.png", dpi=180)
    plt.close()


def plot_scenario_comparison(split_summary: pd.DataFrame, out_dir: Path) -> None:
    best_per_scenario = (
        split_summary.sort_values(
            ["scenario", "f1_macro_val_iid"], ascending=[True, False]
        )
        .groupby("scenario", as_index=False)
        .first()
    )

    plt.figure(figsize=(12, 6))
    idx = range(len(best_per_scenario))
    w = 0.16
    plt.bar(
        [i - 2 * w for i in idx],
        best_per_scenario["f1_macro_val_iid"],
        width=w,
        label="val_iid",
    )
    plt.bar(
        [i - w for i in idx],
        best_per_scenario["f1_macro_test_iid"],
        width=w,
        label="test_iid",
    )
    plt.bar(
        [i for i in idx],
        best_per_scenario["f1_macro_test_hard_source"],
        width=w,
        label="hard_source",
    )
    plt.bar(
        [i + w for i in idx],
        best_per_scenario["f1_macro_test_hard_cluster"],
        width=w,
        label="hard_cluster",
    )
    plt.bar(
        [i + 2 * w for i in idx],
        best_per_scenario["f1_macro_test_deployment"],
        width=w,
        label="deployment",
    )
    plt.xticks(list(idx), best_per_scenario["scenario"], rotation=20, ha="right")
    plt.ylabel("Macro F1")
    plt.title("Best model per scenario across splits")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(out_dir / "scenario_best_model_comparison.png", dpi=180)
    plt.close()


def plot_text_views(view_csv: Path, out_dir: Path) -> None:
    if not view_csv.exists():
        return
    df = pd.read_csv(view_csv)
    piv = df.pivot_table(
        index="split", columns="view", values="f1_macro", aggfunc="first"
    )
    piv = piv.reset_index()

    plt.figure(figsize=(10, 5))
    for col in [c for c in piv.columns if c != "split"]:
        plt.plot(piv["split"], piv[col], marker="o", label=col)
    plt.ylabel("Macro F1")
    plt.title("Body vs Header+Body vs Full text")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "text_view_comparison.png", dpi=180)
    plt.close()


def plot_two_stage_compare(
    two_stage_csv: Path, split_summary: pd.DataFrame, out_dir: Path
) -> None:
    if not two_stage_csv.exists():
        return
    two = pd.read_csv(two_stage_csv)
    if two.empty:
        return

    rows = []
    for _, r in two.iterrows():
        sc = r["scenario"]
        one = split_summary[
            (split_summary["scenario"] == sc)
            & (split_summary["model"] == "hybrid_sgd_hinge")
        ]
        if one.empty:
            continue
        rows.append(
            {
                "scenario": sc,
                "single_stage_f1": float(one.iloc[0]["f1_macro_test_iid"]),
                "two_stage_f1": float(r["test_iid_f1_macro"]),
            }
        )

    if not rows:
        return
    df = pd.DataFrame(rows)
    x = range(len(df))
    w = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(
        [i - w / 2 for i in x], df["single_stage_f1"], width=w, label="single-stage"
    )
    plt.bar([i + w / 2 for i in x], df["two_stage_f1"], width=w, label="two-stage")
    plt.xticks(list(x), df["scenario"], rotation=20, ha="right")
    plt.ylabel("Macro F1 (test_iid)")
    plt.title("Single-stage vs Two-stage (text-only)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "single_vs_two_stage.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready comparison plots"
    )
    parser.add_argument(
        "--split-summary-csv",
        default="results/split_suite_text_only_ghijk_summary.csv",
    )
    parser.add_argument(
        "--text-view-csv",
        default="results/text_view_modes.csv",
    )
    parser.add_argument("--out-dir", default="results/graphs")
    parser.add_argument(
        "--two-stage-csv",
        default="results/two_stage_text_only/summary.csv",
    )
    args = parser.parse_args()

    summary_path = Path(args.split_summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing split summary: {summary_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_summary = pd.read_csv(summary_path)
    plot_top_models(split_summary, out_dir)
    plot_scenario_comparison(split_summary, out_dir)
    plot_text_views(Path(args.text_view_csv), out_dir)
    plot_two_stage_compare(Path(args.two_stage_csv), split_summary, out_dir)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
