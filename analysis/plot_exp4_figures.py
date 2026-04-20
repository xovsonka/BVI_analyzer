from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


def plot_offline_deployment(primary: pd.DataFrame, out_dir: Path) -> None:
    deploy = primary[primary["split"] == "test_deployment"].copy()
    if deploy.empty:
        return

    deploy["name"] = deploy["scenario"].str.replace("scenario_", "", regex=False)
    deploy["name"] = (
        deploy["name"] + "\n" + deploy["model"].str.replace("hybrid_", "", regex=False)
    )
    deploy = deploy.sort_values("f1_macro", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        deploy["name"], deploy["f1_macro"], color="#2a9d8f", edgecolor="black"
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Experiment 4 Track A: test_deployment Macro F1")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    save_figure(fig, out_dir / "exp4_trackA_deployment_macro_f1.png")
    plt.close(fig)


def plot_campaign_binary(campaign: pd.DataFrame, out_dir: Path) -> None:
    if campaign.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    campaign = campaign.copy().sort_values("bin_f1", ascending=False)
    bars = ax.bar(
        campaign["campaign_benchmark"],
        campaign["bin_f1"],
        color="#457b9d",
        edgecolor="black",
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel("Binary F1 (legit vs suspicious)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Experiment 4 Track B: campaign binary ML performance")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_dir / "exp4_trackB_campaign_binary_f1.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Experiment 4 figures")
    parser.add_argument(
        "--offline-primary-csv",
        default="results/experiment_4/exp4_offline_primary_table.csv",
    )
    parser.add_argument(
        "--campaign-csv",
        default="results/experiment_4/exp4_campaign_ml_table.csv",
    )
    parser.add_argument("--out-dir", default="results/graphs_thesis_final")
    args = parser.parse_args()

    primary = pd.read_csv(args.offline_primary_csv)
    campaign = (
        pd.read_csv(args.campaign_csv)
        if Path(args.campaign_csv).exists()
        else pd.DataFrame()
    )
    out_dir = Path(args.out_dir)

    plot_offline_deployment(primary, out_dir)
    plot_campaign_binary(campaign, out_dir)
    print(f"Saved Exp4 figures to: {out_dir}")


if __name__ == "__main__":
    main()
