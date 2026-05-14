from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LABEL_ORDER = ["legit", "phishing", "spam", "financial_fraud"]


plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(ROOT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(ROOT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_per_class_gain(
    baseline_csv: Path,
    adapted_csv: Path,
    title: str,
    stem: str,
) -> None:
    base = pd.read_csv(baseline_csv).set_index("label").reindex(LABEL_ORDER)
    adapt = pd.read_csv(adapted_csv).set_index("label").reindex(LABEL_ORDER)

    x = np.arange(len(LABEL_ORDER))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    bars1 = ax.bar(x - width / 2, base["f1"].to_numpy(), width=width, label="Baseline", color="#7aa6d1")
    bars2 = ax.bar(x + width / 2, adapt["f1"].to_numpy(), width=width, label="Adapted + OVR", color="#1d4f91")

    for bars in [bars1, bars2]:
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Per-class F1")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    _save(fig, stem)


def plot_exp2_per_class_gain() -> None:
    _plot_per_class_gain(
        baseline_csv=ROOT / "results" / "experiment_2_clean" / "exp2_existing_reanalyzed_per_class.csv",
        adapted_csv=ROOT / "results" / "retuned_exp1234" / "semantic_cta_eval" / "exp2_spamboost_ps130_fs160_per_class.csv",
        title="Experiment 2: per-class F1 before and after semantic OVR refinement",
        stem="exp2_per_class_baseline_vs_adapted_ovr",
    )


def plot_exp2_indicator_contribution() -> None:
    rows = [
        {"indicator": "SPF fail", "points": 20, "note": "+20"},
        {"indicator": "DMARC fail", "points": 20, "note": "+20"},
        {"indicator": "From vs Reply-To", "points": 15, "note": "+15"},
        {"indicator": "From vs Return-Path", "points": 15, "note": "+15"},
        {"indicator": "DKIM fail", "points": 15, "note": "+15"},
        {"indicator": "Display-name spoof", "points": 10, "note": "+10"},
        {"indicator": "Received anomaly", "points": 5, "note": "+5"},
        {"indicator": "Message-ID mismatch", "points": 0, "note": "tracked"},
        {"indicator": "Trusted-domain guardrail", "points": -20, "note": "-20"},
    ]
    df = pd.DataFrame(rows).sort_values("points", ascending=True)
    colors = ["#d9534f" if v < 0 else "#4f81bd" for v in df["points"]]

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    bars = ax.barh(df["indicator"], df["points"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Príspevok do heuristického skóre")
    ax.set_title("Experiment 2: relatívna sila hlavičkových a autentifikačných indikátorov")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(-24, 24)

    for b, note, points in zip(bars, df["note"], df["points"]):
        xpos = points + 0.7 if points >= 0 else points - 0.7
        ha = "left" if points >= 0 else "right"
        ax.text(xpos, b.get_y() + b.get_height() / 2, note, va="center", ha=ha, fontsize=10)

    _save(fig, "exp2_header_auth_indicator_contribution")


def plot_exp3_per_class_gain() -> None:
    _plot_per_class_gain(
        baseline_csv=ROOT / "results" / "experiment_3_clean" / "exp3_existing_reanalyzed_per_class.csv",
        adapted_csv=ROOT / "results" / "retuned_exp1234" / "semantic_cta_eval" / "exp3_ps160_fs160_per_class.csv",
        title="Experiment 3: per-class F1 before and after semantic OVR refinement",
        stem="exp3_per_class_baseline_vs_adapted_ovr",
    )


def plot_exp4_tradeoff() -> None:
    rows = [
        {"benchmark": "Exp1 campaign", "variant": "Tuned single-stage", "score": 0.6213146408202562},
        {"benchmark": "Exp1 campaign", "variant": "Semantic OVR", "score": 0.7988632848185588},
        {"benchmark": "Exp2 campaign", "variant": "Tuned single-stage", "score": 0.3262},
        {"benchmark": "Exp2 campaign", "variant": "Semantic OVR", "score": 0.9556623931623932},
        {"benchmark": "Exp3 campaign", "variant": "Tuned single-stage", "score": 0.3003},
        {"benchmark": "Exp3 campaign", "variant": "Semantic OVR", "score": 0.9798976608187134},
        {"benchmark": "test_deployment", "variant": "Tuned single-stage", "score": 0.9076},
        {"benchmark": "test_deployment", "variant": "Semantic OVR", "score": 0.7865471841004643},
    ]
    plot_df = pd.DataFrame(rows)
    benchmarks = ["Exp1 campaign", "Exp2 campaign", "Exp3 campaign", "test_deployment"]
    piv = plot_df.pivot_table(index="benchmark", columns="variant", values="score", aggfunc="first").reindex(benchmarks)

    x = np.arange(len(benchmarks))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    bars1 = ax.bar(x - width / 2, piv["Tuned single-stage"].to_numpy(), width=width, label="Tuned single-stage", color="#2a9d8f")
    bars2 = ax.bar(x + width / 2, piv["Semantic OVR"].to_numpy(), width=width, label="Semantic OVR", color="#e76f51")

    for bars in [bars1, bars2]:
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Macro F1")
    ax.set_title("Experiment 4: trade-off between general and campaign-specialized model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", ncol=2)
    _save(fig, "exp4_general_vs_campaign_tradeoff")


def main() -> None:
    plot_exp2_indicator_contribution()
    plot_exp2_per_class_gain()
    plot_exp3_per_class_gain()
    plot_exp4_tradeoff()
    print("Saved Exp2/3/4 additional thesis figures to:", ROOT)


if __name__ == "__main__":
    main()
