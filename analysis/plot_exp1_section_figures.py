from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "graphs_thesis_final"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = ["legit", "phishing", "spam", "financial_fraud"]


plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_m_progress() -> None:
    m_base_text_only = 0.538412
    m_base_tpf = 0.671022
    m_rebalanced_tpf = 0.714012
    m_final_scaled = 0.730513

    names = [
        "M + LogReg\n(text_only)",
        "M + LogReg\n(text_plus_features)",
        "M rebalanced + LogReg\n(text_plus_features)",
        "M rebalanced + LogReg\n(+ class scale)",
    ]
    vals = [m_base_text_only, m_base_tpf, m_rebalanced_tpf, m_final_scaled]

    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, color=["#6e9bc3", "#4f81bd", "#2f5f98", "#1d3f6e"])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.45, 0.78)
    ax.set_ylabel("MC F1 macro")
    ax.set_title("Experiment 1: evolution of scenario M multiclass performance")
    ax.grid(axis="y", alpha=0.25)

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center")

    _save(fig, "exp1_m_multiclass_progress")


def _load_pred(
    path: Path, model_config: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, low_memory=False)
    if model_config is not None and "model_config" in df.columns:
        df = df[df["model_config"] == model_config].copy()
    y_true = df["y_true_multiclass"].astype(str).to_numpy()
    y_pred = df["pred_label"].astype(str).to_numpy()
    return y_true, y_pred


def _plot_cm(ax: plt.Axes, cm: np.ndarray, title: str) -> None:
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=25, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_title(title)
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        for j in range(cm.shape[1]):
            pct = (100.0 * cm[i, j] / row_sum) if row_sum else 0.0
            ax.text(
                j, i, f"{cm[i, j]}\n{pct:.1f}%", ha="center", va="center", fontsize=10
            )
    return im


def plot_confusion_comparison() -> None:
    y_true_b, y_pred_b = _load_pred(
        RESULTS / "exp1sender_lm_multiclass_predictions_text_plus_features.csv",
        "scenario_m_anti_template_feature_regularized:hybrid_logreg",
    )
    y_true_f, y_pred_f = _load_pred(
        RESULTS / "exp1sender_m_rebalanced_predictions_text_plus_features_scaled.csv",
        "scenario_m_rebalanced_fraud_spam:hybrid_logreg",
    )

    cm_b = confusion_matrix(y_true_b, y_pred_b, labels=LABELS)
    cm_f = confusion_matrix(y_true_f, y_pred_f, labels=LABELS)

    vmax = max(cm_b.max(), cm_f.max())
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.0))
    im1 = axes[0].imshow(cm_b, cmap="Blues", vmin=0, vmax=vmax)
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    axes[0].set_xticks(range(len(LABELS)))
    axes[0].set_yticks(range(len(LABELS)))
    axes[0].set_xticklabels(LABELS, rotation=25, ha="right")
    axes[0].set_yticklabels(LABELS)
    axes[0].set_title("M baseline + LogReg")
    for i in range(cm_b.shape[0]):
        rs = cm_b[i].sum()
        for j in range(cm_b.shape[1]):
            pct = 100.0 * cm_b[i, j] / rs if rs else 0.0
            axes[0].text(
                j, i, f"{cm_b[i, j]}\n{pct:.1f}%", ha="center", va="center", fontsize=9
            )

    axes[1].imshow(cm_f, cmap="Blues", vmin=0, vmax=vmax)
    axes[1].set_xlabel("Predicted class")
    axes[1].set_ylabel("True class")
    axes[1].set_xticks(range(len(LABELS)))
    axes[1].set_yticks(range(len(LABELS)))
    axes[1].set_xticklabels(LABELS, rotation=25, ha="right")
    axes[1].set_yticklabels(LABELS)
    axes[1].set_title("M rebalanced + LogReg (+scale)")
    for i in range(cm_f.shape[0]):
        rs = cm_f[i].sum()
        for j in range(cm_f.shape[1]):
            pct = 100.0 * cm_f[i, j] / rs if rs else 0.0
            axes[1].text(
                j, i, f"{cm_f[i, j]}\n{pct:.1f}%", ha="center", va="center", fontsize=9
            )

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel("Count")
    _save(fig, "exp1_m_confusion_baseline_vs_final")


def plot_final_error_pairs() -> None:
    y_true, y_pred = _load_pred(
        RESULTS / "exp1sender_m_rebalanced_predictions_text_plus_features_scaled.csv",
        "scenario_m_rebalanced_fraud_spam:hybrid_logreg",
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    pairs = []
    for i, true_label in enumerate(LABELS):
        for j, pred_label in enumerate(LABELS):
            if i == j:
                continue
            val = int(cm[i, j])
            if val > 0:
                pairs.append((f"{true_label} -> {pred_label}", val))

    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:8]
    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color="#4f81bd")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of misclassified emails")
    ax.set_title("Final model: most frequent confusion pairs (Exp1)")
    ax.grid(axis="x", alpha=0.25)
    for b, v in zip(bars, values):
        ax.text(v + 0.3, b.get_y() + b.get_height() / 2, str(v), va="center")

    _save(fig, "exp1_final_model_top_confusions")


def main() -> None:
    plot_m_progress()
    plot_confusion_comparison()
    plot_final_error_pairs()
    print("Saved Exp1 section figures to:", OUT)


if __name__ == "__main__":
    main()
