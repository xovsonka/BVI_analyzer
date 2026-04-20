from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 19,
    }
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "graphs_thesis_final"
OUT.mkdir(parents=True, exist_ok=True)


LABELS = ["legit", "phishing", "spam", "financial_fraud"]


MODEL_LABELS = {
    "scenario_l_combined_anti_source:hybrid_sgd_log": "L + SGD log",
    "scenario_m_anti_template_feature_regularized:hybrid_logreg": "M + LogReg",
    "scenario_m_rebalanced_fraud_spam:hybrid_logreg": "M rebalanced + LogReg",
    "scenario_m_rebalanced_fraud_spam:hybrid_logreg_scaled": "M rebalanced + LogReg + scale",
}


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_summary() -> pd.DataFrame:
    lm = pd.read_csv(
        RESULTS / "exp1sender_lm_multiclass_summary_text_plus_features.csv"
    )
    reb = pd.read_csv(
        RESULTS / "exp1sender_m_rebalanced_summary_text_plus_features.csv"
    )
    scaled = pd.read_csv(
        RESULTS / "exp1sender_m_rebalanced_summary_text_plus_features_scaled.csv"
    )
    scaled = scaled.copy()
    scaled["model_config"] = "scenario_m_rebalanced_fraud_spam:hybrid_logreg_scaled"

    keep = [
        "scenario_l_combined_anti_source:hybrid_sgd_log",
        "scenario_m_anti_template_feature_regularized:hybrid_logreg",
        "scenario_m_rebalanced_fraud_spam:hybrid_logreg",
        "scenario_m_rebalanced_fraud_spam:hybrid_logreg_scaled",
    ]
    df = pd.concat([lm, reb, scaled], ignore_index=True, sort=False)
    df = df[df["model_config"].isin(keep)].copy()
    df["model_name"] = df["model_config"].map(MODEL_LABELS)
    order = [MODEL_LABELS[k] for k in keep]
    df["model_name"] = pd.Categorical(df["model_name"], categories=order, ordered=True)
    return df.sort_values("model_name")


def plot_model_overview(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.8))
    x = np.arange(len(df))
    w = 0.23
    ax.bar(x - w, df["mc_f1_macro"], width=w, label="MC F1 macro")
    ax.bar(x, df["mc_balanced_accuracy"], width=w, label="MC balanced acc")
    ax.bar(x + w, df["f1"], width=w, label="Binary F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_name"], rotation=16, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Experiment 1: comparison of current final model candidates")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    save_fig(fig, "exp1_current_model_comparison")


def load_per_class() -> pd.DataFrame:
    lm = pd.read_csv(
        RESULTS / "exp1sender_lm_multiclass_per_class_text_plus_features.csv"
    )
    reb = pd.read_csv(
        RESULTS / "exp1sender_m_rebalanced_per_class_text_plus_features.csv"
    )
    scaled = pd.read_csv(
        RESULTS / "exp1sender_m_rebalanced_per_class_text_plus_features_scaled.csv"
    )
    scaled = scaled.copy()
    scaled["model_config"] = "scenario_m_rebalanced_fraud_spam:hybrid_logreg_scaled"

    keep = [
        "scenario_l_combined_anti_source:hybrid_sgd_log",
        "scenario_m_anti_template_feature_regularized:hybrid_logreg",
        "scenario_m_rebalanced_fraud_spam:hybrid_logreg",
        "scenario_m_rebalanced_fraud_spam:hybrid_logreg_scaled",
    ]

    df = pd.concat([lm, reb, scaled], ignore_index=True, sort=False)
    df = df[df["model_config"].isin(keep)].copy()
    df["model_name"] = df["model_config"].map(MODEL_LABELS)
    return df


def plot_per_class_f1(df: pd.DataFrame) -> None:
    models = [
        MODEL_LABELS["scenario_l_combined_anti_source:hybrid_sgd_log"],
        MODEL_LABELS["scenario_m_anti_template_feature_regularized:hybrid_logreg"],
        MODEL_LABELS["scenario_m_rebalanced_fraud_spam:hybrid_logreg"],
        MODEL_LABELS["scenario_m_rebalanced_fraud_spam:hybrid_logreg_scaled"],
    ]
    cls = ["legit", "phishing", "spam", "financial_fraud"]

    piv = (
        df.pivot_table(
            index="class", columns="model_name", values="f1", aggfunc="first"
        )
        .reindex(cls)
        .reindex(columns=models)
    )

    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    x = np.arange(len(cls))
    w = 0.18
    for i, model in enumerate(models):
        ax.bar(x + (i - 1.5) * w, piv[model].values, width=w, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(cls)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Per-class F1")
    ax.set_title("Experiment 1: per-class F1 after training improvements")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", ncol=2)
    save_fig(fig, "exp1_per_class_f1_current_models")


def _read_pred(path: Path, config: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, low_memory=False)
    part = (
        df[df["model_config"] == config].copy() if "model_config" in df.columns else df
    )
    y_true = part["y_true_multiclass"].astype(str).to_numpy()
    y_pred = part["pred_label"].astype(str).to_numpy()
    return y_true, y_pred


def plot_confusions() -> None:
    base_cfg = "scenario_m_anti_template_feature_regularized:hybrid_logreg"
    reb_cfg = "scenario_m_rebalanced_fraud_spam:hybrid_logreg"

    y_true_base, y_pred_base = _read_pred(
        RESULTS / "exp1sender_lm_multiclass_predictions_text_plus_features.csv",
        base_cfg,
    )
    y_true_reb, y_pred_reb = _read_pred(
        RESULTS / "exp1sender_m_rebalanced_predictions_text_plus_features.csv", reb_cfg
    )
    y_true_scaled, y_pred_scaled = _read_pred(
        RESULTS / "exp1sender_m_rebalanced_predictions_text_plus_features_scaled.csv",
        reb_cfg,
    )

    cms = [
        confusion_matrix(y_true_base, y_pred_base, labels=LABELS),
        confusion_matrix(y_true_reb, y_pred_reb, labels=LABELS),
        confusion_matrix(y_true_scaled, y_pred_scaled, labels=LABELS),
    ]
    titles = ["M baseline", "M rebalanced", "M rebalanced + class scale"]

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.8))
    vmax = max(int(cm.max()) for cm in cms)
    for ax, cm, title in zip(axes, cms, titles):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=25, ha="right")
        ax.set_yticklabels(LABELS)
        ax.set_title(title)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=10)

    axes[0].set_ylabel("True class")
    axes[1].set_xlabel("Predicted class")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel("Count")
    save_fig(fig, "exp1_confusion_matrices_training_evolution")

    # single final confusion matrix for direct thesis inclusion
    cm_final = cms[-1]
    fig2, ax2 = plt.subplots(figsize=(7.2, 6.2))
    im2 = ax2.imshow(cm_final, cmap="Blues")
    ax2.set_xticks(range(len(LABELS)))
    ax2.set_yticks(range(len(LABELS)))
    ax2.set_xticklabels(LABELS, rotation=25, ha="right")
    ax2.set_yticklabels(LABELS)
    ax2.set_xlabel("Predicted class")
    ax2.set_ylabel("True class")
    ax2.set_title("Final model confusion matrix (Exp1 sender-clean)")
    for i in range(cm_final.shape[0]):
        for j in range(cm_final.shape[1]):
            ax2.text(
                j, i, str(int(cm_final[i, j])), ha="center", va="center", fontsize=10
            )
    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    save_fig(fig2, "final_model_confusion_matrix")


def main() -> None:
    summary = load_summary()
    plot_model_overview(summary)
    per_class = load_per_class()
    plot_per_class_f1(per_class)
    plot_confusions()
    print("Saved updated figures to:", OUT)


if __name__ == "__main__":
    main()
