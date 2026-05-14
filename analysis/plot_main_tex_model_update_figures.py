from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["legit", "phishing", "spam", "financial_fraud"]


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


def _binary_mcc_from_pred_csv(path: Path) -> float:
    df = pd.read_csv(path, low_memory=False)
    yt = df["y_true_binary"].astype(str).str.lower().eq("suspicious").astype(int)
    yp = df["ml_pred_binary"].astype(str).str.lower().eq("suspicious").astype(int)
    return float(matthews_corrcoef(yt, yp))


def plot_exp1_model_comparison() -> None:
    binary_df = pd.read_csv(ROOT / "results" / "experiment_5" / "exp5_method_compare_binary.csv")
    heur = binary_df[(binary_df["split"] == "source:exp1_campaign") & (binary_df["method"] == "heuristic_only")].iloc[0]
    tuned = pd.read_csv(ROOT / "results" / "retuned_exp1234" / "exp1_summary_tuned_equalscale.csv").iloc[0]
    adapted = pd.read_csv(
        ROOT / "results" / "retuned_exp1234" / "semantic_cta_eval" / "exp1_spamboost_ps160_fs160_summary.csv"
    ).iloc[0]

    data = pd.DataFrame(
        [
            {
                "model": "Heuristic",
                "binary_f1": float(heur["f1"]),
                "balanced_accuracy": float(heur["balanced_accuracy"]),
                "mcc": float(heur["mcc"]),
            },
            {
                "model": "Tuned single-stage",
                "binary_f1": float(tuned["bin_f1"]),
                "balanced_accuracy": float(tuned["bin_balanced_accuracy"]),
                "mcc": _binary_mcc_from_pred_csv(
                    ROOT / "results" / "retuned_exp1234" / "exp1_predictions_tuned_equalscale.csv"
                ),
            },
            {
                "model": "Semantic OVR",
                "binary_f1": float(adapted["bin_f1"]),
                "balanced_accuracy": float(adapted["bin_balanced_accuracy"]),
                "mcc": _binary_mcc_from_pred_csv(
                    ROOT / "results" / "retuned_exp1234" / "semantic_cta_eval" / "exp1_spamboost_ps160_fs160_pred.csv"
                ),
            },
        ]
    )

    metrics = ["binary_f1", "balanced_accuracy", "mcc"]
    metric_labels = ["Binary F1", "Balanced acc.", "MCC"]
    x = np.arange(len(data))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    colors = ["#6b8db3", "#2f5f98", "#1b3d6d"]
    for idx, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        vals = data[metric].to_numpy()
        bars = ax.bar(x + (idx - 1) * width, vals, width=width, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(data["model"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Experiment 1: heuristic, tuned baseline, and semantic OVR")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    _save(fig, "exp1_model_comparison")


def plot_exp1_multiclass_comparison() -> None:
    mc_df = pd.read_csv(ROOT / "results" / "experiment_5" / "exp5_method_compare_multiclass.csv")
    heur = mc_df[(mc_df["split"] == "source:exp1_campaign") & (mc_df["method"] == "heuristic_only")].iloc[0]
    tuned = pd.read_csv(ROOT / "results" / "retuned_exp1234" / "exp1_summary_tuned_equalscale.csv").iloc[0]
    adapted = pd.read_csv(
        ROOT / "results" / "retuned_exp1234" / "semantic_cta_eval" / "exp1_spamboost_ps160_fs160_summary.csv"
    ).iloc[0]

    data = pd.DataFrame(
        [
            {"model": "Heuristic", "mc_f1_macro": float(heur["mc_f1_macro"]), "mc_balanced_accuracy": float(heur["mc_balanced_accuracy"])},
            {"model": "Tuned single-stage", "mc_f1_macro": float(tuned["mc_f1_macro"]), "mc_balanced_accuracy": float(tuned["mc_balanced_accuracy"])},
            {"model": "Semantic OVR", "mc_f1_macro": float(adapted["mc_f1_macro"]), "mc_balanced_accuracy": float(adapted["mc_balanced_accuracy"])},
        ]
    )

    x = np.arange(len(data))
    width = 0.32
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    bars1 = ax.bar(x - width / 2, data["mc_f1_macro"], width=width, label="MC F1 macro", color="#4f81bd")
    bars2 = ax.bar(x + width / 2, data["mc_balanced_accuracy"], width=width, label="MC balanced acc.", color="#7aa6d1")
    for bars in [bars1, bars2]:
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(data["model"])
    ax.set_ylim(0.0, 0.9)
    ax.set_ylabel("Score")
    ax.set_title("Experiment 1: multiclass comparison of current candidates")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    _save(fig, "exp1_multiclass_method_comparison")


def plot_exp1_confusion() -> None:
    df = pd.read_csv(ROOT / "results" / "retuned_exp1234" / "semantic_cta_eval" / "exp1_spamboost_ps160_fs160_pred.csv", low_memory=False)
    y_true = df["y_true_multiclass"].astype(str).to_numpy()
    y_pred = df["ml_pred"].astype(str).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=25, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Experiment 1: semantic OVR confusion matrix")
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        for j in range(cm.shape[1]):
            pct = 100.0 * cm[i, j] / row_sum if row_sum else 0.0
            ax.text(j, i, f"{cm[i, j]}\n{pct:.1f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "exp1_m_confusion_best_final")


def plot_exp4_offline_comparison() -> None:
    tuned = pd.read_csv(ROOT / "results" / "m_shap_optimization_m_original" / "metrics_baseline_vs_tuned.csv")
    tuned = tuned[tuned["variant"] == "tuned"].copy()
    tuned = tuned[tuned["split"].isin(["test_iid", "test_hard_source", "test_hard_cluster", "test_deployment"])]
    tuned["variant"] = "Tuned single-stage"
    tuned = tuned.rename(columns={"f1_macro": "score"})[["split", "variant", "score"]]

    adapted = pd.read_csv(ROOT / "results" / "retuned_exp1234" / "semantic_cta_offline" / "offline_summary.csv")
    adapted = adapted[adapted["split"].isin(["test", "test_hard_source", "test_hard_cluster", "test_deployment"])]
    adapted = adapted.copy()
    adapted["split"] = adapted["split"].replace({"test": "test_iid"})
    adapted["variant"] = "Semantic OVR"
    adapted = adapted.rename(columns={"mc_f1_macro": "score"})[["split", "variant", "score"]]

    order = ["test_iid", "test_hard_source", "test_hard_cluster", "test_deployment"]
    merged = pd.concat([tuned, adapted], ignore_index=True)
    piv = merged.pivot_table(index="split", columns="variant", values="score", aggfunc="first").reindex(order)

    x = np.arange(len(order))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    bars1 = ax.bar(x - width / 2, piv["Tuned single-stage"].to_numpy(), width=width, label="Tuned single-stage", color="#2a9d8f")
    bars2 = ax.bar(x + width / 2, piv["Semantic OVR"].to_numpy(), width=width, label="Semantic OVR", color="#e76f51")
    for bars in [bars1, bars2]:
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["test_iid", "hard_source", "hard_cluster", "deployment"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Macro F1")
    ax.set_title("Experiment 4 Track A: general tuned vs semantic OVR")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    _save(fig, "exp4_trackA_deployment_macro_f1")


def plot_lm_hard_grouped_models() -> None:
    candidates = pd.read_csv(ROOT / "results" / "experiment_4" / "exp4_final_candidates_table.csv")
    tuned = pd.read_csv(ROOT / "results" / "m_shap_optimization_m_original" / "metrics_baseline_vs_tuned.csv")
    tuned = tuned[tuned["variant"] == "tuned"].copy().set_index("split")
    semantic = pd.read_csv(ROOT / "results" / "retuned_exp1234" / "semantic_cta_offline" / "offline_summary.csv")
    semantic = semantic.copy().set_index("split")

    rows = [
        {
            "model": "L + SGD",
            "split": "test_hard_source",
            "score": float(
                candidates[(candidates["scenario"] == "scenario_l_combined_anti_source") & (candidates["model"] == "hybrid_sgd_log")]["f1_macro_test_hard_source"].iloc[0]
            ),
        },
        {
            "model": "L + SGD",
            "split": "test_hard_cluster",
            "score": float(
                candidates[(candidates["scenario"] == "scenario_l_combined_anti_source") & (candidates["model"] == "hybrid_sgd_log")]["f1_macro_test_hard_cluster"].iloc[0]
            ),
        },
        {
            "model": "L + SGD",
            "split": "test_deployment",
            "score": float(
                candidates[(candidates["scenario"] == "scenario_l_combined_anti_source") & (candidates["model"] == "hybrid_sgd_log")]["f1_macro_test_deployment"].iloc[0]
            ),
        },
        {"model": "Tuned single-stage", "split": "test_hard_source", "score": float(tuned.loc["test_hard_source", "f1_macro"])} ,
        {"model": "Tuned single-stage", "split": "test_hard_cluster", "score": float(tuned.loc["test_hard_cluster", "f1_macro"])} ,
        {"model": "Tuned single-stage", "split": "test_deployment", "score": float(tuned.loc["test_deployment", "f1_macro"])} ,
        {"model": "Semantic OVR", "split": "test_hard_source", "score": float(semantic.loc["test_hard_source", "mc_f1_macro"])} ,
        {"model": "Semantic OVR", "split": "test_hard_cluster", "score": float(semantic.loc["test_hard_cluster", "mc_f1_macro"])} ,
        {"model": "Semantic OVR", "split": "test_deployment", "score": float(semantic.loc["test_deployment", "mc_f1_macro"])} ,
    ]
    df = pd.DataFrame(rows)
    order = ["test_hard_source", "test_hard_cluster", "test_deployment"]
    models = ["L + SGD", "Tuned single-stage", "Semantic OVR"]
    piv = df.pivot_table(index="split", columns="model", values="score", aggfunc="first").reindex(order).reindex(columns=models)

    fig, ax = plt.subplots(figsize=(11.6, 6.0))
    x = np.arange(len(order))
    width = 0.24
    colors = ["#6b8db3", "#4f81bd", "#1d3f6e"]
    for idx, (model, color) in enumerate(zip(models, colors)):
        vals = piv[model].to_numpy()
        bars = ax.bar(x + (idx - 1) * width, vals, width=width, label=model, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["hard_source", "hard_cluster", "deployment"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Macro F1")
    ax.set_title("Final robust model candidates on hard and deployment splits")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    _save(fig, "lm_hard_grouped_models")


def plot_final_model_confusion_deployment() -> None:
    conf = pd.read_csv(ROOT / "results" / "retuned_exp1234" / "semantic_cta_eval" / "semantic_spamboost_test_deployment_conf.csv")
    labels = LABELS
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for row in conf.itertuples(index=False):
        if row.true_label in label_to_idx and row.pred_label in label_to_idx:
            cm[label_to_idx[row.true_label], label_to_idx[row.pred_label]] = int(row.count)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Semantic OVR model on test_deployment")
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        for j in range(cm.shape[1]):
            pct = 100.0 * cm[i, j] / row_sum if row_sum else 0.0
            ax.text(j, i, f"{cm[i, j]}\n{pct:.1f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "final_model_confusion_matrix_test_deployment")


def plot_exp3_url_correlation() -> None:
    corr = pd.read_csv(
        ROOT / "results" / "retuned_exp1234" / "adapted_ovr_reports" / "exp3_campaign_url_score_correlation.csv"
    )
    corr = corr.copy()
    corr["indicator"] = corr["indicator"].replace(
        {
            "ip_url": "ip_url",
            "shortener_url": "shortener_url",
            "anchor_mismatch": "anchor_mismatch",
            "suspicious_tld": "suspicious_tld",
            "brand_typosquat": "brand_typosquat",
            "obfuscated_url": "obfuscated_url",
        }
    )
    corr = corr.sort_values("spearman_corr_with_score", ascending=False)

    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in corr["spearman_corr_with_score"]]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    bars = ax.bar(corr["indicator"], corr["spearman_corr_with_score"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Spearman correlation")
    ax.set_ylim(-0.5, 0.7)
    ax.set_title("Experiment 3: URL indicator correlation with heuristic score")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    for b, v in zip(bars, corr["spearman_corr_with_score"]):
        offset = 0.02 if v >= 0 else -0.05
        ax.text(b.get_x() + b.get_width() / 2, v + offset, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    _save(fig, "exp3_url_indicator_spearman_correlation")


def main() -> None:
    plot_exp1_model_comparison()
    plot_exp1_multiclass_comparison()
    plot_exp1_confusion()
    plot_exp3_url_correlation()
    plot_exp4_offline_comparison()
    plot_lm_hard_grouped_models()
    plot_final_model_confusion_deployment()
    print("Saved updated main.tex figures to:", ROOT)


if __name__ == "__main__":
    main()
