from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.size": 16,
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


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_lm_hard_grouped() -> None:
    l_df = pd.read_csv(RESULTS / "split_suite_text_plus_features_l_sgd_current.csv")
    m_df = pd.read_csv(RESULTS / "split_suite_text_plus_features_m_current.csv")

    rows = []
    for _, row in l_df.iterrows():
        rows.append(
            {
                "model_name": "L + SGD log",
                "split": row["split"],
                "f1_macro": row["f1_macro"],
            }
        )

    for _, row in m_df.iterrows():
        model = row["model"]
        display = "M + LogReg" if model == "hybrid_logreg" else "M + SGD log"
        rows.append(
            {
                "model_name": display,
                "split": row["split"],
                "f1_macro": row["f1_macro"],
            }
        )

    df = pd.DataFrame(rows)
    split_order = ["test_hard_source", "test_hard_cluster", "test_deployment"]
    models = ["L + SGD log", "M + SGD log", "M + LogReg"]
    piv = (
        df[df["split"].isin(split_order)]
        .pivot_table(
            index="split", columns="model_name", values="f1_macro", aggfunc="first"
        )
        .reindex(split_order)
        .reindex(columns=models)
    )

    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    x = np.arange(len(split_order))
    w = 0.24
    for i, model in enumerate(models):
        vals = piv[model].to_numpy()
        ax.bar(x + (i - 1) * w, vals, width=w, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("test_", "") for s in split_order])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Macro F1")
    ax.set_title("L/M hard-split comparison (training/testing datasets)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    save(fig, "lm_hard_grouped_models")


def plot_final_confusion_from_test_deployment() -> None:
    detail_path = (
        RESULTS
        / "split_suite_details"
        / "scenario_m_anti_template_feature_regularized__hybrid_logreg.json"
    )
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    split = detail["splits"]["test_deployment"]
    cm = np.array(split["confusion_matrix"], dtype=int)
    labels = split["labels"]

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Final model confusion matrix on test_deployment")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "final_model_confusion_matrix_test_deployment")


def main() -> None:
    plot_lm_hard_grouped()
    plot_final_confusion_from_test_deployment()
    print("Saved training/testing-only figures to:", OUT)


if __name__ == "__main__":
    main()
