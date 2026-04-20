from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 17,
        "figure.titlesize": 22,
    }
)


SCENARIO_LETTER_RE = re.compile(r"^scenario_([a-z])_")


def save_figure(fig, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")


def scenario_letter(name: str) -> str:
    m = SCENARIO_LETTER_RE.match(str(name))
    return m.group(1) if m else ""


def load_and_merge_summaries(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError("No split summary CSV was found.")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["scenario", "model"], keep="last")
    return merged


def best_per_scenario(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.sort_values(
        ["scenario", "f1_macro_val_iid"], ascending=[True, False]
    )
    best = ranked.groupby("scenario", as_index=False).first()
    best["scenario_letter"] = best["scenario"].map(scenario_letter)
    best = best.sort_values("scenario_letter")
    return best


def plot_scenario_evolution(best: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        "f1_macro_test_hard_source",
        "f1_macro_test_hard_cluster",
        "f1_macro_test_deployment",
    ]
    labels = ["test_hard_source", "test_hard_cluster", "test_deployment"]

    x = np.arange(len(best))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (c, lbl) in enumerate(zip(cols, labels)):
        ax.bar(x + (i - 1) * width, best[c], width=width, label=lbl)

    tick_labels = [
        f"{r.scenario_letter.upper()}\n{r.model.replace('hybrid_', '')}"
        for r in best.itertuples(index=False)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Macro F1")
    ax.set_title("Scenario Evolution (A->M): best model per scenario")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_dir / "scenario_evolution_A_to_M.png")
    plt.close(fig)


def plot_lm_hard_grouped(summary: pd.DataFrame, out_dir: Path) -> None:
    lm = summary[
        summary["scenario"].isin(
            [
                "scenario_l_combined_anti_source",
                "scenario_m_anti_template_feature_regularized",
            ]
        )
    ].copy()
    if lm.empty:
        return

    models = [
        "hybrid_logreg",
        "hybrid_linear_svc_cal",
        "hybrid_sgd_log",
        "hybrid_sgd_hinge",
    ]
    split_cols = ["f1_macro_test_hard_source", "f1_macro_test_hard_cluster"]
    split_titles = ["Hard Source", "Hard Cluster"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharey=True)
    for ax, split_col, split_title in zip(axes, split_cols, split_titles):
        vals_l = []
        vals_m = []
        for m in models:
            lrow = lm[
                (lm["scenario"] == "scenario_l_combined_anti_source")
                & (lm["model"] == m)
            ]
            mrow = lm[
                (lm["scenario"] == "scenario_m_anti_template_feature_regularized")
                & (lm["model"] == m)
            ]
            vals_l.append(float(lrow.iloc[0][split_col]) if not lrow.empty else np.nan)
            vals_m.append(float(mrow.iloc[0][split_col]) if not mrow.empty else np.nan)

        x = np.arange(len(models))
        w = 0.36
        bars_l = ax.bar(x - w / 2, vals_l, width=w, label="L", color="#1f77b4")
        bars_m = ax.bar(x + w / 2, vals_m, width=w, label="M", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("hybrid_", "") for m in models],
            rotation=15,
            ha="right",
            fontsize=16,
        )
        ax.set_title(split_title, fontsize=18)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars_l, fmt="%.3f", fontsize=13, padding=3)
        ax.bar_label(bars_m, fmt="%.3f", fontsize=13, padding=3)
        ax.tick_params(axis="y", labelsize=15)

    axes[0].set_ylabel("Macro F1", fontsize=18)
    axes[0].legend(fontsize=16)
    fig.suptitle("L vs M detail by model on hard splits", fontsize=22)
    fig.tight_layout()
    save_figure(fig, out_dir / "lm_hard_grouped_models.png")
    plt.close(fig)


def plot_single_vs_two_stage(
    summary: pd.DataFrame, two_stage_csv: Path, out_dir: Path
) -> None:
    if not two_stage_csv.exists():
        return
    two = pd.read_csv(two_stage_csv)
    if two.empty:
        return

    rows = []
    for r in two.itertuples(index=False):
        one = summary[
            (summary["scenario"] == r.scenario) & (summary["model"] == r.stage1_model)
        ]
        if one.empty:
            one = summary[
                (summary["scenario"] == r.scenario)
                & (summary["model"] == "hybrid_sgd_hinge")
            ]
        if one.empty:
            continue
        o = one.iloc[0]
        rows.append(
            {
                "scenario": r.scenario,
                "single_f1": float(o["f1_macro_test_iid"]),
                "two_f1": float(r.test_iid_f1_macro),
                "single_bal_acc": float(o["balanced_accuracy_test_iid"]),
                "two_bal_acc": float(r.test_iid_balanced_accuracy),
            }
        )

    if not rows:
        return

    df = pd.DataFrame(rows)
    df["scenario_short"] = (
        df["scenario"]
        .str.replace("scenario_", "", regex=False)
        .str.replace("_anti_template_feature_regularized", "_m", regex=False)
        .str.replace("_combined_anti_source", "_l", regex=False)
    )
    x = np.arange(len(df))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    b1 = axes[0].bar(
        x - w / 2,
        df["single_f1"],
        width=w,
        label="single-stage",
        color="#1f77b4",
    )
    b2 = axes[0].bar(
        x + w / 2,
        df["two_f1"],
        width=w,
        label="two-stage",
        color="#ff7f0e",
    )
    axes[0].set_title("Macro F1 (test_iid)", fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["scenario_short"], rotation=15, ha="right")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].bar_label(b1, fmt="%.3f", fontsize=9, padding=2)
    axes[0].bar_label(b2, fmt="%.3f", fontsize=9, padding=2)

    b3 = axes[1].bar(
        x - w / 2,
        df["single_bal_acc"],
        width=w,
        label="single-stage",
        color="#1f77b4",
    )
    b4 = axes[1].bar(
        x + w / 2,
        df["two_bal_acc"],
        width=w,
        label="two-stage",
        color="#ff7f0e",
    )
    axes[1].set_title("Balanced Accuracy (test_iid)", fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["scenario_short"], rotation=15, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].bar_label(b3, fmt="%.3f", fontsize=9, padding=2)
    axes[1].bar_label(b4, fmt="%.3f", fontsize=9, padding=2)

    axes[0].set_ylabel("Score", fontsize=12)
    axes[0].legend(fontsize=11)
    fig.suptitle("Single-stage vs Two-stage", fontsize=14)
    fig.tight_layout()
    save_figure(fig, out_dir / "single_vs_two_stage_f1_balacc.png")
    plt.close(fig)


def plot_text_view_modes(text_view_csv: Path, out_dir: Path) -> None:
    if not text_view_csv.exists():
        return
    df = pd.read_csv(text_view_csv)
    if df.empty:
        return

    piv = df.pivot_table(
        index="split", columns="view", values="f1_macro", aggfunc="first"
    )
    split_order = [
        "val_iid",
        "test_iid",
        "test_hard_source",
        "test_hard_cluster",
        "test_deployment",
    ]
    piv = piv.reindex([s for s in split_order if s in piv.index])

    x = np.arange(len(piv.index))
    views = [
        c for c in ["body_only", "header_plus_body", "full_text"] if c in piv.columns
    ]
    if not views:
        return
    w = 0.24

    fig, ax = plt.subplots(figsize=(13, 7))
    for i, v in enumerate(views):
        ax.bar(x + (i - (len(views) - 1) / 2) * w, piv[v], width=w, label=v)

    ax.set_xticks(x)
    ax.set_xticklabels(list(piv.index), rotation=20, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Text view comparison")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_dir / "text_view_comparison_grouped.png")
    plt.close(fig)


def load_detail_json(details_dir: Path, scenario: str, model: str) -> dict:
    fp = details_dir / f"{scenario}__{model}.json"
    if not fp.exists():
        raise FileNotFoundError(f"Missing detail file: {fp}")
    return json.loads(fp.read_text(encoding="utf-8"))


def plot_confusion_matrix_final(detail: dict, split: str, out_dir: Path) -> None:
    split_data = detail["splits"].get(split)
    if not split_data:
        return
    cm = np.asarray(split_data["confusion_matrix"], dtype=float)
    labels = split_data["labels"]
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=16)
    ax.set_yticklabels(labels, fontsize=16)
    ax.set_xlabel("Predicted label", fontsize=18)
    ax.set_ylabel("True label", fontsize=18)
    ax.set_title(f"Final model confusion matrix ({split}, row-normalized)", fontsize=22)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            color = "white" if val >= 0.6 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=14,
                color=color,
                fontweight="bold" if val >= 0.6 else "normal",
            )

    highlight_rows = [
        labels.index(x) for x in ["spam", "financial_fraud"] if x in labels
    ]
    for r in highlight_rows:
        rect = Rectangle(
            (-0.5, r - 0.5), len(labels), 1, fill=False, edgecolor="red", linewidth=2
        )
        ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    fig.tight_layout()
    save_figure(fig, out_dir / "final_model_confusion_matrix.png")
    plt.close(fig)


def plot_per_class_recall(detail: dict, out_dir: Path) -> None:
    split_order = [
        "test_iid",
        "test_hard_source",
        "test_hard_cluster",
        "test_deployment",
    ]
    rows = []

    for split in split_order:
        split_data = detail["splits"].get(split)
        if not split_data:
            continue
        report = split_data.get("classification_report", {})
        for cls, stats in report.items():
            if cls in {"accuracy", "macro avg", "weighted avg"}:
                continue
            if isinstance(stats, dict) and "recall" in stats:
                rows.append(
                    {"split": split, "label": cls, "recall": float(stats["recall"])}
                )

    if not rows:
        return

    df = pd.DataFrame(rows)
    classes = ["financial_fraud", "spam", "phishing", "legit"]
    classes = [c for c in classes if c in set(df["label"])] + [
        c for c in sorted(df["label"].unique()) if c not in classes
    ]
    piv = df.pivot_table(
        index="label", columns="split", values="recall", aggfunc="first"
    )
    piv = piv.reindex(classes)
    splits = [s for s in split_order if s in piv.columns]
    piv = piv[splits]

    x = np.arange(len(piv.index))
    w = 0.18

    fig, ax = plt.subplots(figsize=(15, 7.5))
    for i, s in enumerate(splits):
        bars = ax.bar(
            x + (i - (len(splits) - 1) / 2) * w,
            piv[s],
            width=w,
            label=s,
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(list(piv.index), rotation=15, ha="right", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Per-class recall across evaluation splits", fontsize=14)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=10)
    fig.tight_layout()
    save_figure(fig, out_dir / "final_model_per_class_recall.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate requested thesis experiment plots"
    )
    parser.add_argument(
        "--summary-csvs",
        nargs="+",
        default=[
            "results/split_suite_text_only_ghijk_summary.csv",
            "results/split_suite_text_only_lm_final_summary.csv",
        ],
        help="One or more split suite summary CSV files to merge",
    )
    parser.add_argument("--text-view-csv", default="results/text_view_modes_j.csv")
    parser.add_argument(
        "--two-stage-csv", default="results/two_stage_text_only/summary.csv"
    )
    parser.add_argument("--details-dir", default="results/split_suite_details")
    parser.add_argument(
        "--final-scenario",
        default="scenario_m_anti_template_feature_regularized",
    )
    parser.add_argument("--final-model", default="hybrid_sgd_hinge")
    parser.add_argument("--final-split", default="test_deployment")
    parser.add_argument("--out-dir", default="results/graphs_thesis_final")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = [Path(p) for p in args.summary_csvs]
    summary = load_and_merge_summaries(summary_paths)

    best = best_per_scenario(summary)
    plot_scenario_evolution(best, out_dir)
    plot_lm_hard_grouped(summary, out_dir)
    plot_single_vs_two_stage(summary, Path(args.two_stage_csv), out_dir)
    plot_text_view_modes(Path(args.text_view_csv), out_dir)

    detail = load_detail_json(
        Path(args.details_dir), args.final_scenario, args.final_model
    )
    plot_confusion_matrix_final(detail, args.final_split, out_dir)
    plot_per_class_recall(detail, out_dir)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
