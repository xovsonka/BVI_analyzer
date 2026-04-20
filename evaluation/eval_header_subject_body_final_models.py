from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 18,
        "legend.title_fontsize": 19,
        "figure.titlesize": 22,
    }
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import build_model, select_model_input


def parse_configs(values: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for v in values:
        if ":" not in v:
            raise ValueError(f"Invalid config '{v}'. Use scenario:model format.")
        sc, model = v.split(":", 1)
        out.append((sc.strip(), model.strip()))
    return out


def load_shared_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    shared = processed_dir / "shared"
    candidates = {
        "val_iid": [shared / "val_iid.csv", shared / "val.csv"],
        "test_iid": [shared / "test_iid.csv", shared / "test.csv"],
        "test_hard_source": [shared / "test_hard_source.csv"],
        "test_hard_cluster": [shared / "test_hard_cluster.csv"],
        "test_deployment": [shared / "test_deployment.csv"],
    }
    out: dict[str, pd.DataFrame] = {}
    for split, paths in candidates.items():
        fp = next((p for p in paths if p.exists()), None)
        if fp is not None:
            out[split] = pd.read_csv(fp, low_memory=False)
    return out


def _safe_text_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[col].fillna("").astype(str)


def build_header_text(df: pd.DataFrame) -> pd.Series:
    cols = [
        "sender",
        "receiver",
        "date",
        "from_domain",
        "reply_to_domain",
        "return_path_domain",
        "spf_result",
        "dkim_result",
        "dmarc_result",
    ]
    out = []
    for _, row in df.iterrows():
        tokens = []
        for c in cols:
            v = row[c] if c in row else ""
            if pd.notna(v) and str(v).strip() != "":
                tokens.append(f"{c}:{str(v).strip()}")
        out.append(" ".join(tokens).strip())
    return pd.Series(out, index=df.index)


def view_text(df: pd.DataFrame, view: str) -> pd.Series:
    if view == "header_only":
        return build_header_text(df)
    if view == "subject_only":
        return _safe_text_col(df, "subject")
    if view == "body_only":
        return _safe_text_col(df, "body")
    raise ValueError(f"Unknown view: {view}")


def plot_body_selection(df: pd.DataFrame, out_png: Path) -> None:
    split_order = [
        "val_iid",
        "test_iid",
        "test_hard_source",
        "test_hard_cluster",
        "test_deployment",
    ]
    models = list(df["model_config"].drop_duplicates())
    fig, axes = plt.subplots(
        len(models), 1, figsize=(13, 6.8 * len(models)), sharey=True
    )
    if len(models) == 1:
        axes = [axes]

    for ax, mc in zip(axes, models):
        part = df[df["model_config"] == mc].copy()
        piv = part.pivot_table(
            index="split", columns="view", values="f1_macro", aggfunc="first"
        )
        piv = piv.reindex([s for s in split_order if s in piv.index])
        x = range(len(piv.index))
        for view in ["header_only", "subject_only", "body_only"]:
            if view in piv.columns:
                ax.plot(
                    list(x),
                    piv[view],
                    marker="o",
                    markersize=7,
                    linewidth=2.2,
                    label=view,
                )
        ax.set_xticks(list(x))
        ax.set_xticklabels(list(piv.index), rotation=15, ha="right", fontsize=15)
        ax.set_title(mc)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Macro F1", fontsize=17)
        ax.tick_params(axis="y", labelsize=15)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=18, title="Input view", title_fontsize=19, loc="lower left")

    fig.suptitle("Header vs Subject vs Body on final model candidates")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare header/subject/body views for final model candidates"
    )
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "scenario_l_combined_anti_source:hybrid_linear_svc_cal",
            "scenario_m_anti_template_feature_regularized:hybrid_sgd_hinge",
        ],
        help="List of scenario:model configs",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=["header_only", "subject_only", "body_only"],
        choices=["header_only", "subject_only", "body_only"],
    )
    parser.add_argument(
        "--output-csv",
        default="results/final_models_header_subject_body.csv",
    )
    parser.add_argument(
        "--output-json",
        default="results/final_models_header_subject_body.json",
    )
    parser.add_argument(
        "--output-plot",
        default="results/graphs_thesis_final/final_models_header_subject_body.png",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    splits = load_shared_splits(processed_dir)
    if not splits:
        raise RuntimeError("No shared splits found in dataset/processed/shared")

    configs = parse_configs(args.configs)
    all_rows: list[dict] = []

    for scenario, model_name in configs:
        train_path = processed_dir / scenario / "train.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing train file: {train_path}")
        train_df = pd.read_csv(train_path, low_memory=False)

        labels = sorted(set(train_df["label"].astype(str)))
        for d in splits.values():
            labels = sorted(set(labels).union(set(d["label"].astype(str))))
        le = LabelEncoder()
        le.fit(labels)
        y_train = le.transform(train_df["label"].astype(str))

        for view in args.views:
            train_input = pd.DataFrame({"text_input": view_text(train_df, view)})
            x_train = select_model_input(train_input, input_mode="text_only")
            model = build_model(model_name, input_mode="text_only")
            model.fit(x_train, y_train)

            for split_name, split_df in splits.items():
                split_input = pd.DataFrame({"text_input": view_text(split_df, view)})
                x_test = select_model_input(split_input, input_mode="text_only")
                y_test = le.transform(split_df["label"].astype(str))
                pred = model.predict(x_test)
                row = {
                    "scenario": scenario,
                    "model": model_name,
                    "model_config": f"{scenario} | {model_name}",
                    "view": view,
                    "split": split_name,
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                    "f1_macro": float(
                        f1_score(y_test, pred, average="macro", zero_division=0)
                    ),
                    "f1_weighted": float(
                        f1_score(y_test, pred, average="weighted", zero_division=0)
                    ),
                    "rows": int(len(split_df)),
                }
                all_rows.append(row)
                print(
                    f"{scenario:<44} {model_name:<22} {view:<12} {split_name:<16} f1_macro={row['f1_macro']:.4f}"
                )

    out_df = pd.DataFrame(all_rows)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    plot_body_selection(out_df, Path(args.output_plot))

    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")
    print(f"Saved plot: {args.output_plot}")


if __name__ == "__main__":
    main()
