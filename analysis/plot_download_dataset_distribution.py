from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 22,
        "legend.title_fontsize": 24,
        "figure.titlesize": 24,
    }
)


def read_csv_fallback(csv_path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin1"):
        try:
            return pd.read_csv(csv_path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(csv_path, low_memory=False)


def collect_summary(source_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(source_dir.rglob("*.csv")):
        df = read_csv_fallback(csv_path)
        rel = csv_path.relative_to(source_dir)
        source_group = rel.parts[0] if rel.parts else "unknown"
        rows.append(
            {
                "source_group": source_group,
                "file": str(rel),
                "rows": int(len(df)),
            }
        )
    if not rows:
        return pd.DataFrame(
            {
                "source_group": pd.Series(dtype="string"),
                "file": pd.Series(dtype="string"),
                "rows": pd.Series(dtype="int64"),
            }
        )
    return pd.DataFrame(rows)


def plot_distribution(summary_df: pd.DataFrame, out_png: Path) -> None:
    by_group = (
        summary_df.groupby("source_group", as_index=False)
        .agg(total_rows=("rows", "sum"), files=("file", "count"))
        .sort_values(by="total_rows", ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(by_group["source_group"], by_group["total_rows"], color="#1f77b4")
    axes[0].set_title("Rows by downloaded source group")
    axes[0].set_ylabel("Number of rows")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(by_group["source_group"], by_group["files"], color="#ff7f0e")
    axes[1].set_title("CSV files by downloaded source group")
    axes[1].set_ylabel("Number of CSV files")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Dataset distribution right after download")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_percentage_pie(
    summary_df: pd.DataFrame,
    out_pie_png: Path,
    out_pie_pdf: Path | None = None,
) -> None:
    pie_df = summary_df.copy()
    pie_df["label"] = pie_df.apply(
        lambda r: Path(r["file"]).stem
        if str(r["source_group"]) == "phishing"
        else str(r["source_group"]),
        axis=1,
    )

    agg = (
        pie_df.groupby("label", as_index=False)
        .agg(total_rows=("rows", "sum"))
        .sort_values(by="total_rows", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 7.5))
    pie_out = ax.pie(
        agg["total_rows"],
        labels=None,
        startangle=120,
        radius=1.12,
        textprops={"fontsize": 13},
    )
    wedges = pie_out[0]

    total = float(agg["total_rows"].sum())
    legend_labels = [
        f"{lbl} ({(rows / total) * 100:.1f}%)"
        for lbl, rows in zip(agg["label"], agg["total_rows"])
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Sources / datasets",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=20,
        title_fontsize=22,
    )

    ax.set_title(
        "Percentual distribution after download (phishing split by dataset)",
        fontsize=18,
        pad=18,
    )
    ax.axis("equal")
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    fig.savefig(out_pie_png, dpi=400, bbox_inches="tight")
    if out_pie_pdf is not None:
        fig.savefig(out_pie_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot distribution of raw downloaded datasets"
    )
    parser.add_argument("--source-dir", default="dataset/source")
    parser.add_argument(
        "--output-png",
        default="results/graphs_thesis_final/download_dataset_distribution.png",
    )
    parser.add_argument(
        "--output-csv",
        default="results/download_dataset_distribution_summary.csv",
    )
    parser.add_argument(
        "--output-pie-png",
        default="results/graphs_thesis_final/download_dataset_distribution_pie.png",
    )
    parser.add_argument(
        "--output-pie-pdf",
        default="results/graphs_thesis_final/download_dataset_distribution_pie.pdf",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source dir: {source_dir}")

    summary_df = collect_summary(source_dir)
    if summary_df.empty:
        raise RuntimeError(f"No CSV files found under: {source_dir}")

    out_png = Path(args.output_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_pie_png = Path(args.output_pie_png)
    out_pie_png.parent.mkdir(parents=True, exist_ok=True)
    out_pie_pdf = Path(args.output_pie_pdf)
    out_pie_pdf.parent.mkdir(parents=True, exist_ok=True)

    plot_distribution(summary_df, out_png)
    plot_percentage_pie(summary_df, out_pie_png, out_pie_pdf)
    summary_df.to_csv(out_csv, index=False)

    by_group = (
        summary_df.groupby("source_group", as_index=False)
        .agg(total_rows=("rows", "sum"), files=("file", "count"))
        .sort_values(by="total_rows", ascending=False)
    )
    print(by_group.to_string(index=False))
    print(f"Saved plot: {out_png}")
    print(f"Saved pie plot: {out_pie_png}")
    print(f"Saved pie PDF: {out_pie_pdf}")
    print(f"Saved summary: {out_csv}")


if __name__ == "__main__":
    main()
