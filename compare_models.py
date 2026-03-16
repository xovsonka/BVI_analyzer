from pathlib import Path
import csv
import json


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_CSV = MODELS_DIR / "model_comparison.csv"


def load_metrics_files(models_dir: Path):
    return sorted(models_dir.glob("*_metrics.json"))


def extract_row(metrics: dict):
    test = metrics.get("test", {})
    cm = test.get("confusion_matrix", [[None, None], [None, None]])
    tn = cm[0][0] if len(cm) > 0 and len(cm[0]) > 0 else None
    fp = cm[0][1] if len(cm) > 0 and len(cm[0]) > 1 else None
    fn = cm[1][0] if len(cm) > 1 and len(cm[1]) > 0 else None
    tp = cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else None

    return {
        "model": metrics.get("model", "unknown"),
        "best_threshold": metrics.get("best_threshold"),
        "validation_best_f1": metrics.get("validation_best_f1"),
        "test_accuracy": test.get("accuracy"),
        "test_precision_label1": test.get("precision_label1"),
        "test_recall_label1": test.get("recall_label1"),
        "test_f1_label1": test.get("f1_label1"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def print_table(rows: list[dict]):
    if not rows:
        print("No metrics files found.")
        return

    rows_sorted = sorted(
        rows,
        key=lambda r: -1.0
        if r["test_f1_label1"] is None
        else float(r["test_f1_label1"]),
        reverse=True,
    )

    def fmt(value):
        return "n/a" if value is None else f"{float(value):.4f}"

    print("Model Comparison (sorted by test F1 label=1)")
    print("-" * 90)
    for row in rows_sorted:
        print(
            f"{row['model']:<15} "
            f"F1={fmt(row['test_f1_label1'])} "
            f"P={fmt(row['test_precision_label1'])} "
            f"R={fmt(row['test_recall_label1'])} "
            f"ACC={fmt(row['test_accuracy'])} "
            f"thr={row['best_threshold']}"
        )


def save_csv(rows: list[dict], output_csv: Path):
    if not rows:
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    metric_files = load_metrics_files(MODELS_DIR)
    rows = []

    for file_path in metric_files:
        with file_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(extract_row(metrics))

    print_table(rows)
    save_csv(rows, OUTPUT_CSV)

    if rows:
        print(f"\nSaved comparison CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
