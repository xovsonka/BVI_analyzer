from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import build_model, ensure_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot multiclass ROC/PR curves for one scenario+model"
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario folder name, e.g. scenario_b_full_weighted",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "hybrid_logreg",
            "hybrid_linear_svc_cal",
            "hybrid_sgd_log",
            "hybrid_sgd_hinge",
        ],
        help="Model to train and evaluate",
    )
    parser.add_argument(
        "--processed-dir",
        default="dataset/processed",
        help="Processed dataset directory",
    )
    parser.add_argument("--fit-on", choices=["train", "train_val"], default="train_val")
    parser.add_argument(
        "--output-dir", default="results/curves", help="Where to save curve files"
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    processed_dir = Path(args.processed_dir)
    scenario_dir = processed_dir / args.scenario
    train_path = scenario_dir / "train.csv"
    val_path = scenario_dir / "val.csv"
    test_path = processed_dir / "shared" / "test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    train_df = ensure_features(pd.read_csv(train_path, low_memory=False))
    val_df = ensure_features(pd.read_csv(val_path, low_memory=False))
    test_df = ensure_features(pd.read_csv(test_path, low_memory=False))

    if args.fit_on == "train_val":
        fit_df = pd.concat([train_df, val_df], ignore_index=True)
    else:
        fit_df = train_df

    labels = sorted(set(test_df["label"].astype(str).tolist()))
    le = LabelEncoder()
    le.fit(labels)
    class_names = list(le.classes_)

    x_fit = fit_df
    y_fit = le.transform(fit_df["label"].astype(str))
    x_test = test_df
    y_test = le.transform(test_df["label"].astype(str))

    model = build_model(args.model)
    model.fit(x_fit, y_fit)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)
    else:
        raise RuntimeError(
            "Selected model does not expose predict_proba, cannot plot ROC/PR curves"
        )

    y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))

    roc_auc_per_class = {}
    ap_per_class = {}

    output_dir = Path(args.output_dir) / f"{args.scenario}__{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ROC plot
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        auc_val = roc_auc_score(y_test_bin[:, i], y_score[:, i])
        roc_auc_per_class[class_name] = float(auc_val)
        plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC={auc_val:.3f})")

    micro_roc_auc = roc_auc_score(
        y_test_bin, y_score, average="micro", multi_class="ovr"
    )
    macro_roc_auc = roc_auc_score(
        y_test_bin, y_score, average="macro", multi_class="ovr"
    )
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multiclass ROC (scenario={args.scenario}, model={args.model})")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    roc_png = output_dir / "roc_multiclass.png"
    plt.tight_layout()
    plt.savefig(roc_png, dpi=160)
    plt.close()

    # PR plot
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        ap_val = average_precision_score(y_test_bin[:, i], y_score[:, i])
        ap_per_class[class_name] = float(ap_val)
        plt.plot(recall, precision, lw=2, label=f"{class_name} (AP={ap_val:.3f})")

    micro_ap = average_precision_score(y_test_bin, y_score, average="micro")
    macro_ap = average_precision_score(y_test_bin, y_score, average="macro")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Multiclass PR (scenario={args.scenario}, model={args.model})")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    pr_png = output_dir / "pr_multiclass.png"
    plt.tight_layout()
    plt.savefig(pr_png, dpi=160)
    plt.close()

    # Save metrics
    metrics = {
        "scenario": args.scenario,
        "model": args.model,
        "fit_on": args.fit_on,
        "classes": class_names,
        "roc_auc_micro": float(micro_roc_auc),
        "roc_auc_macro": float(macro_roc_auc),
        "average_precision_micro": float(micro_ap),
        "average_precision_macro": float(macro_ap),
        "roc_auc_per_class": roc_auc_per_class,
        "average_precision_per_class": ap_per_class,
        "roc_png": str(roc_png),
        "pr_png": str(pr_png),
    }
    metrics_path = output_dir / "curve_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved ROC plot: {roc_png}")
    print(f"Saved PR plot: {pr_png}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
