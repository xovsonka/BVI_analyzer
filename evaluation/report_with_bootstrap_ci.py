from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import build_model, ensure_features


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = 400,
    alpha: float = 0.95,
    seed: int = 42,
) -> dict:
    if len(y_true) == 0:
        return {"value": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(float(metric_fn(y_true[idx], y_pred[idx])))

    vals = np.asarray(vals)
    lo_q = (1.0 - alpha) / 2.0
    hi_q = 1.0 - lo_q
    return {
        "value": float(metric_fn(y_true, y_pred)),
        "ci_low": float(np.quantile(vals, lo_q)),
        "ci_high": float(np.quantile(vals, hi_q)),
    }


def eval_split(
    model,
    le: LabelEncoder,
    df: pd.DataFrame,
    split_name: str,
    n_boot: int,
) -> dict:
    y_true = le.transform(df["label"].astype(str))
    y_pred = model.predict(df)

    m_macro = bootstrap_ci(
        y_true,
        y_pred,
        metric_fn=lambda a, b: f1_score(a, b, average="macro", zero_division=0),
        n_boot=n_boot,
    )
    m_bal = bootstrap_ci(
        y_true,
        y_pred,
        metric_fn=lambda a, b: balanced_accuracy_score(a, b),
        n_boot=n_boot,
    )

    return {
        "split": split_name,
        "rows": int(len(df)),
        "label_support": df["label"].value_counts().to_dict(),
        "f1_macro": m_macro,
        "balanced_accuracy": m_bal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model with bootstrap CI across split types"
    )
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument("--scenario", default="scenario_g_source_balanced")
    parser.add_argument(
        "--model",
        default="hybrid_linear_svc_cal",
        choices=[
            "hybrid_logreg",
            "hybrid_linear_svc_cal",
            "hybrid_sgd_log",
            "hybrid_sgd_hinge",
        ],
    )
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--output-json", default="results/bootstrap_ci_report.json")
    parser.add_argument("--output-csv", default="results/bootstrap_ci_report.csv")
    args = parser.parse_args()

    p = Path(args.processed_dir) / "shared"
    train_path = Path(args.processed_dir) / args.scenario / "train.csv"
    split_files = {
        "val_iid": p / "val_iid.csv",
        "test_iid": p / "test_iid.csv",
        "test_hard_source": p / "test_hard_source.csv",
        "test_hard_cluster": p / "test_hard_cluster.csv",
        "test_deployment": p / "test_deployment.csv",
    }
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")
    for name, fp in split_files.items():
        if not fp.exists():
            raise FileNotFoundError(f"Missing split file {name}: {fp}")

    train_df = ensure_features(pd.read_csv(train_path, low_memory=False))
    labels = sorted(set(train_df["label"].astype(str)))
    for fp in split_files.values():
        labels = sorted(
            set(labels).union(
                set(pd.read_csv(fp, usecols=["label"])["label"].astype(str))
            )
        )

    le = LabelEncoder()
    le.fit(labels)
    y_train = le.transform(train_df["label"].astype(str))

    model = build_model(args.model)
    model.fit(train_df, y_train)

    rows = []
    for name, fp in split_files.items():
        df = ensure_features(pd.read_csv(fp, low_memory=False))
        r = eval_split(model, le, df, name, n_boot=args.bootstrap)
        rows.append(r)

    flat = []
    for r in rows:
        flat.append(
            {
                "split": r["split"],
                "rows": r["rows"],
                "f1_macro": r["f1_macro"]["value"],
                "f1_macro_ci_low": r["f1_macro"]["ci_low"],
                "f1_macro_ci_high": r["f1_macro"]["ci_high"],
                "balanced_accuracy": r["balanced_accuracy"]["value"],
                "balanced_accuracy_ci_low": r["balanced_accuracy"]["ci_low"],
                "balanced_accuracy_ci_high": r["balanced_accuracy"]["ci_high"],
            }
        )

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat).to_csv(out_csv, index=False)

    payload = {
        "scenario": args.scenario,
        "model": args.model,
        "bootstrap": args.bootstrap,
        "results": rows,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
