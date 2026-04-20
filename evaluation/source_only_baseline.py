from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import ensure_features


def get_columns(mode: str) -> tuple[list[str], list[str]]:
    source_cat = ["source", "source_profile"]
    meta_cat = [
        "spf_result",
        "dkim_result",
        "dmarc_result",
        "from_domain",
        "reply_to_domain",
        "return_path_domain",
        "receiver_domain",
    ]
    meta_num = [
        "from_reply_mismatch",
        "from_return_path_mismatch",
        "sender_receiver_domain_mismatch",
        "received_hops_count",
        "url_count",
        "ip_url_count",
        "shortener_url_count",
        "unique_domain_count",
        "suspicious_tld_count",
    ]

    if mode == "source_only":
        return source_cat, []
    if mode == "metadata_only":
        return meta_cat, meta_num
    if mode == "source_plus_metadata":
        return source_cat + meta_cat, meta_num
    raise ValueError(f"Unknown mode: {mode}")


def evaluate_model(model: Pipeline, x_df: pd.DataFrame, y_true: pd.Series) -> dict:
    pred = model.predict(x_df)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, pred, average="weighted", zero_division=0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label baseline using only source/metadata (no email text)"
    )
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument("--scenario", default="scenario_g_source_balanced")
    parser.add_argument(
        "--mode",
        default="source_plus_metadata",
        choices=["source_only", "metadata_only", "source_plus_metadata"],
    )
    parser.add_argument("--output-json", default="results/source_only_baseline.json")
    parser.add_argument("--output-csv", default="results/source_only_baseline.csv")
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    train_path = processed / args.scenario / "train.csv"
    val_path = processed / args.scenario / "val.csv"
    test_path = processed / "shared" / "test.csv"
    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    train_df = ensure_features(pd.read_csv(train_path, low_memory=False))
    val_df = ensure_features(pd.read_csv(val_path, low_memory=False))
    test_df = ensure_features(pd.read_csv(test_path, low_memory=False))

    cat_cols, num_cols = get_columns(args.mode)
    for col in cat_cols:
        if col not in train_df.columns:
            train_df[col] = ""
            val_df[col] = ""
            test_df[col] = ""
        train_df[col] = train_df[col].fillna("").astype(str)
        val_df[col] = val_df[col].fillna("").astype(str)
        test_df[col] = test_df[col].fillna("").astype(str)

    for col in num_cols:
        if col not in train_df.columns:
            train_df[col] = 0.0
            val_df[col] = 0.0
            test_df[col] = 0.0
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0)
        val_df[col] = pd.to_numeric(val_df[col], errors="coerce").fillna(0.0)
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce").fillna(0.0)

    all_labels = sorted(
        set(train_df["label"].astype(str))
        .union(set(val_df["label"].astype(str)))
        .union(set(test_df["label"].astype(str)))
    )
    le = LabelEncoder()
    le.fit(all_labels)

    y_train = le.transform(train_df["label"].astype(str))
    y_val = le.transform(val_df["label"].astype(str))
    y_test = le.transform(test_df["label"].astype(str))

    pre = ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="")),
                        (
                            "ohe",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                cat_cols,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                num_cols,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        [
            ("prep", pre),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )

    x_train = train_df[cat_cols + num_cols]
    x_val = val_df[cat_cols + num_cols]
    x_test = test_df[cat_cols + num_cols]

    model.fit(x_train, y_train)
    val_metrics = evaluate_model(model, x_val, y_val)
    test_metrics = evaluate_model(model, x_test, y_test)

    out_row = {
        "scenario": args.scenario,
        "mode": args.mode,
        "cat_cols": ",".join(cat_cols),
        "num_cols": ",".join(num_cols),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([out_row]).to_csv(out_csv, index=False)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_row, indent=2), encoding="utf-8")

    print(
        f"mode={args.mode} test_acc={test_metrics['accuracy']:.4f} "
        f"test_f1_macro={test_metrics['f1_macro']:.4f}"
    )
    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
