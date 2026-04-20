from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import ensure_features


def build_text_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )


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
            if c in row and pd.notna(row[c]):
                tokens.append(f"{c}:{str(row[c]).strip()}")
        out.append(" ".join(tokens).strip())
    return pd.Series(out, index=df.index)


def view_text(df: pd.DataFrame, view: str) -> pd.Series:
    if view == "body_only":
        return df["body"].astype(str)
    if view == "header_plus_body":
        hdr = build_header_text(df)
        return (hdr + " " + df["body"].astype(str)).str.strip()
    if view == "full_text":
        return df["text_input"].astype(str)
    raise ValueError(f"Unknown view: {view}")


def load_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
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
            out[split] = ensure_features(pd.read_csv(fp, low_memory=False))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare text views: body vs header+body"
    )
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument("--scenario", default="scenario_j_hard_source_selected")
    parser.add_argument(
        "--views",
        nargs="+",
        default=["body_only", "header_plus_body", "full_text"],
        choices=["body_only", "header_plus_body", "full_text"],
    )
    parser.add_argument("--output-csv", default="results/text_view_modes.csv")
    parser.add_argument("--output-json", default="results/text_view_modes.json")
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    train_path = processed / args.scenario / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")

    train_df = ensure_features(pd.read_csv(train_path, low_memory=False))
    splits = load_splits(processed)
    if not splits:
        raise RuntimeError("No shared eval splits found")

    labels = sorted(set(train_df["label"].astype(str)))
    for d in splits.values():
        labels = sorted(set(labels).union(set(d["label"].astype(str))))
    le = LabelEncoder()
    le.fit(labels)
    y_train = le.transform(train_df["label"].astype(str))

    rows = []
    for view in args.views:
        model = build_text_model()
        x_train = view_text(train_df, view)
        model.fit(x_train, y_train)

        for split_name, split_df in splits.items():
            x_test = view_text(split_df, view)
            y_test = le.transform(split_df["label"].astype(str))
            pred = model.predict(x_test)
            row = {
                "scenario": args.scenario,
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
            rows.append(row)
            print(f"{view:<16} {split_name:<16} f1_macro={row['f1_macro']:.4f}")

    out_df = pd.DataFrame(rows)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
