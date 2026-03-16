from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def load_splits(base_dir: Path):
    data_dir = base_dir / "dataset" / "processed"
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    return (
        train_df["text"].astype(str),
        train_df["label"].astype(int),
        val_df["text"].astype(str),
        val_df["label"].astype(int),
        test_df["text"].astype(str),
        test_df["label"].astype(int),
    )


def tune_threshold(y_true, probs):
    best_thr, best_f1 = 0.5, -1.0
    for thr in [x / 100 for x in range(10, 91)]:
        pred = (probs >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


def evaluate_thresholded(model, x_val, y_val, x_test, y_test):
    val_probs = model.predict_proba(x_val)[:, 1]
    threshold, val_f1 = tune_threshold(y_val, val_probs)

    val_pred = (val_probs >= threshold).astype(int)
    val_p, val_r, _, _ = precision_recall_fscore_support(
        y_val, val_pred, average="binary", zero_division=0
    )

    test_probs = model.predict_proba(x_test)[:, 1]
    test_pred = (test_probs >= threshold).astype(int)
    test_p, test_r, test_f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average="binary", zero_division=0
    )

    return {
        "threshold": threshold,
        "val_precision_label1": float(val_p),
        "val_recall_label1": float(val_r),
        "val_f1_label1": float(val_f1),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_precision_label1": float(test_p),
        "test_recall_label1": float(test_r),
        "test_f1_label1": float(test_f1),
        "test_confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }


def build_pipeline(model_name: str, tfidf_params: dict, model_params: dict):
    tfidf = TfidfVectorizer(**tfidf_params)

    if model_name == "logreg":
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
            **model_params,
        )
    elif model_name == "naive_bayes":
        clf = MultinomialNB(**model_params)
    elif model_name == "linear_svc":
        base = LinearSVC(class_weight="balanced", random_state=42, **model_params)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def search_model(
    model_name: str, x_train, y_train, x_val, y_val, x_test, y_test, limit: int | None
):
    tfidf_grid = {
        "ngram_range": [(1, 1), (1, 2)],
        "min_df": [2, 5],
        "max_features": [30000, 50000],
    }

    if model_name == "logreg":
        model_grid = {"C": [0.5, 1.0, 2.0]}
    elif model_name == "naive_bayes":
        model_grid = {"alpha": [0.1, 0.5, 1.0]}
    elif model_name == "linear_svc":
        model_grid = {"C": [0.5, 1.0, 2.0]}
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    tfidf_candidates = list(ParameterGrid(tfidf_grid))
    model_candidates = list(ParameterGrid(model_grid))

    results = []
    trial = 0

    for tfidf_params in tfidf_candidates:
        for model_params in model_candidates:
            trial += 1
            if limit is not None and trial > limit:
                break

            model = build_pipeline(model_name, tfidf_params, model_params)
            model.fit(x_train, y_train)

            metrics = evaluate_thresholded(model, x_val, y_val, x_test, y_test)
            result = {
                "model": model_name,
                "trial": trial,
                "tfidf": tfidf_params,
                "params": model_params,
                **metrics,
                "pipeline": model,
            }
            results.append(result)
            print(
                f"[{model_name}] trial={trial} val_f1={metrics['val_f1_label1']:.4f} "
                f"test_f1={metrics['test_f1_label1']:.4f}"
            )

        if limit is not None and trial >= limit:
            break

    if not results:
        raise RuntimeError(f"No tuning results for model {model_name}")

    best = max(results, key=lambda row: row["val_f1_label1"])
    return best, results


def strip_for_json(row: dict):
    return {k: v for k, v in row.items() if k != "pipeline"}


def main():
    parser = argparse.ArgumentParser(
        description="Tune text models for phishing detection"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "naive_bayes", "linear_svc"],
        choices=["logreg", "naive_bayes", "linear_svc"],
    )
    parser.add_argument(
        "--limit-per-model",
        type=int,
        default=None,
        help="Optional limit of hyperparameter trials per model",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "models" / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val, x_test, y_test = load_splits(base_dir)

    summary = []

    for model_name in args.models:
        best, all_results = search_model(
            model_name,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            args.limit_per_model,
        )

        best_model_path = out_dir / f"best_{model_name}.joblib"
        joblib.dump(best["pipeline"], best_model_path)

        best_json_path = out_dir / f"best_{model_name}.json"
        with best_json_path.open("w", encoding="utf-8") as handle:
            json.dump(strip_for_json(best), handle, indent=2)

        trials_json_path = out_dir / f"trials_{model_name}.json"
        with trials_json_path.open("w", encoding="utf-8") as handle:
            json.dump([strip_for_json(row) for row in all_results], handle, indent=2)

        summary.append(strip_for_json(best))

        print(
            f"Best {model_name}: val_f1={best['val_f1_label1']:.4f}, "
            f"test_f1={best['test_f1_label1']:.4f}, threshold={best['threshold']}"
        )

    summary_path = out_dir / "best_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nSaved tuning artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
