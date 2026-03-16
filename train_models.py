from pathlib import Path
import json

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
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def evaluate_with_threshold(model, X_val, y_val, X_test, y_test):
    val_probs = model.predict_proba(X_val)[:, 1]
    best_thr, val_best_f1 = tune_threshold(y_val, val_probs)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_pred = (test_probs >= best_thr).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_test, test_pred)
    cm = confusion_matrix(y_test, test_pred).tolist()

    return {
        "best_threshold": best_thr,
        "validation_best_f1": val_best_f1,
        "test": {
            "accuracy": acc,
            "precision_label1": p,
            "recall_label1": r,
            "f1_label1": f1,
            "confusion_matrix": cm,
        },
    }


def train_logreg(X_train, y_train):
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", max_iter=2000, random_state=42
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train, y_train):
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000),
            ),
            ("clf", MultinomialNB()),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_linear_svc(X_train, y_train):
    base_svc = LinearSVC(class_weight="balanced", random_state=42)
    calibrated = CalibratedClassifierCV(base_svc, method="sigmoid", cv=3)

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000),
            ),
            ("clf", calibrated),
        ]
    )
    model.fit(X_train, y_train)
    return model


def run_model(
    model_name,
    train_fn,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    models_dir: Path,
):
    model = train_fn(X_train, y_train)
    metrics = evaluate_with_threshold(model, X_val, y_val, X_test, y_test)
    metrics["model"] = model_name

    joblib.dump(model, models_dir / f"{model_name}.joblib")
    with (models_dir / f"{model_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"{model_name} done: F1={metrics['test']['f1_label1']:.4f}, "
        f"thr={metrics['best_threshold']}"
    )


def main():
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(base_dir)

    run_model(
        "logreg",
        train_logreg,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        models_dir,
    )
    run_model(
        "naive_bayes",
        train_naive_bayes,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        models_dir,
    )
    run_model(
        "linear_svc",
        train_linear_svc,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        models_dir,
    )


if __name__ == "__main__":
    main()
