from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import NUMERIC_FEATURES, ensure_features


CLASS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "phishing": ("verify", "mailbox", "login", "password", "account", "security"),
    "financial_fraud": ("invoice", "payment", "billing", "bank", "transfer", "due"),
    "spam": ("offer", "reward", "promo", "discount", "selected", "claim"),
}

DEFAULT_SPLITS = ["test_iid", "test_hard_source", "test_hard_cluster", "test_deployment"]


def _parse_float_grid(raw: str) -> list[float]:
    vals = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not vals:
        raise ValueError(f"Invalid grid values: {raw}")
    return vals


def _load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return ensure_features(pd.read_csv(path, low_memory=False))


def _resolve_shared_split(shared_dir: Path, split_name: str) -> Path | None:
    candidates = {
        "test_iid": [shared_dir / "test_iid.csv", shared_dir / "test.csv"],
        "test_hard_source": [shared_dir / "test_hard_source.csv"],
        "test_hard_cluster": [shared_dir / "test_hard_cluster.csv"],
        "test_deployment": [shared_dir / "test_deployment.csv"],
    }.get(split_name)
    if candidates is None:
        raise ValueError(f"Unsupported split name: {split_name}")
    for path in candidates:
        if path.exists():
            return path
    return None


def _mean_abs_shap_per_feature(shap_values: object, feature_count: int) -> np.ndarray:
    if isinstance(shap_values, list):
        per_class = [np.abs(v).mean(axis=0) for v in shap_values]
        return np.mean(np.vstack(per_class), axis=0)

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return np.mean(np.abs(arr), axis=(0, 2))
    if arr.ndim == 2 and arr.shape[1] == feature_count:
        return np.mean(np.abs(arr), axis=0)
    raise RuntimeError(f"Unsupported SHAP output shape: {arr.shape}")


def _build_model(model_name: str, numeric_features: list[str]) -> Pipeline:
    text_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=60000,
        sublinear_tf=True,
    )
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )
    prep = ColumnTransformer(
        [
            ("text", text_vectorizer, "text_input"),
            ("num", num_pipe, numeric_features),
        ],
        remainder="drop",
    )

    if model_name == "hybrid_logreg":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
    elif model_name == "hybrid_sgd_log":
        clf = SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            alpha=1e-5,
            max_iter=3000,
            tol=1e-3,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([("prep", prep), ("clf", clf)])


def _select_input(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    return df[["text_input", *numeric_features]].copy()


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _count_keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    low = str(text or "").lower()
    return sum(1 for kw in keywords if kw in low)


def _add_hard_negatives(train_df: pd.DataFrame, multiplier: float, seed: int) -> pd.DataFrame:
    if multiplier <= 1.0:
        return train_df.copy()

    hard_indices: list[int] = []
    for idx, row in train_df.iterrows():
        label = str(row.get("label", "")).strip().lower()
        if label == "legit":
            continue
        text = str(row.get("text_input", ""))
        overlap_count = 0
        for other_label, keywords in CLASS_KEYWORDS.items():
            if other_label == label:
                continue
            overlap_count += int(_count_keyword_hits(text, keywords) > 0)
        if overlap_count > 0:
            hard_indices.append(idx)

    if not hard_indices:
        return train_df.copy()

    hard_df = train_df.loc[hard_indices].copy()
    extra_n = int(len(hard_df) * (multiplier - 1.0))
    if extra_n <= 0:
        return train_df.copy()

    extra = hard_df.sample(n=extra_n, replace=True, random_state=seed).copy()
    extra["hardneg_aug"] = 1
    out = pd.concat([train_df.copy(), extra], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _class_sample_weights(
    labels: pd.Series,
    spam_weight: float,
    phishing_weight: float,
    fraud_weight: float,
) -> np.ndarray:
    label_str = labels.astype(str)
    out = np.ones(len(label_str), dtype=float)
    out[label_str.eq("spam").to_numpy()] *= float(spam_weight)
    out[label_str.eq("phishing").to_numpy()] *= float(phishing_weight)
    out[label_str.eq("financial_fraud").to_numpy()] *= float(fraud_weight)
    return out


def _tune_class_scales(
    y_true_idx: np.ndarray,
    proba: np.ndarray,
    class_labels: list[str],
) -> dict[str, float]:
    scales = {label: 1.0 for label in class_labels}
    candidates = [0.75, 0.9, 1.0, 1.1, 1.25]

    def _score(current: dict[str, float]) -> float:
        vec = np.array([current[label] for label in class_labels], dtype=float)
        pred = (proba / vec).argmax(axis=1)
        return float(f1_score(y_true_idx, pred, average="macro", zero_division=0))

    best = _score(scales)
    improved = True
    while improved:
        improved = False
        for label in class_labels:
            local_best = scales[label]
            local_score = best
            for cand in candidates:
                trial = dict(scales)
                trial[label] = cand
                trial_score = _score(trial)
                if trial_score > local_score:
                    local_best = cand
                    local_score = trial_score
            if local_best != scales[label]:
                scales[label] = local_best
                best = local_score
                improved = True
    return scales


def _predict_with_scales(
    model: Pipeline,
    x_df: pd.DataFrame,
    class_labels: list[str],
    scales: dict[str, float],
) -> np.ndarray:
    proba = model.predict_proba(x_df)
    vec = np.array([scales[label] for label in class_labels], dtype=float)
    return (proba / vec).argmax(axis=1)


def _run_shap_audit(
    train_df: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    numeric_features: list[str],
    max_rows_per_split: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["label"].astype(str))
    x_train = train_df[numeric_features].copy()

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(label_encoder.classes_),
        n_estimators=320,
        learning_rate=0.05,
        num_leaves=63,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)

    shap_tables: dict[str, pd.DataFrame] = {}
    for split_name, split_df in split_frames.items():
        source = split_df
        if len(source) > max_rows_per_split:
            source = source.sample(n=max_rows_per_split, random_state=42)
        x_eval = source[numeric_features].copy()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_eval)
        importance = _mean_abs_shap_per_feature(shap_values, feature_count=len(numeric_features))
        shap_df = pd.DataFrame(
            {
                "feature": numeric_features,
                "mean_abs_shap": importance,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        shap_tables[split_name] = shap_df

    ratio_rows = []
    total = max(1, len(train_df))
    for feature in numeric_features:
        values = pd.to_numeric(train_df[feature], errors="coerce").fillna(0.0)
        ratio_rows.append(
            {
                "feature": feature,
                "train_non_zero_ratio": float((values != 0).sum() / total),
                "train_mean": float(values.mean()),
                "train_std": float(values.std(ddof=0)),
                "train_unique_values": int(values.nunique(dropna=True)),
            }
        )
    stats_df = pd.DataFrame(ratio_rows)
    for split_name, shap_df in shap_tables.items():
        stats_df = stats_df.merge(
            shap_df.rename(columns={"mean_abs_shap": f"mean_abs_shap_{split_name}"}),
            on="feature",
            how="left",
        )

    shap_cols = [col for col in stats_df.columns if col.startswith("mean_abs_shap_")]
    stats_df["mean_abs_shap_overall"] = stats_df[shap_cols].mean(axis=1)
    return shap_tables, stats_df


def _decide_feature_subset(
    stats_df: pd.DataFrame,
    min_nonzero_ratio: float,
    min_mean_abs_shap: float,
) -> tuple[list[str], pd.DataFrame]:
    decisions = stats_df.copy()
    decisions["drop_reason"] = ""

    low_signal = (
        (decisions["train_non_zero_ratio"] < min_nonzero_ratio)
        & (decisions["mean_abs_shap_overall"] < min_mean_abs_shap)
    )
    zero_variance = decisions["train_unique_values"] <= 1

    decisions.loc[low_signal, "drop_reason"] = "low_density_low_shap"
    decisions.loc[zero_variance, "drop_reason"] = "zero_variance"
    decisions["keep"] = decisions["drop_reason"].eq("")

    keep_features = decisions.loc[decisions["keep"], "feature"].tolist()
    if not keep_features:
        keep_features = decisions["feature"].tolist()
        decisions["keep"] = True
        decisions["drop_reason"] = "fallback_keep_all"
    return keep_features, decisions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1-5 optimization for scenario M using SHAP + threshold/class tuning"
    )
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument("--scenario", default="scenario_m_feature_probe_rawfix")
    parser.add_argument(
        "--model",
        default="hybrid_logreg",
        choices=["hybrid_logreg", "hybrid_sgd_log"],
    )
    parser.add_argument(
        "--splits",
        default="test_iid,test_hard_source,test_hard_cluster,test_deployment",
        help="Comma-separated shared eval splits.",
    )
    parser.add_argument("--max-shap-rows", type=int, default=6000)
    parser.add_argument("--min-nonzero-ratio", type=float, default=0.0002)
    parser.add_argument("--min-mean-abs-shap", type=float, default=0.001)
    parser.add_argument("--search-train-rows", type=int, default=40000)
    parser.add_argument("--spam-weight-grid", default="1.0,1.2")
    parser.add_argument("--phishing-weight-grid", default="1.0")
    parser.add_argument("--fraud-weight-grid", default="1.0,1.3")
    parser.add_argument("--hardneg-grid", default="1.0,1.25")
    parser.add_argument("--out-dir", default="results/m_shap_optimization")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    scenario_dir = processed_dir / args.scenario
    shared_dir = processed_dir / "shared"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load_split(scenario_dir / "train.csv")
    val_df = _load_split(scenario_dir / "val.csv")

    split_names = [part.strip() for part in args.splits.split(",") if part.strip()]
    eval_splits: dict[str, pd.DataFrame] = {}
    for split_name in split_names:
        split_path = _resolve_shared_split(shared_dir, split_name)
        if split_path is None:
            print(f"[WARN] Missing split skipped: {split_name}")
            continue
        eval_splits[split_name] = _load_split(split_path)

    numeric_features = [feature for feature in NUMERIC_FEATURES if feature in train_df.columns]
    if not numeric_features:
        raise RuntimeError("No numeric features available in train split")

    shap_tables, feature_stats = _run_shap_audit(
        train_df=train_df,
        split_frames={"val_iid": val_df, **eval_splits},
        numeric_features=numeric_features,
        max_rows_per_split=args.max_shap_rows,
    )
    for split_name, table in shap_tables.items():
        table.to_csv(out_dir / f"shap_feature_importance_{split_name}.csv", index=False)

    keep_features, feature_decisions = _decide_feature_subset(
        stats_df=feature_stats,
        min_nonzero_ratio=float(args.min_nonzero_ratio),
        min_mean_abs_shap=float(args.min_mean_abs_shap),
    )
    feature_decisions.to_csv(out_dir / "feature_audit.csv", index=False)

    labels = sorted(
        set(train_df["label"].astype(str)).union(set(val_df["label"].astype(str)))
    )
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    class_labels = [str(label) for label in label_encoder.classes_]

    baseline_model = _build_model(args.model, numeric_features=numeric_features)
    baseline_x_train = _select_input(train_df, numeric_features)
    baseline_y_train = label_encoder.transform(train_df["label"].astype(str))
    baseline_model.fit(baseline_x_train, baseline_y_train)

    baseline_rows = []
    for split_name, split_df in {"val_iid": val_df, **eval_splits}.items():
        x_eval = _select_input(split_df, numeric_features)
        y_eval = label_encoder.transform(split_df["label"].astype(str))
        pred_eval = baseline_model.predict(x_eval)
        baseline_rows.append(
            {
                "variant": "baseline",
                "split": split_name,
                "feature_count": len(numeric_features),
                **_compute_metrics(y_eval, pred_eval),
                "rows": int(len(split_df)),
            }
        )

    search_train = train_df
    if args.search_train_rows > 0 and len(train_df) > args.search_train_rows:
        search_train = train_df.sample(n=args.search_train_rows, random_state=42).reset_index(drop=True)

    spam_grid = _parse_float_grid(args.spam_weight_grid)
    phishing_grid = _parse_float_grid(args.phishing_weight_grid)
    fraud_grid = _parse_float_grid(args.fraud_weight_grid)
    hardneg_grid = _parse_float_grid(args.hardneg_grid)

    search_rows = []
    best_cfg: dict | None = None
    best_val_f1 = -1.0
    for spam_weight, phishing_weight, fraud_weight, hardneg_mult in itertools.product(
        spam_grid,
        phishing_grid,
        fraud_grid,
        hardneg_grid,
    ):
        tuned_train = _add_hard_negatives(search_train, multiplier=hardneg_mult, seed=42)
        x_train = _select_input(tuned_train, keep_features)
        y_train = label_encoder.transform(tuned_train["label"].astype(str))

        model = _build_model(args.model, numeric_features=keep_features)
        sample_weight = _class_sample_weights(
            labels=tuned_train["label"],
            spam_weight=spam_weight,
            phishing_weight=phishing_weight,
            fraud_weight=fraud_weight,
        )
        model.fit(x_train, y_train, clf__sample_weight=sample_weight)

        x_val = _select_input(val_df, keep_features)
        y_val = label_encoder.transform(val_df["label"].astype(str))
        proba_val = model.predict_proba(x_val)
        scales = _tune_class_scales(y_true_idx=y_val, proba=proba_val, class_labels=class_labels)
        pred_val = _predict_with_scales(
            model=model,
            x_df=x_val,
            class_labels=class_labels,
            scales=scales,
        )
        val_metrics = _compute_metrics(y_val, pred_val)
        row = {
            "spam_weight": spam_weight,
            "phishing_weight": phishing_weight,
            "fraud_weight": fraud_weight,
            "hardneg_multiplier": hardneg_mult,
            "feature_count": len(keep_features),
            **val_metrics,
            "scales": json.dumps(scales, sort_keys=True),
        }
        search_rows.append(row)

        if float(val_metrics["f1_macro"]) > best_val_f1:
            best_val_f1 = float(val_metrics["f1_macro"])
            best_cfg = {
                "spam_weight": spam_weight,
                "phishing_weight": phishing_weight,
                "fraud_weight": fraud_weight,
                "hardneg_multiplier": hardneg_mult,
                "scales": scales,
            }

    if best_cfg is None:
        raise RuntimeError("No tuning config evaluated")

    search_df = pd.DataFrame(search_rows).sort_values("f1_macro", ascending=False)
    search_df.to_csv(out_dir / "tuning_search_results.csv", index=False)

    tuned_train_full = _add_hard_negatives(
        train_df,
        multiplier=float(best_cfg["hardneg_multiplier"]),
        seed=42,
    )
    tuned_x_train = _select_input(tuned_train_full, keep_features)
    tuned_y_train = label_encoder.transform(tuned_train_full["label"].astype(str))
    tuned_model = _build_model(args.model, numeric_features=keep_features)
    tuned_weight = _class_sample_weights(
        labels=tuned_train_full["label"],
        spam_weight=float(best_cfg["spam_weight"]),
        phishing_weight=float(best_cfg["phishing_weight"]),
        fraud_weight=float(best_cfg["fraud_weight"]),
    )
    tuned_model.fit(tuned_x_train, tuned_y_train, clf__sample_weight=tuned_weight)

    tuned_val_x = _select_input(val_df, keep_features)
    tuned_val_y = label_encoder.transform(val_df["label"].astype(str))
    tuned_val_proba = tuned_model.predict_proba(tuned_val_x)
    tuned_scales = _tune_class_scales(
        y_true_idx=tuned_val_y,
        proba=tuned_val_proba,
        class_labels=class_labels,
    )

    tuned_rows = []
    for split_name, split_df in {"val_iid": val_df, **eval_splits}.items():
        x_eval = _select_input(split_df, keep_features)
        y_eval = label_encoder.transform(split_df["label"].astype(str))
        pred_eval = _predict_with_scales(
            model=tuned_model,
            x_df=x_eval,
            class_labels=class_labels,
            scales=tuned_scales,
        )
        tuned_rows.append(
            {
                "variant": "tuned",
                "split": split_name,
                "feature_count": len(keep_features),
                **_compute_metrics(y_eval, pred_eval),
                "rows": int(len(split_df)),
            }
        )

    metrics_df = pd.DataFrame(baseline_rows + tuned_rows)
    metrics_df.to_csv(out_dir / "metrics_baseline_vs_tuned.csv", index=False)

    delta_rows = []
    for split_name in metrics_df["split"].unique().tolist():
        base_row = metrics_df[(metrics_df["variant"] == "baseline") & (metrics_df["split"] == split_name)]
        tuned_row = metrics_df[(metrics_df["variant"] == "tuned") & (metrics_df["split"] == split_name)]
        if base_row.empty or tuned_row.empty:
            continue
        b = base_row.iloc[0]
        t = tuned_row.iloc[0]
        delta_rows.append(
            {
                "split": split_name,
                "delta_f1_macro": float(t["f1_macro"] - b["f1_macro"]),
                "delta_balanced_accuracy": float(t["balanced_accuracy"] - b["balanced_accuracy"]),
                "delta_accuracy": float(t["accuracy"] - b["accuracy"]),
            }
        )
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "metrics_delta_tuned_vs_baseline.csv", index=False)

    summary = {
        "scenario": args.scenario,
        "model": args.model,
        "feature_count_baseline": int(len(numeric_features)),
        "feature_count_tuned": int(len(keep_features)),
        "best_config_from_search": {
            "spam_weight": float(best_cfg["spam_weight"]),
            "phishing_weight": float(best_cfg["phishing_weight"]),
            "fraud_weight": float(best_cfg["fraud_weight"]),
            "hardneg_multiplier": float(best_cfg["hardneg_multiplier"]),
        },
        "tuned_class_scales": tuned_scales,
        "metrics_delta_tuned_vs_baseline": delta_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved optimization outputs to: {out_dir}")
    for _, row in delta_df.iterrows():
        print(
            f"{row['split']}: delta_f1_macro={row['delta_f1_macro']:.4f}, "
            f"delta_bal_acc={row['delta_balanced_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
