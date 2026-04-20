from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import NUMERIC_FEATURES, ensure_features
from prepare_dataset_multiclass import add_engineered_features


CLASS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "phishing": ("verify", "mailbox", "login", "password", "account", "security"),
    "financial_fraud": ("invoice", "payment", "billing", "bank", "transfer", "due"),
    "spam": ("offer", "reward", "promo", "discount", "selected", "claim"),
}

PROMO_SPAM_KEYWORDS = (
    "offer",
    "reward",
    "promo",
    "discount",
    "bonus",
    "member",
    "unsubscribe",
    "deal",
    "campaign",
)

SUSPICIOUS_LABELS = ["financial_fraud", "phishing", "spam"]


def _parse_class_scales(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    value = (raw or "").strip()
    if not value:
        return out
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid class scale entry: {item}")
        key, val = item.split(":", 1)
        out[key.strip()] = float(val.strip())
    return out


def _normalize_attack_type(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "nan"}:
        return None
    aliases = {
        "legitimate": "legit",
        "benign": "legit",
        "fraud": "financial_fraud",
    }
    return aliases.get(text, text)


def _apply_spoofing_policy(value: str | None, policy: str) -> str | None:
    if value is None:
        return None
    label = str(value).strip().lower()
    if label != "spoofing":
        return label
    if policy == "map_to_phishing":
        return "phishing"
    if policy == "drop":
        return None
    return label


def _build_seed_attack_map(seed_csv: Path, spoofing_policy: str) -> dict[int, str]:
    if not seed_csv.exists():
        return {}
    seed = pd.read_csv(seed_csv, low_memory=False)
    if "id" not in seed.columns or "attack_type" not in seed.columns:
        return {}
    seed = seed.copy()
    seed["id"] = pd.to_numeric(seed["id"], errors="coerce")
    seed["attack_type"] = seed["attack_type"].map(_normalize_attack_type)
    seed["attack_type"] = seed["attack_type"].map(
        lambda x: _apply_spoofing_policy(x, spoofing_policy)
    )
    seed = seed.dropna(subset=["id", "attack_type"])
    out: dict[int, str] = {}
    for _, row in seed.iterrows():
        out[int(row["id"])] = str(row["attack_type"])
    return out


def _build_file_to_dataset_id_map(metadata_csv: Path, id_re: re.Pattern[str]) -> dict[str, int]:
    if not metadata_csv.exists():
        return {}
    meta = pd.read_csv(metadata_csv, low_memory=False)
    if "eml_path" not in meta.columns or "to" not in meta.columns:
        return {}
    out: dict[str, int] = {}
    for _, row in meta.iterrows():
        file_key = Path(str(row.get("eml_path", ""))).name
        m = id_re.search(str(row.get("to", "")))
        if not file_key or m is None:
            continue
        out[file_key] = int(m.group(1))
    return out


def _parse_dataset_id_from_receiver(receiver: object, id_re: re.Pattern[str]) -> int | None:
    m = id_re.search(str(receiver or ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _resolve_dataset_id(
    analyzed: pd.DataFrame,
    file_to_dataset_id: dict[str, int],
    id_re: re.Pattern[str],
) -> pd.Series:
    out = pd.Series([pd.NA] * len(analyzed), index=analyzed.index, dtype="Int64")
    for col in ["dataset_id", "dataset_id_hint"]:
        if col in analyzed.columns:
            candidate = pd.to_numeric(analyzed[col], errors="coerce").astype("Int64")
            out = out.where(out.notna(), candidate)

    if "receiver" in analyzed.columns:
        receiver_id = analyzed["receiver"].map(
            lambda value: _parse_dataset_id_from_receiver(value, id_re)
        )
        receiver_id = pd.to_numeric(receiver_id, errors="coerce").astype("Int64")
        out = out.where(out.notna(), receiver_id)

    if "file" in analyzed.columns and file_to_dataset_id:
        file_id = analyzed["file"].map(lambda p: file_to_dataset_id.get(Path(str(p)).name))
        file_id = pd.to_numeric(file_id, errors="coerce").astype("Int64")
        out = out.where(out.notna(), file_id)
    return out


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


def _class_sample_weight(
    labels: pd.Series,
    spam_weight: float,
    phishing_weight: float,
    fraud_weight: float,
) -> np.ndarray:
    y = labels.astype(str)
    out = np.ones(len(y), dtype=float)
    out[y.eq("spam").to_numpy()] *= float(spam_weight)
    out[y.eq("phishing").to_numpy()] *= float(phishing_weight)
    out[y.eq("financial_fraud").to_numpy()] *= float(fraud_weight)
    return out


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
        overlap = 0
        for other_label, keywords in CLASS_KEYWORDS.items():
            if other_label == label:
                continue
            overlap += int(_count_keyword_hits(text, keywords) > 0)
        if overlap > 0:
            hard_indices.append(idx)

    if not hard_indices:
        return train_df.copy()
    hard_df = train_df.loc[hard_indices].copy()
    extra_n = int(len(hard_df) * (multiplier - 1.0))
    if extra_n <= 0:
        return train_df.copy()
    extra = hard_df.sample(n=extra_n, replace=True, random_state=seed)
    out = pd.concat([train_df.copy(), extra], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _ensure_label_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "label" not in out.columns and "attack_type" in out.columns:
        out["label"] = out["attack_type"]
    if "label" not in out.columns:
        raise ValueError("Dataset is missing required 'label' or 'attack_type' column")
    out["label"] = out["label"].astype(str).str.strip().str.lower().replace({"spoofing": "phishing"})
    return out


def _combine_text_for_spam_filter(df: pd.DataFrame) -> pd.Series:
    pieces = []
    for col in ["subject", "body", "text", "subject_raw", "body_raw", "text_raw"]:
        if col in df.columns:
            pieces.append(df[col].fillna("").astype(str))
    if not pieces:
        return pd.Series([""] * len(df), index=df.index, dtype=str)
    combined = pieces[0]
    for piece in pieces[1:]:
        combined = combined + "\n" + piece
    return combined.str.lower()


def _narrow_spam_train_rows(train_raw: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict[str, object]]:
    if mode == "none":
        return train_raw.copy(), {"enabled": False, "mode": mode}

    out = _ensure_label_column(train_raw)
    spam_mask = out["label"].eq("spam")
    spam_total = int(spam_mask.sum())
    if spam_total == 0:
        return out, {"enabled": True, "mode": mode, "spam_rows_before": 0, "spam_rows_after": 0}

    text = _combine_text_for_spam_filter(out)
    promo_mask = pd.Series(False, index=out.index)
    for kw in PROMO_SPAM_KEYWORDS:
        promo_mask = promo_mask | text.str.contains(rf"\b{re.escape(kw)}\b", na=False)

    phishing_like = pd.Series(False, index=out.index)
    for kw in CLASS_KEYWORDS["phishing"]:
        phishing_like = phishing_like | text.str.contains(rf"\b{re.escape(kw)}\b", na=False)

    fraud_like = pd.Series(False, index=out.index)
    for kw in CLASS_KEYWORDS["financial_fraud"]:
        fraud_like = fraud_like | text.str.contains(rf"\b{re.escape(kw)}\b", na=False)

    keep_spam = promo_mask & ~phishing_like & ~fraud_like
    if int((spam_mask & keep_spam).sum()) == 0:
        keep_spam = promo_mask
    if int((spam_mask & keep_spam).sum()) == 0:
        return out, {
            "enabled": True,
            "mode": mode,
            "spam_rows_before": spam_total,
            "spam_rows_after": spam_total,
            "fallback": "no_spam_rows_removed",
        }

    keep_mask = (~spam_mask) | keep_spam
    filtered = out.loc[keep_mask].copy().reset_index(drop=True)
    return filtered, {
        "enabled": True,
        "mode": mode,
        "spam_rows_before": spam_total,
        "spam_rows_after": int((filtered["label"] == "spam").sum()),
    }


def _load_adaptation_rows(adaptation_csv: Path) -> pd.DataFrame:
    adapt = pd.read_csv(adaptation_csv, low_memory=False)
    adapt = _ensure_label_column(adapt)
    missing_numeric = [col for col in NUMERIC_FEATURES if col not in adapt.columns]
    if missing_numeric:
        adapt = add_engineered_features(adapt)
    return adapt


def _mix_adaptation_rows(
    train_raw: pd.DataFrame,
    adaptation_csv: str,
    adaptation_weight: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    base = _ensure_label_column(train_raw).copy()
    base["_row_weight_multiplier"] = 1.0
    base["_train_origin"] = "base"

    if not adaptation_csv:
        return base, {"enabled": False, "rows_added": 0, "adaptation_weight": float(adaptation_weight)}

    adapt_path = Path(adaptation_csv)
    if not adapt_path.exists():
        raise FileNotFoundError(f"Adaptation CSV not found: {adapt_path}")

    adapt = _load_adaptation_rows(adapt_path)
    adapt["_row_weight_multiplier"] = float(adaptation_weight)
    adapt["_train_origin"] = "adaptation"

    mixed = pd.concat([base, adapt], ignore_index=True, sort=False)
    return mixed, {
        "enabled": True,
        "rows_added": int(len(adapt)),
        "adaptation_weight": float(adaptation_weight),
        "adaptation_path": str(adapt_path),
    }


def _build_sample_weight(train_df: pd.DataFrame, spam_weight: float, phishing_weight: float, fraud_weight: float) -> np.ndarray:
    base = _class_sample_weight(
        labels=train_df["label"],
        spam_weight=float(spam_weight),
        phishing_weight=float(phishing_weight),
        fraud_weight=float(fraud_weight),
    )
    row_mult = pd.to_numeric(train_df.get("_row_weight_multiplier", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    return base * row_mult


def _predict_with_multiclass_head(
    model: Pipeline,
    x_eval: pd.DataFrame,
    class_labels: list[str],
    class_scales: dict[str, float],
    scale_mode: str,
) -> np.ndarray:
    if class_scales and hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_eval)
        scale_vec = np.array([class_scales.get(name, 1.0) for name in class_labels], dtype=float)
        if scale_mode == "divide":
            return (proba / scale_vec).argmax(axis=1)
        return (proba * scale_vec).argmax(axis=1)
    return model.predict(x_eval)


def _positive_class_proba(model: Pipeline, x_eval: pd.DataFrame, positive_label: str) -> np.ndarray:
    proba = model.predict_proba(x_eval)
    classes = [str(c) for c in model.classes_]
    if positive_label not in classes:
        raise RuntimeError(f"Missing class '{positive_label}' in classifier classes: {classes}")
    return proba[:, classes.index(positive_label)]


def _predict_with_ovr_subclass_head(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    selected_features: list[str],
    model_name: str,
    sample_weight: np.ndarray,
    class_scales: dict[str, float],
    scale_mode: str,
) -> pd.Series:
    x_train = train_df[["text_input", *selected_features]].copy()
    x_eval = eval_df[["text_input", *selected_features]].copy()

    y_stage1 = pd.Series(
        np.where(train_df["label"].astype(str).eq("legit"), "legit", "suspicious"),
        index=train_df.index,
    )
    stage1 = _build_model(model_name, numeric_features=selected_features)
    stage1.fit(x_train, y_stage1, clf__sample_weight=sample_weight)
    suspicious_proba = _positive_class_proba(stage1, x_eval, positive_label="suspicious")
    pred_bin = np.where(suspicious_proba >= 0.5, "suspicious", "legit")

    suspicious_mask = train_df["label"].astype(str).ne("legit")
    suspicious_train = train_df.loc[suspicious_mask].copy()
    suspicious_weight = sample_weight[suspicious_mask.to_numpy()]
    subclass_models: dict[str, Pipeline] = {}
    subclass_labels = [label for label in SUSPICIOUS_LABELS if label in suspicious_train["label"].astype(str).unique().tolist()]

    if not subclass_labels:
        raise RuntimeError("No suspicious subclass labels available for OVR head")

    suspicious_x_train = suspicious_train[["text_input", *selected_features]].copy()
    for label in subclass_labels:
        target = suspicious_train["label"].astype(str).eq(label).astype(int)
        model = _build_model(model_name, numeric_features=selected_features)
        model.fit(suspicious_x_train, target, clf__sample_weight=suspicious_weight)
        subclass_models[label] = model

    pred_labels = pd.Series(["legit"] * len(eval_df), index=eval_df.index, dtype=str)
    eval_suspicious_mask = pd.Series(pred_bin, index=eval_df.index).eq("suspicious")
    if eval_suspicious_mask.any():
        x_susp_eval = x_eval.loc[eval_suspicious_mask].copy()
        scores = []
        for label in subclass_labels:
            label_scores = _positive_class_proba(subclass_models[label], x_susp_eval, positive_label="1")
            scale = float(class_scales.get(label, 1.0)) if class_scales else 1.0
            if scale_mode == "divide":
                label_scores = label_scores / scale
            else:
                label_scores = label_scores * scale
            scores.append(label_scores)
        score_matrix = np.column_stack(scores)
        best_idx = score_matrix.argmax(axis=1)
        pred_labels.loc[eval_suspicious_mask] = [subclass_labels[i] for i in best_idx]
    return pred_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate campaign with tuned scenario M model")
    parser.add_argument("--processed-scenario-dir", required=True)
    parser.add_argument("--analyzed-csv", required=True)
    parser.add_argument("--seed-csv", required=True)
    parser.add_argument("--metadata-csv", default="")
    parser.add_argument(
        "--ground-truth-csv",
        default="",
        help=(
            "Optional CSV with file->y_true_multiclass mapping (e.g. prior prediction export). "
            "When provided, this mapping is preferred over seed/metadata reconstruction."
        ),
    )
    parser.add_argument("--receiver-id-pattern", default=r"user(\d+)@")
    parser.add_argument(
        "--spoofing-policy",
        choices=["map_to_phishing", "drop", "keep"],
        default="map_to_phishing",
    )
    parser.add_argument("--model", default="hybrid_logreg", choices=["hybrid_logreg", "hybrid_sgd_log"])
    parser.add_argument(
        "--head-mode",
        choices=["multiclass", "ovr_subclass"],
        default="multiclass",
        help="Prediction head used after feature extraction.",
    )
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--feature-audit-csv", default="")
    parser.add_argument(
        "--adaptation-csv",
        default="",
        help="Optional non-leaky campaign-style adaptation rows mixed into train.",
    )
    parser.add_argument(
        "--adaptation-weight",
        type=float,
        default=0.35,
        help="Sample-weight multiplier for adaptation rows.",
    )
    parser.add_argument(
        "--spam-narrowing-mode",
        choices=["none", "marketing_only"],
        default="none",
        help="Optionally narrow broad spam training rows to marketing-like subset.",
    )
    parser.add_argument(
        "--class-scales",
        default="financial_fraud:1.25,legit:0.75,phishing:1.25,spam:1.25",
    )
    parser.add_argument("--scale-mode", choices=["divide", "multiply"], default="divide")
    parser.add_argument("--spam-weight", type=float, default=1.2)
    parser.add_argument("--phishing-weight", type=float, default=1.0)
    parser.add_argument("--fraud-weight", type=float, default=1.3)
    parser.add_argument("--hardneg-multiplier", type=float, default=1.0)
    parser.add_argument("--output-pred-csv", required=True)
    parser.add_argument("--output-summary-csv", required=True)
    parser.add_argument("--output-per-class-csv", required=True)
    parser.add_argument("--output-confusion-csv", required=True)
    args = parser.parse_args()

    scenario_dir = Path(args.processed_scenario_dir)
    train_path = scenario_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split: {train_path}")

    train_raw = pd.read_csv(train_path, low_memory=False)
    if args.max_train_rows > 0 and len(train_raw) > args.max_train_rows:
        train_raw = train_raw.sample(n=args.max_train_rows, random_state=42)
    train_raw, spam_narrow_meta = _narrow_spam_train_rows(
        train_raw,
        mode=str(args.spam_narrowing_mode),
    )
    train_raw, adaptation_meta = _mix_adaptation_rows(
        train_raw,
        adaptation_csv=str(args.adaptation_csv),
        adaptation_weight=float(args.adaptation_weight),
    )
    train_df = ensure_features(train_raw)
    train_df = _add_hard_negatives(train_df, multiplier=float(args.hardneg_multiplier), seed=42)

    analyzed_raw = pd.read_csv(args.analyzed_csv, low_memory=False)
    if "label" not in analyzed_raw.columns:
        analyzed_raw["label"] = "unknown"
    analyzed = ensure_features(analyzed_raw)

    selected_features = [feature for feature in NUMERIC_FEATURES if feature in train_df.columns]
    if args.feature_audit_csv:
        audit = pd.read_csv(args.feature_audit_csv)
        if "feature" in audit.columns and "keep" in audit.columns:
            keep = audit[audit["keep"].astype(bool)]["feature"].astype(str).tolist()
            selected_features = [feature for feature in selected_features if feature in keep]
    if not selected_features:
        raise RuntimeError("No numeric features selected for model input")

    seed_map = _build_seed_attack_map(Path(args.seed_csv), spoofing_policy=args.spoofing_policy)
    id_re = re.compile(args.receiver_id_pattern, flags=re.IGNORECASE)
    file_to_dataset_id = {}
    if args.metadata_csv:
        file_to_dataset_id = _build_file_to_dataset_id_map(Path(args.metadata_csv), id_re)

    analyzed["dataset_id"] = _resolve_dataset_id(analyzed, file_to_dataset_id, id_re)
    analyzed["y_true_multiclass"] = analyzed["dataset_id"].map(seed_map)

    if args.ground_truth_csv:
        gt_path = Path(args.ground_truth_csv)
        if gt_path.exists():
            gt = pd.read_csv(gt_path, low_memory=False)
            if "file" in gt.columns and "y_true_multiclass" in gt.columns:
                gt = gt.copy()
                gt["file_key"] = gt["file"].astype(str).map(lambda p: Path(str(p)).name)
                gt_map = (
                    gt.dropna(subset=["y_true_multiclass"])
                    .drop_duplicates("file_key", keep="last")
                    .set_index("file_key")["y_true_multiclass"]
                    .to_dict()
                )
                analyzed["file_key"] = analyzed["file"].astype(str).map(lambda p: Path(str(p)).name)
                mapped_gt = analyzed["file_key"].map(gt_map)
                analyzed["y_true_multiclass"] = analyzed["y_true_multiclass"].where(
                    analyzed["y_true_multiclass"].notna(),
                    mapped_gt,
                )
                if "dataset_id" in gt.columns:
                    gt_id_map = (
                        gt.dropna(subset=["dataset_id"]) 
                        .drop_duplicates("file_key", keep="last")
                        .set_index("file_key")["dataset_id"]
                        .to_dict()
                    )
                    mapped_id = pd.to_numeric(analyzed["file_key"].map(gt_id_map), errors="coerce").astype("Int64")
                    analyzed["dataset_id"] = analyzed["dataset_id"].where(analyzed["dataset_id"].notna(), mapped_id)
        else:
            print(f"[WARN] ground truth CSV not found: {gt_path}")

    analyzed["y_true_multiclass"] = analyzed["y_true_multiclass"].astype("object")
    unresolved_mask = analyzed["y_true_multiclass"].isna()
    if unresolved_mask.any() and "label" in analyzed.columns:
        fallback_labels = analyzed.loc[unresolved_mask, "label"].map(_normalize_attack_type)
        fallback_labels = fallback_labels.map(
            lambda x: _apply_spoofing_policy(x, args.spoofing_policy)
        )
        analyzed.loc[unresolved_mask, "y_true_multiclass"] = fallback_labels

    analyzed = analyzed[analyzed["y_true_multiclass"].notna()].copy()
    if analyzed.empty:
        raise RuntimeError("No rows with reconstructed multiclass ground truth")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["label"].astype(str))
    x_train = train_df[["text_input", *selected_features]].copy()

    sample_weight = _build_sample_weight(
        train_df=train_df,
        spam_weight=float(args.spam_weight),
        phishing_weight=float(args.phishing_weight),
        fraud_weight=float(args.fraud_weight),
    )

    x_eval = analyzed[["text_input", *selected_features]].copy()
    class_scales = _parse_class_scales(args.class_scales)
    if args.head_mode == "multiclass":
        model = _build_model(args.model, numeric_features=selected_features)
        model.fit(x_train, y_train, clf__sample_weight=sample_weight)
        pred_idx = _predict_with_multiclass_head(
            model=model,
            x_eval=x_eval,
            class_labels=[str(c) for c in label_encoder.classes_],
            class_scales=class_scales,
            scale_mode=str(args.scale_mode),
        )
        pred_label = pd.Series(label_encoder.inverse_transform(pred_idx), index=analyzed.index)
    else:
        pred_label = _predict_with_ovr_subclass_head(
            train_df=train_df,
            eval_df=analyzed,
            selected_features=selected_features,
            model_name=str(args.model),
            sample_weight=sample_weight,
            class_scales=class_scales,
            scale_mode=str(args.scale_mode),
        )
    y_true_mc = analyzed["y_true_multiclass"].astype(str)
    y_pred_mc = pred_label.astype(str)

    y_true_bin = pd.Series(np.where(y_true_mc.eq("legit"), "legit", "suspicious"), index=y_true_mc.index)
    y_pred_bin = pd.Series(np.where(y_pred_mc.eq("legit"), "legit", "suspicious"), index=y_pred_mc.index)
    y_true_bin_num = y_true_bin.eq("suspicious").astype(int)
    y_pred_bin_num = y_pred_bin.eq("suspicious").astype(int)

    summary = {
        "rows": int(len(y_true_mc)),
        "bin_accuracy": float(accuracy_score(y_true_bin_num, y_pred_bin_num)),
        "bin_balanced_accuracy": float(balanced_accuracy_score(y_true_bin_num, y_pred_bin_num)),
        "bin_precision": float(precision_score(y_true_bin_num, y_pred_bin_num, zero_division=0)),
        "bin_recall": float(recall_score(y_true_bin_num, y_pred_bin_num, zero_division=0)),
        "bin_f1": float(f1_score(y_true_bin_num, y_pred_bin_num, zero_division=0)),
        "mc_accuracy": float(accuracy_score(y_true_mc, y_pred_mc)),
        "mc_balanced_accuracy": float(balanced_accuracy_score(y_true_mc, y_pred_mc)),
        "mc_f1_macro": float(f1_score(y_true_mc, y_pred_mc, average="macro", zero_division=0)),
        "mc_f1_weighted": float(f1_score(y_true_mc, y_pred_mc, average="weighted", zero_division=0)),
        "model": args.model,
        "head_mode": args.head_mode,
        "feature_count": int(len(selected_features)),
        "scale_mode": args.scale_mode,
        "class_scales": args.class_scales,
        "spam_weight": float(args.spam_weight),
        "phishing_weight": float(args.phishing_weight),
        "fraud_weight": float(args.fraud_weight),
        "hardneg_multiplier": float(args.hardneg_multiplier),
        "adaptation_csv": str(args.adaptation_csv),
        "adaptation_weight": float(args.adaptation_weight),
        "adaptation_rows_added": int(adaptation_meta.get("rows_added", 0)),
        "spam_narrowing_mode": str(args.spam_narrowing_mode),
        "spam_rows_before": int(spam_narrow_meta.get("spam_rows_before", 0) or 0),
        "spam_rows_after": int(spam_narrow_meta.get("spam_rows_after", 0) or 0),
    }

    report = classification_report(y_true_mc, y_pred_mc, output_dict=True, zero_division=0)
    per_class_rows = []
    for key, value in report.items():
        if key in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if not isinstance(value, dict):
            continue
        per_class_rows.append(
            {
                "label": key,
                "precision": float(value.get("precision", 0.0)),
                "recall": float(value.get("recall", 0.0)),
                "f1": float(value.get("f1-score", 0.0)),
                "support": int(value.get("support", 0)),
            }
        )

    labels = sorted(set(y_true_mc.tolist()) | set(y_pred_mc.tolist()))
    conf = confusion_matrix(y_true_mc, y_pred_mc, labels=labels)
    conf_rows = []
    for i, true_label in enumerate(labels):
        for j, pred in enumerate(labels):
            conf_rows.append(
                {
                    "true_label": true_label,
                    "pred_label": pred,
                    "count": int(conf[i][j]),
                }
            )

    pred_cols = [
        col
        for col in [
            "file",
            "mailhog_id",
            "subject",
            "heuristic_score",
            "dataset_id",
            "y_true_multiclass",
        ]
        if col in analyzed.columns
    ]
    pred_out = analyzed[pred_cols].copy()
    pred_out["ml_pred"] = y_pred_mc.values
    pred_out["y_true_binary"] = y_true_bin.values
    pred_out["ml_pred_binary"] = y_pred_bin.values

    for path in [args.output_pred_csv, args.output_summary_csv, args.output_per_class_csv, args.output_confusion_csv]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    pred_out.to_csv(args.output_pred_csv, index=False)
    pd.DataFrame([summary]).to_csv(args.output_summary_csv, index=False)
    pd.DataFrame(per_class_rows).to_csv(args.output_per_class_csv, index=False)
    pd.DataFrame(conf_rows).to_csv(args.output_confusion_csv, index=False)

    print(f"Saved predictions: {args.output_pred_csv}")
    print(f"Saved summary: {args.output_summary_csv}")
    print(f"Saved per-class report: {args.output_per_class_csv}")
    print(f"Saved confusion table: {args.output_confusion_csv}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
