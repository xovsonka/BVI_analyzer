from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import NUMERIC_FEATURES, ensure_features


SEMANTIC_FEATURES = [
    "cta_login_flag",
    "cta_verify_flag",
    "cta_recovery_flag",
    "account_security_phrase_count",
    "cta_payment_flag",
    "cta_bank_transfer_flag",
    "invoice_reference_flag",
    "remittance_reference_flag",
    "currency_amount_pattern_flag",
    "due_date_pattern_flag",
    "spam_reward_flag",
    "spam_unsubscribe_context_flag",
    "spam_marketing_phrase_count",
]


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


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "label" not in df.columns:
        df["label"] = "unknown"
    return ensure_features(df)


def _eval_shap(model: lgb.LGBMClassifier, df: pd.DataFrame, feature_names: list[str], max_rows: int) -> pd.DataFrame:
    src = df if len(df) <= max_rows else df.sample(n=max_rows, random_state=42)
    x_eval = src[feature_names].copy()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_eval)
    importance = _mean_abs_shap_per_feature(shap_values, feature_count=len(feature_names))
    return pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance}).sort_values("mean_abs_shap", ascending=False)


def main() -> None:
    train_path = PROJECT_ROOT / "dataset" / "processed" / "scenario_m_anti_template_feature_regularized" / "train.csv"
    adaptation_path = PROJECT_ROOT / "results" / "adaptation" / "campaign_style_train_adaptation.csv"
    eval_paths = {
        "test_deployment": PROJECT_ROOT / "dataset" / "processed" / "shared" / "test_deployment.csv",
        "exp1_campaign": PROJECT_ROOT / "results" / "experiment_5" / "exp1_campaign_model_input.csv",
        "exp2_campaign": PROJECT_ROOT / "results" / "experiment_2_clean" / "exp2_existing_reanalyzed_model_input.csv",
        "exp3_campaign": PROJECT_ROOT / "results" / "experiment_3_clean" / "exp3_existing_reanalyzed_model_input.csv",
    }

    out_dir = PROJECT_ROOT / "results" / "semantic_shap_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load(train_path)
    adaptation_rows_added = 0
    if adaptation_path.exists():
        adapt_df = _load(adaptation_path)
        adaptation_rows_added = int(len(adapt_df))
        train_df = pd.concat([train_df, adapt_df], ignore_index=True, sort=False)
    feature_names = [feature for feature in NUMERIC_FEATURES if feature in train_df.columns]

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"].astype(str))
    x_train = train_df[feature_names].copy()

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(le.classes_),
        n_estimators=320,
        learning_rate=0.05,
        num_leaves=63,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)

    ratio_rows = []
    for feature in feature_names:
        values = pd.to_numeric(train_df[feature], errors="coerce").fillna(0.0)
        ratio_rows.append(
            {
                "feature": feature,
                "is_semantic": feature in SEMANTIC_FEATURES,
                "train_non_zero_ratio": float((values != 0).sum() / max(1, len(values))),
                "train_mean": float(values.mean()),
            }
        )
    stats_df = pd.DataFrame(ratio_rows)

    top_summary: dict[str, list[dict[str, object]]] = {}
    for name, path in eval_paths.items():
        eval_df = _load(path)
        shap_df = _eval_shap(model, eval_df, feature_names=feature_names, max_rows=6000)
        shap_df.to_csv(out_dir / f"{name}_shap.csv", index=False)
        top_summary[name] = shap_df.head(15).to_dict(orient="records")
        stats_df = stats_df.merge(
            shap_df.rename(columns={"mean_abs_shap": f"mean_abs_shap_{name}"}),
            on="feature",
            how="left",
        )

    shap_cols = [c for c in stats_df.columns if c.startswith("mean_abs_shap_")]
    stats_df["mean_abs_shap_overall"] = stats_df[shap_cols].mean(axis=1)
    stats_df = stats_df.sort_values(["is_semantic", "mean_abs_shap_overall"], ascending=[False, False])
    stats_df.to_csv(out_dir / "feature_audit.csv", index=False)

    semantic_df = stats_df[stats_df["is_semantic"]].copy()
    semantic_df.to_csv(out_dir / "semantic_feature_audit.csv", index=False)

    summary = {
        "train_rows": int(len(train_df)),
        "adaptation_rows_added": adaptation_rows_added,
        "feature_count": int(len(feature_names)),
        "semantic_features": SEMANTIC_FEATURES,
        "top_semantic_by_overall_shap": semantic_df.head(10).to_dict(orient="records"),
        "top_features_per_eval": top_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved semantic SHAP audit to: {out_dir}")


if __name__ == "__main__":
    main()
