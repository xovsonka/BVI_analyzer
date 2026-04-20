from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import build_model, ensure_features


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def get_probabilities(model, x_df: pd.DataFrame, n_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_df)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_df)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-scores))
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T
        return softmax(scores)

    # fallback: one-hot from predictions
    pred = model.predict(x_df)
    probs = np.zeros((len(pred), n_classes), dtype=float)
    for i, p in enumerate(pred):
        probs[i, int(p)] = 1.0
    return probs


def risk_band(score_0_100: float) -> str:
    if score_0_100 < 30:
        return "low"
    if score_0_100 < 60:
        return "medium"
    return "high"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute final hybrid risk score (heuristic + ML probability) for EML-derived rows"
    )
    parser.add_argument("--scenario", default="scenario_b_full_weighted")
    parser.add_argument(
        "--model",
        default="hybrid_logreg",
        choices=[
            "hybrid_logreg",
            "hybrid_linear_svc_cal",
            "hybrid_sgd_log",
            "hybrid_sgd_hinge",
        ],
    )
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument("--eml-input", default="results/eml_model_input.csv")
    parser.add_argument("--output-csv", default="results/final_hybrid_eval.csv")
    parser.add_argument("--output-json", default="results/final_hybrid_summary.json")
    parser.add_argument("--benign-labels", default="legit")
    parser.add_argument(
        "--alpha-heuristic",
        type=float,
        default=0.5,
        help="Weight for heuristic score component",
    )
    parser.add_argument(
        "--beta-ml",
        type=float,
        default=0.5,
        help="Weight for ML suspicious probability component",
    )
    args = parser.parse_args()

    if args.alpha_heuristic < 0 or args.beta_ml < 0:
        raise ValueError("alpha/beta must be non-negative")
    if args.alpha_heuristic + args.beta_ml == 0:
        raise ValueError("alpha + beta must be > 0")

    alpha = args.alpha_heuristic / (args.alpha_heuristic + args.beta_ml)
    beta = args.beta_ml / (args.alpha_heuristic + args.beta_ml)

    processed_dir = Path(args.processed_dir)
    scenario_dir = processed_dir / args.scenario
    train_path = scenario_dir / "train.csv"
    val_path = scenario_dir / "val.csv"
    eml_input_path = Path(args.eml_input)

    for p in [train_path, val_path, eml_input_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    train_df = ensure_features(pd.read_csv(train_path, low_memory=False))
    val_df = ensure_features(pd.read_csv(val_path, low_memory=False))
    fit_df = pd.concat([train_df, val_df], ignore_index=True)

    inference_df = pd.read_csv(eml_input_path, low_memory=False)
    if "label" not in inference_df.columns:
        inference_df["label"] = "legit"
    inference_df = ensure_features(inference_df)

    labels = sorted(set(fit_df["label"].astype(str).tolist()))
    le = LabelEncoder()
    le.fit(labels)

    y_fit = le.transform(fit_df["label"].astype(str))
    model = build_model(args.model)
    model.fit(fit_df, y_fit)

    probs = get_probabilities(model, inference_df, n_classes=len(le.classes_))
    pred_idx = np.argmax(probs, axis=1)
    pred_label = le.inverse_transform(pred_idx)

    benign_labels = {x.strip() for x in args.benign_labels.split(",") if x.strip()}
    benign_indices = {i for i, name in enumerate(le.classes_) if name in benign_labels}
    suspicious_prob = np.array(
        [
            1.0 - row[list(benign_indices)].sum() if benign_indices else row.max()
            for row in probs
        ]
    )

    heur = pd.to_numeric(
        inference_df.get("heuristic_score", 0), errors="coerce"
    ).fillna(0.0)
    heur_norm = np.clip(heur / 100.0, 0.0, 1.0)
    final_risk = 100.0 * (alpha * heur_norm + beta * suspicious_prob)

    out = inference_df.copy()
    out["ml_pred_label"] = pred_label
    out["ml_suspicious_prob"] = suspicious_prob
    out["heuristic_norm"] = heur_norm
    out["final_risk_score"] = final_risk.round(3)
    out["final_risk_level"] = out["final_risk_score"].map(risk_band)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    summary = {
        "scenario": args.scenario,
        "model": args.model,
        "weights": {"alpha_heuristic": alpha, "beta_ml": beta},
        "rows": int(len(out)),
        "final_risk_level_distribution": out["final_risk_level"]
        .value_counts()
        .to_dict(),
        "avg_final_risk_score": float(out["final_risk_score"].mean())
        if len(out)
        else 0.0,
        "top5_high_risk_files": out.sort_values("final_risk_score", ascending=False)[
            ["file", "final_risk_score", "ml_pred_label"]
        ]
        .head(5)
        .to_dict(orient="records"),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved hybrid CSV: {output_csv}")
    print(f"Saved hybrid summary: {output_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
