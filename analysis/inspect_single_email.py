from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyze_campaign import (
    build_model_input_row,
    compute_heuristic_score,
    extract_features_from_parts,
    parse_eml_file,
)
from modeling_core import build_model, ensure_features, select_model_input


def _print_section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def _bool(v: object) -> bool:
    try:
        return float(v) > 0
    except Exception:
        return False


def _format_float(v: float) -> str:
    return f"{v:.4f}" if np.isfinite(v) else "n/a"


def _binary_from_multiclass(label: str) -> str:
    return "legit" if str(label).strip().lower() == "legit" else "suspicious"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect single .eml with heuristic and ML outputs"
    )
    parser.add_argument("--email", required=True, help="Path to .eml file")
    parser.add_argument(
        "--processed-dir",
        default="dataset/processed/scenario_m_rebalanced_fraud_spam",
        help="Scenario directory with train.csv",
    )
    parser.add_argument("--model", default="hybrid_logreg")
    parser.add_argument("--input-mode", default="text_plus_features")
    parser.add_argument("--heur-threshold", type=int, default=20)
    parser.add_argument("--max-train-rows", type=int, default=30000)
    parser.add_argument(
        "--class-scales",
        default="financial_fraud:1.1",
        help="Optional class scales class:scale,class:scale",
    )
    args = parser.parse_args()

    email_path = Path(args.email)
    if not email_path.exists():
        raise FileNotFoundError(f"Email file not found: {email_path}")

    parts = parse_eml_file(email_path)
    features = extract_features_from_parts(parts)
    heuristic = compute_heuristic_score(features)
    row = build_model_input_row(parts, features, heuristic)
    eval_base = pd.DataFrame([row])
    eval_base["label"] = "unknown"
    eval_df = ensure_features(eval_base)

    train_path = Path(args.processed_dir) / "train.csv"
    train_df = ensure_features(
        pd.read_csv(
            train_path,
            low_memory=False,
            nrows=args.max_train_rows if args.max_train_rows > 0 else None,
        )
    )
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"].astype(str))
    model = build_model(args.model, input_mode=args.input_mode)
    model.fit(select_model_input(train_df, args.input_mode), y_train)

    x_eval = select_model_input(eval_df, args.input_mode)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_eval)[0]
        classes = [str(c) for c in le.classes_]
        scale_map = {}
        if args.class_scales.strip():
            for item in args.class_scales.split(","):
                k, v = item.split(":", 1)
                scale_map[k.strip()] = float(v.strip())
        scale_vec = np.array([scale_map.get(c, 1.0) for c in classes], dtype=float)
        calibrated = proba * scale_vec
        calibrated = calibrated / calibrated.sum()
        pred_idx = int(np.argmax(calibrated))
        ml_pred = classes[pred_idx]
        top = sorted(zip(classes, calibrated), key=lambda x: x[1], reverse=True)
    else:
        pred_idx = int(model.predict(x_eval)[0])
        ml_pred = str(le.inverse_transform([pred_idx])[0])
        top = [(ml_pred, 1.0)]

    ml_binary = _binary_from_multiclass(ml_pred)
    heur_binary = (
        "suspicious"
        if int(heuristic.get("heuristic_score", 0)) >= int(args.heur_threshold)
        else "legit"
    )

    indicator_pairs = [
        ("spf_fail", str(features.get("spf_result", "")).lower() == "fail"),
        ("dkim_fail", str(features.get("dkim_result", "")).lower() == "fail"),
        ("dmarc_fail", str(features.get("dmarc_result", "")).lower() == "fail"),
        ("from_reply_mismatch", _bool(features.get("from_reply_mismatch", 0))),
        (
            "from_return_path_mismatch",
            _bool(features.get("from_return_path_mismatch", 0)),
        ),
        (
            "message_id_domain_mismatch",
            _bool(features.get("message_id_domain_mismatch", 0)),
        ),
        ("display_name_spoof", _bool(features.get("display_name_spoof_flag", 0))),
        ("received_anomaly", _bool(features.get("received_anomaly_flag", 0))),
        ("ip_url", _bool(features.get("ip_url_count", 0))),
        ("shortener_url", _bool(features.get("shortener_url_count", 0))),
        ("anchor_mismatch", _bool(features.get("mismatched_anchor_count", 0))),
        ("suspicious_tld", _bool(features.get("suspicious_tld_count", 0))),
        ("brand_typosquat", _bool(features.get("brand_typosquat_flag", 0))),
        ("obfuscated_url", _bool(features.get("obfuscated_url_count", 0))),
        ("risky_attachment", _bool(features.get("risky_attachment_ext_count", 0))),
    ]

    _print_section("Email")
    print(f"File: {email_path}")
    print(f"Subject: {features.get('subject', '')}")
    print(f"From: {features.get('from', '')}")
    print(f"To: {features.get('to', '')}")
    print(f"URL count: {features.get('url_count', 0)}")

    _print_section("Heuristic")
    print(f"Score: {heuristic.get('heuristic_score', 0)} / 100")
    print(f"Risk level: {heuristic.get('risk_level', 'unknown')}")
    print(f"Binary decision (threshold={args.heur_threshold}): {heur_binary}")
    print("Reasons: " + (heuristic.get("heuristic_reasons", "") or "none"))

    _print_section("Detected Flags")
    any_flag = False
    for name, flag in indicator_pairs:
        if flag:
            any_flag = True
            print(f"- {name}")
    if not any_flag:
        print("- none")

    _print_section("ML")
    print(f"Model: {args.model} ({args.input_mode})")
    print(f"Scenario: {args.processed_dir}")
    print(f"Predicted class: {ml_pred}")
    print(f"Binary decision: {ml_binary}")
    print("Top class probabilities:")
    for cls, prob in top[:4]:
        print(f"- {cls}: {_format_float(float(prob))}")

    _print_section("Summary")
    print(f"Heuristic => {heur_binary} | ML => {ml_binary} ({ml_pred})")


if __name__ == "__main__":
    main()
