from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prepare_dataset_multiclass import (
    add_hard_legit_samples,
    add_engineered_features,
    add_label_id,
    apply_llm_bonus_augmentation,
    make_mild_rebalanced_train,
    save_scenario_outputs,
)


REQUIRED_FEATURE_COLS = {
    "url_count",
    "ip_url_count",
    "shortener_url_count",
    "unique_domain_count",
    "from_reply_mismatch",
    "from_return_path_mismatch",
    "sender_receiver_domain_mismatch",
    "urgent_keyword_count",
    "credential_keyword_count",
    "threat_keyword_count",
    "financial_keyword_count",
    "subject_len",
    "text_len",
    "exclamation_count",
    "uppercase_ratio",
    "digit_ratio",
    "has_html_only",
    "attachment_count",
    "risky_attachment_ext_count",
    "received_hops_count",
    "suspicious_tld_count",
    "brand_typosquat_flag",
    "trusted_domain_only",
}


def ensure_engineered(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_FEATURE_COLS if c not in df.columns]
    if missing:
        return add_engineered_features(df)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate only scenario F from already prepared shared splits"
    )
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument("--llm-labels", default="spam,financial_fraud")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-max-generations", type=int, default=300)
    parser.add_argument("--llm-max-fraction-per-class", type=float, default=0.30)
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-max-tokens", type=int, default=500)
    parser.add_argument("--llm-timeout", type=int, default=60)
    parser.add_argument("--llm-sleep-ms", type=int, default=200)
    parser.add_argument("--f-majority-under-factor", type=float, default=0.6)
    parser.add_argument("--f-minority-over-factor", type=float, default=1.5)
    parser.add_argument("--hard-legit-ratio", type=float, default=0.15)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "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")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")

    processed_dir = Path(args.processed_dir)
    train_path = processed_dir / "shared" / "train_full.csv"
    val_path = processed_dir / "shared" / "val.csv"
    test_path = processed_dir / "shared" / "test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required shared split file: {p}")

    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    train_df = ensure_engineered(train_df)
    val_df = ensure_engineered(val_df)
    test_df = ensure_engineered(test_df)

    llm_labels = {x.strip() for x in args.llm_labels.split(",") if x.strip()}

    # Scenario F pipeline: mild rebalance + LLM bonus augmentation
    train_f_base = make_mild_rebalanced_train(
        train_df,
        majority_under_factor=args.f_majority_under_factor,
        minority_over_factor=args.f_minority_over_factor,
    )
    train_f, llm_generated_counts = apply_llm_bonus_augmentation(
        train_f_base,
        llm_labels=llm_labels,
        api_key=api_key,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_timeout=args.llm_timeout,
        llm_samples_per_label=args.llm_max_generations,
        llm_max_fraction_per_class=args.llm_max_fraction_per_class,
        llm_sleep_ms=args.llm_sleep_ms,
    )
    train_f = add_hard_legit_samples(
        train_f,
        ratio=args.hard_legit_ratio,
        strategy="append",
    )
    train_f = add_engineered_features(train_f)

    train_f["split"] = "train"
    val_df = val_df.copy()
    test_df = test_df.copy()
    val_df["split"] = "val"
    test_df["split"] = "test"

    train_f = add_label_id(train_f)
    val_df = add_label_id(val_df)
    test_df = add_label_id(test_df)

    save_scenario_outputs(
        output_dir=processed_dir,
        scenario_name="scenario_f_llm_augmented",
        train_df=train_f,
        val_df=val_df,
        test_df=test_df,
        extra_meta={
            "description": "Train set with mild majority undersampling + mild minority oversampling, then LLM bonus augmentation for selected labels.",
            "llm_model": args.llm_model,
            "llm_labels": sorted(llm_labels),
            "llm_max_generations": args.llm_max_generations,
            "llm_max_fraction_per_class": args.llm_max_fraction_per_class,
            "llm_generated_by_label": llm_generated_counts,
            "majority_under_factor": args.f_majority_under_factor,
            "minority_over_factor": args.f_minority_over_factor,
            "hard_legit_ratio": args.hard_legit_ratio,
        },
    )

    print("Done")
    print(f"Saved scenario F to: {processed_dir / 'scenario_f_llm_augmented'}")


if __name__ == "__main__":
    main()
