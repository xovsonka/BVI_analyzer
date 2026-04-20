# Experiment 2 - Header and Authentication Indicators

This experiment is split into two complementary tracks.

## Track A - Controlled validation dataset

Goal: verify indicator extraction exactly (`expected` vs `detected`) for header/auth signals.

### 1) Generate controlled EML dataset

```bash
python experiments/experiment_2/generate_header_auth_dataset.py --output-dir dataset/experiment_2/header_auth_eml --expected-csv dataset/experiment_2/header_auth_expected.csv --replicates 6
```

Cases included:

- SPF pass/fail
- DKIM pass/fail
- DMARC pass/fail
- From vs Reply-To mismatch
- From vs Return-Path mismatch
- Message-ID domain mismatch
- display-name spoofing flag
- Received anomaly
- trusted domain guardrail
- combined cases

### 2) Analyze generated EML files

```bash
python analysis/analyze_campaign.py --input-dir dataset/experiment_2/header_auth_eml --output-csv results/experiment_2_header_auth_model_input.csv --output-parts-jsonl results/experiment_2_header_auth_parts.jsonl
```

### 3) Evaluate per-indicator correctness

```bash
python evaluation/eval_experiment2_indicators.py --analyzed-csv results/experiment_2_header_auth_model_input.csv --expected-csv dataset/experiment_2/header_auth_expected.csv --output-detail-csv results/experiment_2_indicator_detail.csv --output-summary-csv results/experiment_2_indicator_summary.csv --output-summary-txt results/experiment_2_indicator_summary.txt
```

Key outputs:

- `results/experiment_2_indicator_detail.csv` (file, indicator, expected, detected, correct)
- `results/experiment_2_indicator_summary.csv` (accuracy/precision/recall per indicator)

## Track B - Realistic campaign (GoPhish + MailHog)

Goal: evaluate indicators and model behavior on campaign-like flow.

### 1) Build mixed campaign seed CSV (100 emails, 50/50)

```bash
python experiments/experiment_2/build_campaign_mixed_dataset.py --output-csv results/experiment_2/gophish_seed_input_mixed_50_50.csv
```

### 1b) Build header-focused campaign seed CSV (recommended for Exp2)

```bash
python experiments/experiment_2/build_campaign_header_focused_dataset.py --output-csv results/experiment_2/gophish_seed_input_header_focused_50_50.csv --expected-csv results/experiment_2/gophish_seed_input_header_focused_expected.csv
```

This variant keeps the same 100-row and 50/50 generation mix, but injects stronger
header/auth-related signals (display-name spoof style senders, sender/url mismatch,
suspicious TLD URLs) and stores expected campaign-level flags.

Default class distribution:

- legit: 20
- phishing: 35
- spam: 20
- financial_fraud: 25

Generation mix:

- 50 LLM-assisted
- 50 rule-based / real-edits

### 2) Seed campaign to GoPhish

```bash
python data_tools/seed_gophish.py --input-csv results/experiment_2/gophish_seed_input_header_focused_50_50.csv --mapping-csv dataset/processed/gophish_seed_map_exp2.csv --rows 9999 --campaign-prefix EXP2 --profile-mode by-sender --smtp-host mailhog:1025 --default-from "security-lab@example.com" --phish-url http://localhost:8080
```

### 3) Collect and analyze campaign emails

```bash
python data_tools/collect_mailhog.py --output-dir dataset/campaign_eml_exp2 --metadata-csv dataset/processed/mailhog_messages_exp2_campaign.csv --limit 5000
python analysis/analyze_campaign.py --input-dir dataset/campaign_eml_exp2 --output-csv results/experiment_2_campaign_model_input.csv --output-parts-jsonl results/experiment_2_campaign_parts.jsonl --metadata-csv dataset/processed/mailhog_messages_exp2_campaign.csv
```

### 4) Predict with final model and build report tables

```bash
python experiments/experiment_2/predict_campaign_with_final_model.py --analyzed-csv results/experiment_2_campaign_model_input.csv --seed-csv results/experiment_2/gophish_seed_input_header_focused_50_50.csv --metadata-csv dataset/processed/mailhog_messages_exp2_campaign.csv --output-pred-csv results/experiment_2_campaign_ml_predictions.csv --output-summary-csv results/experiment_2_campaign_ml_summary.csv
python experiments/experiment_2/report_campaign_results.py --analyzed-csv results/experiment_2_campaign_model_input.csv --ml-pred-csv results/experiment_2_campaign_ml_predictions.csv --seed-csv results/experiment_2/gophish_seed_input_header_focused_50_50.csv --expected-csv results/experiment_2/gophish_seed_input_header_focused_expected.csv --output-summary-csv results/experiment_2_campaign_indicator_summary.csv --output-method-compare-csv results/experiment_2_campaign_method_compare.csv --output-qualitative-csv results/experiment_2_campaign_qualitative_examples.csv
```

Key outputs:

- `results/experiment_2_campaign_indicator_summary.csv`
- `results/experiment_2_campaign_mean_score_by_class.csv`
- `results/experiment_2_campaign_method_compare.csv`
- `results/experiment_2_campaign_binary_confusion_counts.csv`
- `results/experiment_2_campaign_multiclass_summary.csv`
- `results/experiment_2_campaign_multiclass_per_class.csv`
- `results/experiment_2_campaign_multiclass_confusion_long.csv`
- `results/experiment_2_campaign_threshold_sweep.csv`
- `results/experiment_2_campaign_injected_indicator_retention.csv`
- `results/experiment_2_campaign_slice_metrics.csv`
- `results/experiment_2_campaign_qualitative_examples.csv`

These match the final reporting goals for Experiment 2:

- mismatch/spoof/risky URL counts
- average heuristic score by class
- heuristic vs ML vs hybrid comparison
- heuristic threshold sensitivity
- injected indicator retention after transport
- ML multiclass confusion and per-slice error analysis
- short qualitative table (5-8 examples)
