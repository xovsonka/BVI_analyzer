# Experiment 3 - URL Indicator Detection

This experiment validates URL-based phishing indicators and explainability.

## Track A - Controlled URL dataset

Goal: validate `expected vs detected` for URL indicators and binary decision logic.

### 1) Generate controlled `.eml` files

```bash
python experiments/experiment_3/generate_url_indicator_dataset.py --output-dir dataset/experiment_3/url_eml --expected-csv dataset/experiment_3/url_expected.csv --replicates 10 --clean-output
```

Scenarios:

- `clean_url`
- `ip_url`
- `shortener_url`
- `anchor_mismatch`
- `suspicious_tld`
- `brand_typosquat`
- `obfuscated_url`
- `combined_url_signals`
- `borderline_legit_url`

### 2) Analyze generated emails

```bash
python analysis/analyze_campaign.py --input-dir dataset/experiment_3/url_eml --output-csv results/experiment_3_url_model_input.csv --output-parts-jsonl results/experiment_3_url_parts.jsonl
```

### 3) Evaluate URL indicators + explainability

```bash
python evaluation/eval_experiment3_url_indicators.py --analyzed-csv results/experiment_3_url_model_input.csv --expected-csv dataset/experiment_3/url_expected.csv --output-detail-csv results/experiment_3_url_indicator_detail.csv --output-indicator-summary-csv results/experiment_3_url_indicator_summary.csv --output-binary-summary-csv results/experiment_3_url_binary_summary.csv --output-explainability-csv results/experiment_3_url_explainability.csv --output-score-by-case-csv results/experiment_3_url_score_by_case.csv --output-score-by-label-csv results/experiment_3_url_score_by_label.csv --output-threshold-sweep-csv results/experiment_3_url_threshold_sweep.csv --output-summary-txt results/experiment_3_url_summary.txt
```

Key outputs:

- `results/experiment_3_url_indicator_summary.csv`
- `results/experiment_3_url_binary_summary.csv`
- `results/experiment_3_url_explainability_summary.csv`
- `results/experiment_3_url_score_by_case.csv`
- `results/experiment_3_url_threshold_sweep.csv`

## Track B - Realistic campaign (GoPhish + MailHog)

### 1) Build URL-focused campaign seed CSV

```bash
python experiments/experiment_3/build_campaign_url_focused_dataset.py --output-csv results/experiment_3/gophish_seed_input_url_focused_50_50.csv --expected-csv results/experiment_3/gophish_seed_input_url_focused_expected.csv
```

Important Track B notes:

- Source URLs are stripped by default before injection, so URL tags stay controlled.
- Optional hardening: add `--strip-source-emails` to also remove source email addresses.
- `expected_*` columns are treated as injected scenario metadata, not strict ground truth.
- Anchor mismatch is disabled by default in Track B (GoPhish text-body flow may not preserve HTML anchors).
- To test anchor mismatch in real HTML rendering flow, add `--enable-anchor-mismatch` to the builder and use HTML templates in GoPhish.
- Seed CSV now includes `body_text` and `body_html` columns; `seed_gophish.py` uses `body_html` when provided.

### 2) Seed to GoPhish

```bash
python data_tools/seed_gophish.py --input-csv results/experiment_3/gophish_seed_input_url_focused_50_50.csv --mapping-csv dataset/processed/gophish_seed_map_exp3_url.csv --rows 9999 --campaign-prefix EXP3 --profile-mode by-sender --smtp-host mailhog:1025 --default-from "security-lab@example.com" --phish-url http://localhost:8080
```

### 3) Collect + analyze campaign emails

```bash
python data_tools/collect_mailhog.py --output-dir dataset/campaign_eml_exp3_url --metadata-csv dataset/processed/mailhog_messages_exp3_campaign_url.csv --limit 5000
python analysis/analyze_campaign.py --input-dir dataset/campaign_eml_exp3_url --output-csv results/experiment_3_campaign_model_input.csv --output-parts-jsonl results/experiment_3_campaign_parts.jsonl --metadata-csv dataset/processed/mailhog_messages_exp3_campaign_url.csv
```

### 4) Predict and build URL-focused report tables

```bash
python experiments/experiment_2/predict_campaign_with_final_model.py --analyzed-csv results/experiment_3_campaign_model_input.csv --seed-csv results/experiment_3/gophish_seed_input_url_focused_50_50.csv --metadata-csv dataset/processed/mailhog_messages_exp3_campaign_url.csv --max-train-rows 5000 --output-pred-csv results/experiment_3_campaign_ml_predictions.csv --output-summary-csv results/experiment_3_campaign_ml_summary.csv
python experiments/experiment_3/report_url_campaign_results.py --analyzed-csv results/experiment_3_campaign_model_input.csv --ml-pred-csv results/experiment_3_campaign_ml_predictions.csv --seed-csv results/experiment_3/gophish_seed_input_url_focused_50_50.csv --expected-csv results/experiment_3/gophish_seed_input_url_focused_expected.csv --output-indicator-summary-csv results/experiment_3_campaign_url_indicator_summary.csv --output-method-compare-csv results/experiment_3_campaign_method_compare.csv --output-confusion-csv results/experiment_3_campaign_binary_confusion_counts.csv --output-correlation-csv results/experiment_3_campaign_url_score_correlation.csv --output-error-impact-csv results/experiment_3_campaign_url_error_impact.csv --output-qualitative-csv results/experiment_3_campaign_qualitative_examples.csv
```

Key outputs:

- `results/experiment_3_campaign_url_indicator_summary.csv`
- `results/experiment_3_campaign_method_compare.csv`
- `results/experiment_3_campaign_url_score_correlation.csv`
- `results/experiment_3_campaign_url_error_impact.csv`
- `results/experiment_3_campaign_qualitative_examples.csv`
- `results/experiment_3_campaign_binary_confusion_counts.csv`
- `results/experiment_3_campaign_injected_indicator_retention.csv`
- `results/experiment_3_campaign_multiclass_summary.csv`
- `results/experiment_3_campaign_multiclass_per_class.csv`
- `results/experiment_3_campaign_multiclass_confusion_long.csv`
- `results/experiment_3_campaign_threshold_sweep.csv`
- `results/experiment_3_campaign_slice_metrics.csv`
