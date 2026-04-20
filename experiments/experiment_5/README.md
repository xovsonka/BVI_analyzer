# Experiment 5 - Detection Approach Comparison

Experiment 5 compares detection approaches on one unified benchmark package:

- `heuristic_only`
- `ml_only`
- `hybrid_a_conservative`
- `hybrid_b_gated`

Evaluation is reported in two regimes:

- Binary: `legit` vs `suspicious`
- Multiclass: `legit`, `phishing`, `spam`, `financial_fraud`

## 0) Prepare Exp1 campaign predictions (if missing)

```bash
python analysis/analyze_campaign.py --input-dir dataset/campaign_eml_exp1sender --output-csv results/experiment_5/exp1_campaign_model_input.csv --output-parts-jsonl results/experiment_5/exp1_campaign_parts.jsonl --metadata-csv dataset/processed/mailhog_messages_exp1sender.csv

python experiments/experiment_2/predict_campaign_with_final_model.py --analyzed-csv results/experiment_5/exp1_campaign_model_input.csv --seed-csv results/experiment_1/gophish_seed_input_all_clean.csv --metadata-csv dataset/processed/mailhog_messages_exp1sender.csv --max-train-rows 5000 --output-pred-csv results/experiment_5/exp1_campaign_ml_predictions.csv --output-summary-csv results/experiment_5/exp1_campaign_ml_summary.csv
```

## 1) Build unified benchmark dataset (A+B+C)

```bash
python experiments/experiment_5/build_exp5_benchmark.py --exp1-analyzed-csv results/experiment_5/exp1_campaign_model_input.csv --exp1-pred-csv results/experiment_5/exp1_campaign_ml_predictions.csv --exp2-analyzed-csv results/eperiment_2_results/experiment_2_campaign_model_input.csv --exp2-pred-csv results/eperiment_2_results/experiment_2_campaign_ml_predictions.csv --exp3-analyzed-csv results/experiment_3_result/experiment_3_campaign_model_input.csv --exp3-pred-csv results/experiment_3_result/experiment_3_campaign_ml_predictions.csv --exp2-tracka-analyzed-csv results/eperiment_2_results/experiment_2_header_auth_model_input.csv --exp2-tracka-expected-csv dataset/experiment_2/header_auth_expected.csv --exp3-tracka-analyzed-csv results/experiment_3_result/experiment_3_url_model_input.csv --exp3-tracka-expected-csv dataset/experiment_3/url_expected.csv --deployment-test-csv dataset/processed/shared/test_deployment.csv --deployment-sample-rows 800 --output-csv results/experiment_5/exp5_unified_benchmark.csv --output-summary-csv results/experiment_5/exp5_unified_benchmark_summary.csv
```

## 2) Evaluate methods

```bash
python experiments/experiment_5/eval_exp5_methods.py --benchmark-csv results/experiment_5/exp5_unified_benchmark.csv --processed-dir dataset/processed/scenario_m_rebalanced_fraud_spam --model hybrid_logreg --input-mode text_plus_features --heur-threshold 20 --hybrid-a-high-threshold 35 --hybrid-b-fallback phishing --output-binary-csv results/experiment_5/exp5_method_compare_binary.csv --output-multiclass-csv results/experiment_5/exp5_method_compare_multiclass.csv --output-per-class-csv results/experiment_5/exp5_per_class_multiclass.csv --output-confusion-binary-csv results/experiment_5/exp5_confusion_binary_counts.csv --output-confusion-multiclass-csv results/experiment_5/exp5_confusion_multiclass_long.csv --output-source-breakdown-csv results/experiment_5/exp5_source_breakdown.csv --output-qualitative-csv results/experiment_5/exp5_qualitative_examples.csv
```

## Key outputs

- `results/experiment_5/exp5_method_compare_binary.csv`
- `results/experiment_5/exp5_method_compare_multiclass.csv`
- `results/experiment_5/exp5_per_class_multiclass.csv`
- `results/experiment_5/exp5_confusion_binary_counts.csv`
- `results/experiment_5/exp5_confusion_multiclass_long.csv`
- `results/experiment_5/exp5_source_breakdown.csv`
- `results/experiment_5/exp5_qualitative_examples.csv`
