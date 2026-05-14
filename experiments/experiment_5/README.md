# Experiment 5 - Detection Approach Comparison

Experiment 5 compares detection approaches on one unified benchmark package:

- `heuristic_only`
- `ml_only`
- `hybrid_a_conservative`
- `hybrid_b_gated`

Evaluation is reported in two regimes:

- Binary: `legit` vs `suspicious`
- Multiclass: `legit`, `phishing`, `spam`, `financial_fraud`

## 0) Prepare semantic campaign predictions (if missing)

```bash
python analysis/analyze_campaign.py --input-dir dataset/campaign_eml_exp1sender --output-csv results/experiment_5/exp1_campaign_model_input.csv --output-parts-jsonl results/experiment_5/exp1_campaign_parts.jsonl --metadata-csv dataset/processed/mailhog_messages_exp1sender.csv

python evaluation/eval_campaign_with_tuned_model.py --processed-scenario-dir dataset/processed/scenario_m_anti_template_feature_regularized --analyzed-csv results/experiment_5/exp1_campaign_model_input.csv --seed-csv results/experiment_5/exp1_campaign_ml_predictions.csv --ground-truth-csv results/experiment_5/exp1_campaign_ml_predictions.csv --model hybrid_logreg --head-mode ovr_subclass --feature-audit-csv results/m_shap_optimization_m_original/feature_audit.csv --adaptation-csv results/adaptation/campaign_style_train_adaptation.csv --adaptation-weight 0.5 --spam-narrowing-mode marketing_only --class-scales financial_fraud:1,legit:1,phishing:1,spam:1 --scale-mode divide --spam-weight 1.2 --phishing-weight 1.0 --fraud-weight 1.3 --hardneg-multiplier 1.0 --ovr-phishing-spam-hardneg 1.6 --ovr-fraud-spam-hardneg 1.6 --output-pred-csv results/retuned_exp1234/semantic_cta_eval/exp1_spamboost_ps160_fs160_pred.csv --output-summary-csv results/retuned_exp1234/semantic_cta_eval/exp1_spamboost_ps160_fs160_summary.csv --output-per-class-csv results/retuned_exp1234/semantic_cta_eval/exp1_spamboost_ps160_fs160_per_class.csv --output-confusion-csv results/retuned_exp1234/semantic_cta_eval/exp1_spamboost_ps160_fs160_conf.csv
```

## 1) Build unified benchmark dataset (A+B+C)

```bash
python experiments/experiment_5/build_exp5_benchmark.py --exp1-analyzed-csv results/experiment_5/exp1_campaign_model_input.csv --exp1-pred-csv results/retuned_exp1234/semantic_cta_eval/exp1_spamboost_ps160_fs160_pred.csv --exp2-analyzed-csv results/eperiment_2_results/experiment_2_campaign_model_input.csv --exp2-pred-csv results/retuned_exp1234/semantic_cta_eval/exp2_spamboost_ps130_fs160_pred.csv --exp3-analyzed-csv results/experiment_3_result/experiment_3_campaign_model_input.csv --exp3-pred-csv results/retuned_exp1234/semantic_cta_eval/exp3_ps160_fs160_pred.csv --exp2-tracka-analyzed-csv results/eperiment_2_results/experiment_2_header_auth_model_input.csv --exp2-tracka-expected-csv dataset/experiment_2/header_auth_expected.csv --exp3-tracka-analyzed-csv results/experiment_3_result/experiment_3_url_model_input.csv --exp3-tracka-expected-csv dataset/experiment_3/url_expected.csv --deployment-test-csv dataset/processed/shared/test_deployment.csv --deployment-sample-rows 800 --output-csv results/experiment_5/exp5_unified_benchmark.csv --output-summary-csv results/experiment_5/exp5_unified_benchmark_summary.csv
```

## 2) Evaluate methods

```bash
python experiments/experiment_5/eval_exp5_methods.py --benchmark-csv results/experiment_5/exp5_unified_benchmark_semantic.csv --external-ml-pred-csv results/experiment_5/exp5_semantic_ovr_predictions.csv --output-binary-csv results/experiment_5/exp5_method_compare_binary.csv --output-multiclass-csv results/experiment_5/exp5_method_compare_multiclass.csv --output-per-class-csv results/experiment_5/exp5_per_class_multiclass.csv --output-confusion-binary-csv results/experiment_5/exp5_confusion_binary_counts.csv --output-confusion-multiclass-csv results/experiment_5/exp5_confusion_multiclass_long.csv --output-source-breakdown-csv results/experiment_5/exp5_source_breakdown.csv --output-qualitative-csv results/experiment_5/exp5_qualitative_examples.csv
```

## Key outputs

- `results/experiment_5/exp5_method_compare_binary.csv`
- `results/experiment_5/exp5_method_compare_multiclass.csv`
- `results/experiment_5/exp5_per_class_multiclass.csv`
- `results/experiment_5/exp5_confusion_binary_counts.csv`
- `results/experiment_5/exp5_confusion_multiclass_long.csv`
- `results/experiment_5/exp5_source_breakdown.csv`
- `results/experiment_5/exp5_qualitative_examples.csv`
