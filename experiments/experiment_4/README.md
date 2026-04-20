# Experiment 4 - ML Model Evaluation

Experiment 4 evaluates ML classification performance, separated into two branches:

- Track A: offline benchmark on shared splits (`val_iid`, `test_iid`, `test_hard_source`, `test_hard_cluster`, `test_deployment`)
- Track B: realistic campaign benchmark (ML-only performance on campaign data)

Primary goal: evaluate ML model robustness and generalization, not heuristic rules.

## Recommended model set

- Selection shortlist is now ranked on `val_iid` only (default `f1_macro`) to avoid selecting directly on final test splits.
- Current top offline candidate in the committed reports is `scenario_m_rebalanced_fraud_spam + hybrid_sgd_log + text_plus_features`.
- Recommended references to keep in the final comparison table:
  - `scenario_m_rebalanced_fraud_spam + hybrid_logreg + text_plus_features`
  - `scenario_m_anti_template_feature_regularized + hybrid_logreg + text_plus_features`
  - `scenario_l_combined_anti_source + hybrid_sgd_log + text_plus_features`

## Track A - Offline benchmark

Run split-suite evaluation per scenario:

```bash
python evaluation/eval_split_suite.py --processed-dir dataset/processed --scenarios scenario_l_combined_anti_source --models hybrid_logreg hybrid_sgd_log --input-mode text_plus_features --output-csv results/experiment_4/exp4_offline_split_suite_l.csv --output-summary-csv results/experiment_4/exp4_offline_split_suite_l_summary.csv --output-json results/experiment_4/exp4_offline_split_suite_l.json

python evaluation/eval_split_suite.py --processed-dir dataset/processed --scenarios scenario_m_anti_template_feature_regularized --models hybrid_logreg hybrid_sgd_log --input-mode text_plus_features --output-csv results/experiment_4/exp4_offline_split_suite_manti.csv --output-summary-csv results/experiment_4/exp4_offline_split_suite_manti_summary.csv --output-json results/experiment_4/exp4_offline_split_suite_manti.json

python evaluation/eval_split_suite.py --processed-dir dataset/processed --scenarios scenario_m_rebalanced_fraud_spam --models hybrid_logreg hybrid_sgd_log --input-mode text_plus_features --output-csv results/experiment_4/exp4_offline_split_suite_mreb.csv --output-summary-csv results/experiment_4/exp4_offline_split_suite_mreb_summary.csv --output-json results/experiment_4/exp4_offline_split_suite_mreb.json
```

## Track B - Campaign benchmark (ML-only)

Use campaign ML summaries produced by prior experiments (Exp2/Exp3 Track B):

- preferred paths:
  - `results/experiment_2_results/experiment_2_campaign_ml_summary.csv`
  - `results/experiment_3_results/experiment_3_campaign_ml_summary.csv`
- backward-compatible fallback is handled automatically for legacy folders:
  - `results/eperiment_2_results/...`
  - `results/experiment_3_result/...`

If needed, regenerate predictions first:

```bash
python experiments/experiment_2/predict_campaign_with_final_model.py --analyzed-csv results/experiment_3_result/experiment_3_campaign_model_input.csv --seed-csv results/experiment_3/gophish_seed_input_url_focused_50_50.csv --metadata-csv dataset/processed/mailhog_messages_exp3_campaign_url.csv --max-train-rows 5000 --output-pred-csv results/experiment_3_result/experiment_3_campaign_ml_predictions.csv --output-summary-csv results/experiment_3_result/experiment_3_campaign_ml_summary.csv
```

## Build Exp4 report tables

```bash
python experiments/experiment_4/build_experiment4_report_tables.py --offline-csvs results/experiment_4/exp4_offline_split_suite_l.csv results/experiment_4/exp4_offline_split_suite_manti.csv results/experiment_4/exp4_offline_split_suite_mreb.csv --details-dir results/experiment_4/split_suite_details --campaign-summary-csvs results/experiment_2_results/experiment_2_campaign_ml_summary.csv results/experiment_3_results/experiment_3_campaign_ml_summary.csv --campaign-names exp2_track_b exp3_track_b --selection-split val_iid --selection-metric f1_macro --bootstrap 200 --allow-multirow-campaign-summary --output-dir results/experiment_4
```

Main outputs:

- `results/experiment_4/exp4_offline_primary_table.csv`
- `results/experiment_4/exp4_offline_per_class_table.csv`
- `results/experiment_4/exp4_offline_confusion_long.csv`
- `results/experiment_4/exp4_offline_model_rank.csv`
- `results/experiment_4/exp4_offline_ci_table.csv`
- `results/experiment_4/exp4_final_candidates_table.csv`
- `results/experiment_4/exp4_final_confusion_matrix.csv`
- `results/experiment_4/exp4_campaign_ml_table.csv`
- `results/experiment_4/exp4_selection_metadata.json`

## Notes for thesis text

- Track A = primary scientific benchmark for multiclass robustness.
- Track B = supplementary realistic validation after campaign transport.
- `build_experiment4_report_tables.py` now ranks on `val_iid` and uses hard/deployment splits only as reference reporting columns.
- If `--final-scenario/--final-model/--final-input-mode` stay as `auto`, the final confusion export follows the top-ranked offline candidate.
- Keep heuristic-vs-ML-vs-hybrid comparison for Experiment 5; Exp4 focuses on ML quality.
