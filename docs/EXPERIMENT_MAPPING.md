# Experiment Mapping (Current)

## Experiment 1 - Campaign simulation and collection
- Scripts:
  - `data_tools/seed_gophish.py`
  - `data_tools/collect_mailhog.py`
- Outputs:
  - `dataset/campaign_eml/*.eml`
  - `dataset/processed/gophish_seed_map.csv`

## Experiment 2 - Header/URL/heuristic analysis
- Scripts:
  - `analysis/analyze_campaign.py`
  - `analysis/heuristic_stats_report.py`
  - `analysis/eml_model_input_overview.py`
- Outputs:
  - `results/eml_model_input.csv`
  - `results/eml_parts.jsonl`
  - `results/heuristic_stats.json`
  - `results/heuristic_stats.txt`

## Experiment 3 - URL indicator detection (Track A + Track B)
- Scripts:
  - `experiments/experiment_3/generate_url_indicator_dataset.py`
  - `evaluation/eval_experiment3_url_indicators.py`
  - `experiments/experiment_3/build_campaign_url_focused_dataset.py`
  - `experiments/experiment_3/report_url_campaign_results.py`
- Main outputs:
  - `results/experiment_3_result/experiment_3_url_indicator_summary.csv`
  - `results/experiment_3_result/experiment_3_url_explainability_summary.csv`
  - `results/experiment_3_result/experiment_3_campaign_method_compare.csv`
  - `results/experiment_3_result/experiment_3_campaign_injected_indicator_retention.csv`

## Experiment 4 - ML model evaluation (offline + realistic campaign)
- Scripts:
  - `evaluation/eval_split_suite.py`
  - `experiments/experiment_4/build_experiment4_report_tables.py`
  - `analysis/plot_exp4_figures.py`
- Main outputs:
  - `results/experiment_4/exp4_offline_primary_table.csv`
  - `results/experiment_4/exp4_offline_per_class_table.csv`
  - `results/experiment_4/exp4_offline_confusion_long.csv`
  - `results/experiment_4/exp4_campaign_ml_table.csv`
  - `results/graphs_thesis_final/exp4_trackA_deployment_macro_f1.pdf`
  - `results/graphs_thesis_final/exp4_trackB_campaign_binary_f1.pdf`

## Experiment 5 - Heuristic vs ML vs hybrid comparison
- Scripts:
  - `training/train_two_stage.py`
  - `evaluation/eval_text_view_modes.py`
  - `analysis/plot_thesis_graphs.py`
- Outputs:
  - `results/two_stage_text_only/*.json`
  - `results/two_stage_text_only/summary.csv`
  - `results/text_view_modes_*.csv`
  - `results/graphs_thesis/*.png`

## Experiment 6 - Comparison with PhishTool
- Scripts:
  - `experiments/experiment_6/build_exp6_dataset.py`
  - `experiments/experiment_6/build_exp6_our_ioc_table.py`
  - `experiments/experiment_6/eval_exp6_phishtool_compare.py`
- Outputs:
  - `results/experiment_6/exp6_selection.csv`
  - `results/experiment_6/exp6_phishtool_template.csv`
  - `results/experiment_6/exp6_our_ioc_table.csv`
  - `results/experiment_6/exp6_ioc_overlap_summary.csv`
  - `results/experiment_6/exp6_capability_summary.csv`
  - `results/experiment_6/exp6_decision_summary.csv`

## Notes
- Primary decision flow is now `evaluation/eval_split_suite.py` + `evaluation/select_robust_model.py --selection-mode final_rule`.
- Legacy scripts are preserved in `archive/`.
