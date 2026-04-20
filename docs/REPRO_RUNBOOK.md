# Repro Runbook (Current Pipeline)

## 1) Build datasets and scenarios

### Full build (A-M, without LLM F)
```bash
python experiments/prepare_dataset_multiclass.py --target-per-class 30000 --oversample-target 30000 --source-balance-target 20000 --g-source-cap-multiplier 1.5 --g-min-source-rows 100 --i-source-cap-multiplier 1.1 --j-max-source-share 0.25 --k-drop-sources "nazario,nigerian_fraud,enron_raw" --l-hard-legit-ratio 0.05 --m-max-per-template 3 --m-max-per-template-source 2 --m-feature-dropout-prob 0.30 --m-hard-legit-ratio 0.05 --m-synth-ratio-legit 0.20 --m-synth-ratio-phishing 0.30 --m-synth-ratio-spam 0.30 --m-synth-ratio-financial-fraud 0.40 --hard-legit-ratio 0.15 --train-ratio 0.75 --val-ratio 0.10 --iid-test-ratio 0.10 --deployment-test-ratio 0.05 --hard-source-ratio 0.02 --hard-cluster-ratio 0.02 --min-val-per-label 200 --min-test-per-label 300 --min-deployment-per-label 200 --min-hard-per-label 120 --min-train-per-label 1000 --max-label-prior-deviation 0.20 --split-manifest-version v2_grouped_multi_eval_r5
```

### Build only scenario L
```bash
python experiments/prepare_dataset_multiclass.py --target-per-class 30000 --oversample-target 30000 --source-balance-target 20000 --g-source-cap-multiplier 1.5 --g-min-source-rows 100 --i-source-cap-multiplier 1.1 --j-max-source-share 0.25 --k-drop-sources "nazario,nigerian_fraud,enron_raw" --l-hard-legit-ratio 0.05 --hard-legit-ratio 0.15 --train-ratio 0.75 --val-ratio 0.10 --iid-test-ratio 0.10 --deployment-test-ratio 0.05 --hard-source-ratio 0.02 --hard-cluster-ratio 0.02 --min-val-per-label 200 --min-test-per-label 300 --min-deployment-per-label 200 --min-hard-per-label 120 --min-train-per-label 1000 --max-label-prior-deviation 0.20 --split-manifest-version v2_grouped_multi_eval_r5 --only-scenarios l
```

### Build only scenario M
```bash
python experiments/prepare_dataset_multiclass.py --target-per-class 30000 --oversample-target 30000 --source-balance-target 20000 --g-source-cap-multiplier 1.5 --g-min-source-rows 100 --i-source-cap-multiplier 1.1 --j-max-source-share 0.25 --k-drop-sources "nazario,nigerian_fraud,enron_raw" --m-max-per-template 3 --m-max-per-template-source 2 --m-feature-dropout-prob 0.30 --m-hard-legit-ratio 0.05 --m-synth-ratio-legit 0.20 --m-synth-ratio-phishing 0.30 --m-synth-ratio-spam 0.30 --m-synth-ratio-financial-fraud 0.40 --hard-legit-ratio 0.15 --train-ratio 0.75 --val-ratio 0.10 --iid-test-ratio 0.10 --deployment-test-ratio 0.05 --hard-source-ratio 0.02 --hard-cluster-ratio 0.02 --min-val-per-label 200 --min-test-per-label 300 --min-deployment-per-label 200 --min-hard-per-label 120 --min-train-per-label 1000 --max-label-prior-deviation 0.20 --split-manifest-version v2_grouped_multi_eval_r5 --only-scenarios m
```

## 2) Baseline leakage checks
```bash
python evaluation/source_only_baseline.py --processed-dir dataset/processed --scenario scenario_l_combined_anti_source --mode source_only
python evaluation/source_token_leak_scanner.py --processed-dir dataset/processed --scenario scenario_l_combined_anti_source
python evaluation/split_dup_matrix.py --processed-dir dataset/processed --mode shared --max-rows-per-split 30000 --near-thresholds 0.85 0.90 0.95
```

## 3) Main model evaluation (text-only)
```bash
python evaluation/eval_split_suite.py --processed-dir dataset/processed --scenarios scenario_i_strict_source_cap scenario_k_low_leak_sources scenario_l_combined_anti_source --models hybrid_logreg hybrid_linear_svc_cal hybrid_sgd_log hybrid_sgd_hinge --input-mode text_only --output-csv results/split_suite_text_only_ikl.csv --output-summary-csv results/split_suite_text_only_ikl_summary.csv --output-json results/split_suite_text_only_ikl.json
```

Per-model details are stored in:
- `results/split_suite_details/*.json`

## 4) Architecture benchmark (optional report bundle)
```bash
python experiments/benchmark_all_architectures.py --processed-dir dataset/processed --scenarios scenario_i_strict_source_cap scenario_k_low_leak_sources scenario_l_combined_anti_source --models hybrid_logreg hybrid_linear_svc_cal hybrid_sgd_log hybrid_sgd_hinge --skip-two-stage --input-mode text_only
```

## 5) Final model selection
```bash
python evaluation/select_robust_model.py --selection-mode final_rule --split-suite-summary-csv results/split_suite_text_only_ikl_summary.csv --architecture single_stage --output-csv results/robust_model_selection_text_only_ikl.csv --output-json results/robust_model_selection_text_only_ikl.json
```

## 6) Two-stage comparison (text-only)
```bash
python training/train_two_stage.py --scenario scenario_l_combined_anti_source --processed-dir dataset/processed --stage1-model hybrid_sgd_hinge --stage2-model hybrid_sgd_hinge --input-mode text_only --benign-labels legit --benign-output-label legit --results-dir results/two_stage_text_only
```

## 7) Body vs header+body vs full-text
```bash
python evaluation/eval_text_view_modes.py --processed-dir dataset/processed --scenario scenario_l_combined_anti_source --views body_only header_plus_body full_text --output-csv results/text_view_modes_l.csv --output-json results/text_view_modes_l.json
```

## 8) CI + thesis graphs
```bash
python evaluation/report_with_bootstrap_ci.py --processed-dir dataset/processed --scenario scenario_l_combined_anti_source --model hybrid_sgd_hinge --bootstrap 400 --output-json results/bootstrap_ci_l.json --output-csv results/bootstrap_ci_l.csv
python analysis/plot_thesis_graphs.py --split-summary-csv results/split_suite_text_only_ikl_summary.csv --text-view-csv results/text_view_modes_l.csv --two-stage-csv results/two_stage_text_only/summary.csv --out-dir results/graphs_thesis
```

## Archive note
Legacy/diagnostic scripts moved to `archive/`.
