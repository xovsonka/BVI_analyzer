# Experiment 6 - Comparison with PhishTool

Experiment 6 is designed as a capability comparison, not a pure classifier shootout.

It has two branches:

- **Track A (IOC / analytical comparison)**: overlap of extracted indicators
- **Track B (decision-level comparison)**: binary verdict comparison if PhishTool verdicts are available

## 1) Build representative email set + PhishTool template

```bash
python experiments/experiment_6/build_exp6_dataset.py --n-legit 15 --n-phishing 15 --n-spam 10 --n-financial-fraud 10 --n-header-heavy 10 --n-url-heavy 10 --copy-eml --output-selection-csv results/experiment_6/exp6_selection.csv --output-summary-csv results/experiment_6/exp6_selection_summary.csv --output-phishtool-template-csv results/experiment_6/exp6_phishtool_template.csv --output-eml-dir dataset/experiment_6/eml_selected
```

This default mix produces 70 emails (15+15+10+10+10+10). Adjust `--n-*` values if you want exactly 60.

Community-plan profile (25 emails):

```bash
python experiments/experiment_6/build_exp6_dataset.py --n-legit 5 --n-phishing 5 --n-spam 4 --n-financial-fraud 4 --n-header-heavy 4 --n-url-heavy 3 --copy-eml --clean-output --output-selection-csv results/experiment_6/exp6_selection.csv --output-summary-csv results/experiment_6/exp6_selection_summary.csv --output-phishtool-template-csv results/experiment_6/exp6_phishtool_template.csv --output-eml-dir dataset/experiment_6/eml_selected
```

Outputs:

- `results/experiment_6/exp6_selection.csv`
- `results/experiment_6/exp6_selection_summary.csv`
- `results/experiment_6/exp6_phishtool_template.csv`
- selected EMLs in `dataset/experiment_6/eml_selected/`

## 2) Build our-tool IOC table for the same set

```bash
python experiments/experiment_6/build_exp6_our_ioc_table.py --selection-csv results/experiment_6/exp6_selection.csv --output-csv results/experiment_6/exp6_our_ioc_table.csv
```

Output:

- `results/experiment_6/exp6_our_ioc_table.csv`

## 3) Run PhishTool manually and fill template

Take emails from `dataset/experiment_6/eml_selected/` and run them through PhishTool.

Fill these columns in `results/experiment_6/exp6_phishtool_template.csv`:

- `phishtool_spf_fail`
- `phishtool_dkim_fail`
- `phishtool_dmarc_fail`
- `phishtool_from_reply_mismatch`
- `phishtool_from_return_path_mismatch`
- `phishtool_message_id_domain_mismatch`
- `phishtool_received_anomaly`
- `phishtool_display_name_spoof`
- `phishtool_ip_url`
- `phishtool_shortener_url`
- `phishtool_anchor_mismatch`
- `phishtool_suspicious_tld`
- `phishtool_brand_typosquat`
- `phishtool_obfuscated_url`
- `phishtool_risky_attachment`

Optional:

- `phishtool_pred_binary` (`legit` / `suspicious`)
- `analyst_notes`

Save filled file as:

- `results/experiment_6/exp6_phishtool_filled.csv`

## 4) Evaluate overlap and decision comparison

```bash
python experiments/experiment_6/eval_exp6_phishtool_compare.py --our-ioc-csv results/experiment_6/exp6_our_ioc_table.csv --phishtool-csv results/experiment_6/exp6_phishtool_filled.csv --output-overlap-long-csv results/experiment_6/exp6_ioc_overlap_long.csv --output-overlap-summary-csv results/experiment_6/exp6_ioc_overlap_summary.csv --output-capability-summary-csv results/experiment_6/exp6_capability_summary.csv --output-decision-compare-csv results/experiment_6/exp6_decision_compare.csv --output-decision-summary-csv results/experiment_6/exp6_decision_summary.csv --output-case-study-csv results/experiment_6/exp6_case_studies.csv
```

Key outputs:

- `results/experiment_6/exp6_ioc_overlap_summary.csv`
- `results/experiment_6/exp6_capability_summary.csv`
- `results/experiment_6/exp6_decision_summary.csv`
- `results/experiment_6/exp6_case_studies.csv`

## Notes

- If `phishtool_pred_binary` is not available, Track B remains descriptive (agreement cannot be computed).
- This experiment is intended as reference capability comparison and explainability audit.
