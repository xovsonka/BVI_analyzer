Experiment 1 Scripts (Phishing Simulation)

This folder contains scripts for creating controlled email scenarios for Experiment 1.

Workflow

1) Generate draft templates (LLM-ready, editable TXT):

python experiments/experiment_1/generate_llm_drafts.py --out-dir results/experiment_1/llm_drafts

2) Mutate templates (URL/header/text variants):

python experiments/experiment_1/mutate_email_drafts.py --input-csv results/experiment_1/llm_drafts/templates.csv --out-dir results/experiment_1/mutated --variants-per-template 4

3) Build realistic edits from Enron:

python experiments/experiment_1/build_enron_real_edits.py --enron-csv dataset/source/enron/emails.csv --out-dir results/experiment_1/enron_edits --samples 30

Outputs

- TXT files for manual review/editing
- CSV files with metadata and ready-to-use seed columns:
  - id
  - text
  - label (0=legit, 1=malicious)
  - source

Notes

- Keep manual review step before seeding GoPhish campaigns.
- If you want to merge mutated + enron-edited CSV files, concatenate by columns.
