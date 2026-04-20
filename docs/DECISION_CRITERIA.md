# Decision Criteria (Model + Scenario)

Tento dokument definuje pravidlá výberu finálnej konfigurácie pre prácu.

## Primárne kritérium
- `mean source-holdout f1_macro` (čím vyššie, tým lepšie)

## Sekundárne kritérium
- `worst-source f1_macro` (čím vyššie, tým lepšie)

## Terciárne kritérium
- `shared-test f1_macro` (čím vyššie, tým lepšie)

## Doplňujúce podmienky
- `single_stage` má prednosť pred `two_stage`, pokiaľ `two_stage` neprinesie jasné zlepšenie robustnosti na source holdout.
- Konfigurácia musí mať stabilný výkon (`std_holdout_f1_macro` nízke; nepoužiť model s veľkou variabilitou medzi source).
- Pri zdrojoch s jednou triedou (`specialized_source`) sa reportuje aj binary metrika (`benign vs suspicious`).

## Praktický postup
1. Model selection robiť iba na `val_iid` (zdieľané scenárové `val.csv`).
2. Headline benchmark reportovať na `test_iid` (zdieľané scenárové `test.csv`).
3. Robustness gate reportovať na `test_hard_source` a `test_hard_cluster`.
4. Deployment decision robiť na `test_deployment` (prirodzený prior, bez train-only augmentácií).
5. Až po shortlist-e spustiť finálny compare (`evaluation/select_robust_model.py --selection-mode hierarchy`) + GoPhish/MailHog validáciu.

## Split Freeze Rule
- Používaj zamrazený split z `dataset/processed/shared/split_manifest.json`.
- Pri zmene split logiky zvýš `--split-manifest-version` a benchmark považuj za novú generáciu.
