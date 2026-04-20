from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PRIMARY_SPLITS = [
    "val_iid",
    "test_iid",
    "test_hard_source",
    "test_hard_cluster",
    "test_deployment",
]

DEFAULT_FINAL_CANDIDATES = [
    ("scenario_m_rebalanced_fraud_spam", "hybrid_sgd_log", "text_plus_features"),
    ("scenario_m_rebalanced_fraud_spam", "hybrid_logreg", "text_plus_features"),
    ("scenario_l_combined_anti_source", "hybrid_sgd_log", "text_plus_features"),
    (
        "scenario_m_anti_template_feature_regularized",
        "hybrid_logreg",
        "text_plus_features",
    ),
]


def _resolve_known_path_alias(path: Path) -> Path:
    mapping = {
        "results/experiment_2_results/experiment_2_campaign_ml_summary.csv": "results/eperiment_2_results/experiment_2_campaign_ml_summary.csv",
        "results/experiment_3_results/experiment_3_campaign_ml_summary.csv": "results/experiment_3_result/experiment_3_campaign_ml_summary.csv",
    }
    key = str(path).replace("\\", "/")
    if path.exists() or key not in mapping:
        return path
    alt = Path(mapping[key])
    if alt.exists():
        print(f"[WARN] Using fallback path for missing input: {alt}")
        return alt
    return path


def _load_offline_rows(csv_paths: list[Path], fail_on_duplicates: bool) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        resolved = _resolve_known_path_alias(path)
        if resolved.exists():
            frames.append(pd.read_csv(resolved))
        else:
            print(f"[WARN] Missing offline CSV skipped: {path}")
    if not frames:
        raise FileNotFoundError("No offline split-suite CSV files found")
    out = pd.concat(frames, ignore_index=True)

    dup_mask = out.duplicated(
        subset=["scenario", "model", "input_mode", "split"], keep=False
    )
    if dup_mask.any():
        dup_count = int(dup_mask.sum())
        msg = (
            f"Found {dup_count} duplicate offline rows by "
            "(scenario, model, input_mode, split)"
        )
        if fail_on_duplicates:
            raise ValueError(msg)
        print(f"[WARN] {msg}; keeping last occurrence")

    out = out.drop_duplicates(
        subset=["scenario", "model", "input_mode", "split"], keep="last"
    )
    return out


def _parse_detail_jsons(details_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_class_rows: list[dict] = []
    confusion_rows: list[dict] = []

    for detail_path in sorted(details_dir.glob("*.json")):
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        scenario = str(detail.get("scenario", ""))
        model = str(detail.get("model", ""))
        input_mode = str(detail.get("input_mode", ""))

        splits = detail.get("splits", {})
        for split, split_data in splits.items():
            report = split_data.get("classification_report", {})
            for label, stats in report.items():
                if label in {"accuracy", "macro avg", "weighted avg"}:
                    continue
                if not isinstance(stats, dict):
                    continue
                per_class_rows.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "input_mode": input_mode,
                        "split": split,
                        "label": str(label),
                        "precision": float(stats.get("precision", 0.0)),
                        "recall": float(stats.get("recall", 0.0)),
                        "f1": float(stats.get("f1-score", 0.0)),
                        "support": int(stats.get("support", 0)),
                    }
                )

            labels = [str(x) for x in split_data.get("labels", [])]
            matrix = split_data.get("confusion_matrix", [])
            if labels and matrix:
                for i, true_label in enumerate(labels):
                    for j, pred_label in enumerate(labels):
                        count = int(matrix[i][j])
                        confusion_rows.append(
                            {
                                "scenario": scenario,
                                "model": model,
                                "input_mode": input_mode,
                                "split": split,
                                "true_label": true_label,
                                "pred_label": pred_label,
                                "count": count,
                            }
                        )

    return pd.DataFrame(per_class_rows), pd.DataFrame(confusion_rows)


def _load_campaign_tables(
    paths: list[Path], names: list[str], allow_multirow_summary: bool
) -> pd.DataFrame:
    rows = []
    missing_paths = []
    for idx, path in enumerate(paths):
        resolved = _resolve_known_path_alias(path)
        if not resolved.exists():
            missing_paths.append(str(path))
            continue
        name = names[idx] if idx < len(names) else f"campaign_{idx + 1}"
        df = pd.read_csv(resolved)
        if df.empty:
            print(f"[WARN] Empty campaign summary skipped: {resolved}")
            continue
        if len(df) != 1 and not allow_multirow_summary:
            raise ValueError(
                f"Expected exactly 1 row in campaign summary {resolved}, got {len(df)}"
            )
        if len(df) != 1:
            print(
                f"[WARN] Campaign summary has {len(df)} rows in {resolved}; "
                "preserving all rows"
            )
        for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
            row_dict = row.to_dict()
            if (
                "campaign_benchmark" not in row_dict
                or not str(row_dict.get("campaign_benchmark", "")).strip()
            ):
                row_dict["campaign_benchmark"] = name
            row_dict["campaign_row"] = row_idx
            row_dict["source_csv"] = str(resolved)
            rows.append(row_dict)
    if missing_paths:
        print("[WARN] Missing campaign summary CSVs:")
        for mp in missing_paths:
            print(f"  - {mp}")
    if not rows:
        print("[WARN] No campaign benchmark rows were loaded")
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols: list[str] = []
    for col in df.columns:
        if isinstance(col, tuple):
            left = str(col[0]).strip()
            right = str(col[1]).strip()
            cols.append(f"{left}_{right}" if right else left)
        else:
            cols.append(str(col))
    out = df.copy()
    out.columns = cols
    return out


def _parse_candidate_specs(raw_items: list[str]) -> list[tuple[str, str, str]]:
    out = []
    for item in raw_items:
        parts = [p.strip() for p in item.split("|")]
        if len(parts) != 3:
            raise ValueError(
                f"Final candidate spec must be 'scenario|model|input_mode', got: {item}"
            )
        out.append((parts[0], parts[1], parts[2]))
    return out


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(float(metric_fn(y_true[idx], y_pred[idx])))
    vals_arr = np.asarray(vals, dtype=float)
    return (
        float(metric_fn(y_true, y_pred)),
        float(np.quantile(vals_arr, 0.025)),
        float(np.quantile(vals_arr, 0.975)),
    )


def _build_ci_table(confusion_df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    if confusion_df.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["scenario", "model", "input_mode", "split"]
    for idx, (keys, group) in enumerate(
        confusion_df.groupby(group_cols, sort=False, observed=False)
    ):
        y_true: list[str] = []
        y_pred: list[str] = []
        for row in group.itertuples(index=False):
            count = int(row.count)
            if count <= 0:
                continue
            y_true.extend([str(row.true_label)] * count)
            y_pred.extend([str(row.pred_label)] * count)

        y_true_arr = np.asarray(y_true, dtype=object)
        y_pred_arr = np.asarray(y_pred, dtype=object)
        acc, acc_lo, acc_hi = _bootstrap_ci(
            y_true_arr,
            y_pred_arr,
            metric_fn=lambda a, b: accuracy_score(a, b),
            n_boot=n_boot,
            seed=seed + idx,
        )
        bal, bal_lo, bal_hi = _bootstrap_ci(
            y_true_arr,
            y_pred_arr,
            metric_fn=lambda a, b: balanced_accuracy_score(a, b),
            n_boot=n_boot,
            seed=seed + 1000 + idx,
        )
        f1m, f1m_lo, f1m_hi = _bootstrap_ci(
            y_true_arr,
            y_pred_arr,
            metric_fn=lambda a, b: f1_score(a, b, average="macro", zero_division=0),
            n_boot=n_boot,
            seed=seed + 2000 + idx,
        )
        rows.append(
            {
                "scenario": keys[0],
                "model": keys[1],
                "input_mode": keys[2],
                "split": keys[3],
                "rows": int(len(y_true_arr)),
                "accuracy": acc,
                "accuracy_ci_low": acc_lo,
                "accuracy_ci_high": acc_hi,
                "balanced_accuracy": bal,
                "balanced_accuracy_ci_low": bal_lo,
                "balanced_accuracy_ci_high": bal_hi,
                "f1_macro": f1m,
                "f1_macro_ci_low": f1m_lo,
                "f1_macro_ci_high": f1m_hi,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Experiment 4 offline + campaign report tables"
    )
    parser.add_argument(
        "--offline-csvs",
        nargs="+",
        default=[
            "results/experiment_4/exp4_offline_split_suite_l.csv",
            "results/experiment_4/exp4_offline_split_suite_manti.csv",
            "results/experiment_4/exp4_offline_split_suite_mreb.csv",
        ],
    )
    parser.add_argument(
        "--details-dir",
        default="results/experiment_4/split_suite_details",
    )
    parser.add_argument(
        "--campaign-summary-csvs",
        nargs="+",
        default=[
            "results/experiment_2_results/experiment_2_campaign_ml_summary.csv",
            "results/experiment_3_results/experiment_3_campaign_ml_summary.csv",
        ],
    )
    parser.add_argument(
        "--campaign-names",
        nargs="+",
        default=["exp2_track_b", "exp3_track_b"],
    )
    parser.add_argument(
        "--allow-multirow-campaign-summary",
        action="store_true",
        help="Allow campaign summary CSVs with multiple rows (all rows are preserved)",
    )
    parser.add_argument(
        "--fail-on-duplicates",
        action="store_true",
        help="Fail if offline CSVs contain duplicate scenario/model/input/split rows",
    )
    parser.add_argument(
        "--final-scenario",
        default="auto",
    )
    parser.add_argument("--final-model", default="auto")
    parser.add_argument("--final-input-mode", default="auto")
    parser.add_argument("--final-split", default="test_deployment")
    parser.add_argument(
        "--selection-split",
        default="val_iid",
        choices=PRIMARY_SPLITS,
        help="Split used for model ranking to avoid selecting on final test splits.",
    )
    parser.add_argument(
        "--selection-metric",
        default="f1_macro",
        choices=["f1_macro", "balanced_accuracy", "accuracy"],
        help="Primary metric used on --selection-split for ranking.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Bootstrap replicates for offline confidence intervals built from confusion counts.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--final-candidates",
        nargs="+",
        default=["|".join(x) for x in DEFAULT_FINAL_CANDIDATES],
        help="Candidate rows as scenario|model|input_mode",
    )
    parser.add_argument("--output-dir", default="results/experiment_4")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    offline_csv_paths = [Path(p) for p in args.offline_csvs]
    offline = _load_offline_rows(
        offline_csv_paths, fail_on_duplicates=args.fail_on_duplicates
    )
    offline = offline[offline["split"].isin(PRIMARY_SPLITS)].copy()
    offline["split"] = pd.Categorical(
        offline["split"], categories=PRIMARY_SPLITS, ordered=True
    )
    offline = offline.sort_values(["scenario", "model", "split"]).reset_index(drop=True)
    offline.to_csv(output_dir / "exp4_offline_combined.csv", index=False)

    offline_primary = offline[
        [
            "scenario",
            "model",
            "input_mode",
            "split",
            "rows",
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "f1_weighted",
        ]
    ].copy()
    offline_primary.to_csv(output_dir / "exp4_offline_primary_table.csv", index=False)

    details_dir = Path(args.details_dir)
    per_class_df, confusion_df = _parse_detail_jsons(details_dir)
    if not per_class_df.empty:
        per_class_df = per_class_df[per_class_df["split"].isin(PRIMARY_SPLITS)].copy()
        per_class_df["split"] = pd.Categorical(
            per_class_df["split"], categories=PRIMARY_SPLITS, ordered=True
        )
        per_class_df = per_class_df.sort_values(
            ["scenario", "model", "split", "label"]
        ).reset_index(drop=True)
    per_class_df.to_csv(output_dir / "exp4_offline_per_class_table.csv", index=False)

    if not confusion_df.empty:
        confusion_df = confusion_df[confusion_df["split"].isin(PRIMARY_SPLITS)].copy()
        confusion_df["split"] = pd.Categorical(
            confusion_df["split"], categories=PRIMARY_SPLITS, ordered=True
        )
        confusion_df = confusion_df.sort_values(
            ["scenario", "model", "split", "true_label", "pred_label"]
        ).reset_index(drop=True)
    confusion_df.to_csv(output_dir / "exp4_offline_confusion_long.csv", index=False)

    ci_df = _build_ci_table(
        confusion_df,
        n_boot=args.bootstrap,
        seed=args.bootstrap_seed,
    )
    if not ci_df.empty:
        ci_df["split"] = pd.Categorical(
            ci_df["split"], categories=PRIMARY_SPLITS, ordered=True
        )
        ci_df = ci_df.sort_values(["scenario", "model", "split"]).reset_index(drop=True)
    ci_df.to_csv(output_dir / "exp4_offline_ci_table.csv", index=False)

    model_rank = (
        offline_primary.pivot_table(
            index=["scenario", "model", "input_mode"],
            columns="split",
            values=["f1_macro", "balanced_accuracy", "accuracy"],
            aggfunc="first",
            observed=False,
        )
        .reset_index()
        .copy()
    )
    model_rank = _flatten_columns(model_rank)

    selection_col = f"{args.selection_metric}_{args.selection_split}"
    tie_bal_col = f"balanced_accuracy_{args.selection_split}"
    tie_acc_col = f"accuracy_{args.selection_split}"
    reference_f1_cols = [
        "f1_macro_test_hard_source",
        "f1_macro_test_hard_cluster",
        "f1_macro_test_deployment",
    ]
    for c in [selection_col, tie_bal_col, tie_acc_col, *reference_f1_cols]:
        if c not in model_rank.columns:
            model_rank[c] = 0.0

    if (
        "f1_macro_test_deployment" in model_rank.columns
        and "f1_macro_test_hard_source" in model_rank.columns
    ):
        model_rank["robustness_gap_deploy_minus_hard_source"] = (
            model_rank["f1_macro_test_deployment"]
            - model_rank["f1_macro_test_hard_source"]
        )
    model_rank["reference_hard_split_mean_f1"] = model_rank[
        ["f1_macro_test_hard_source", "f1_macro_test_hard_cluster"]
    ].mean(axis=1)
    model_rank["reference_eval_mean_f1"] = model_rank[reference_f1_cols].mean(axis=1)
    model_rank["selection_split"] = args.selection_split
    model_rank["selection_metric"] = args.selection_metric
    model_rank["selection_score"] = model_rank[selection_col]
    model_rank["primary_exp4_score"] = model_rank["selection_score"]
    model_rank = model_rank.sort_values(
        ["selection_score", tie_bal_col, tie_acc_col],
        ascending=False,
    ).reset_index(drop=True)

    if not ci_df.empty:
        ci_rank = ci_df.pivot_table(
            index=["scenario", "model", "input_mode"],
            columns="split",
            values=["f1_macro_ci_low", "f1_macro_ci_high"],
            aggfunc="first",
            observed=False,
        ).reset_index()
        ci_rank = _flatten_columns(ci_rank)
        model_rank = model_rank.merge(
            ci_rank,
            on=["scenario", "model", "input_mode"],
            how="left",
            validate="one_to_one",
        )

    model_rank.to_csv(output_dir / "exp4_offline_model_rank.csv", index=False)

    final_candidates = _parse_candidate_specs(args.final_candidates)
    candidates_df = pd.DataFrame(
        final_candidates, columns=["scenario", "model", "input_mode"]
    )
    final_candidates_table = candidates_df.merge(
        model_rank,
        on=["scenario", "model", "input_mode"],
        how="left",
        validate="one_to_one",
    )
    final_candidates_table.to_csv(
        output_dir / "exp4_final_candidates_table.csv", index=False
    )

    if model_rank.empty:
        raise RuntimeError("Model rank table is empty; cannot select final model")

    if (
        str(args.final_scenario).lower() == "auto"
        or str(args.final_model).lower() == "auto"
        or str(args.final_input_mode).lower() == "auto"
    ):
        top_row = model_rank.iloc[0]
        final_scenario = str(top_row["scenario"])
        final_model = str(top_row["model"])
        final_input_mode = str(top_row["input_mode"])
    else:
        final_scenario = args.final_scenario
        final_model = args.final_model
        final_input_mode = args.final_input_mode

    selection_meta = {
        "selection_split": args.selection_split,
        "selection_metric": args.selection_metric,
        "selection_col": selection_col,
        "final_scenario": final_scenario,
        "final_model": final_model,
        "final_input_mode": final_input_mode,
        "final_split": args.final_split,
        "bootstrap": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
    }
    (output_dir / "exp4_selection_metadata.json").write_text(
        json.dumps(selection_meta, indent=2), encoding="utf-8"
    )

    final_conf_long = confusion_df[
        (confusion_df["scenario"] == final_scenario)
        & (confusion_df["model"] == final_model)
        & (confusion_df["input_mode"] == final_input_mode)
        & (confusion_df["split"] == args.final_split)
    ].copy()
    final_conf_long.to_csv(output_dir / "exp4_final_confusion_long.csv", index=False)
    if not final_conf_long.empty:
        final_conf_matrix = final_conf_long.pivot_table(
            index="true_label",
            columns="pred_label",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        final_conf_matrix.to_csv(output_dir / "exp4_final_confusion_matrix.csv")
    else:
        print(
            "[WARN] Final confusion selection returned no rows; "
            "check --final-scenario/--final-model/--final-input-mode/--final-split"
        )

    campaign_paths = [Path(p) for p in args.campaign_summary_csvs]
    campaign_df = _load_campaign_tables(
        campaign_paths,
        args.campaign_names,
        allow_multirow_summary=args.allow_multirow_campaign_summary,
    )
    campaign_df.to_csv(output_dir / "exp4_campaign_ml_table.csv", index=False)

    print(f"Saved: {output_dir / 'exp4_offline_combined.csv'}")
    print(f"Saved: {output_dir / 'exp4_offline_primary_table.csv'}")
    print(f"Saved: {output_dir / 'exp4_offline_per_class_table.csv'}")
    print(f"Saved: {output_dir / 'exp4_offline_confusion_long.csv'}")
    print(f"Saved: {output_dir / 'exp4_offline_ci_table.csv'}")
    print(f"Saved: {output_dir / 'exp4_offline_model_rank.csv'}")
    print(f"Saved: {output_dir / 'exp4_final_candidates_table.csv'}")
    print(f"Saved: {output_dir / 'exp4_selection_metadata.json'}")
    print(f"Saved: {output_dir / 'exp4_final_confusion_long.csv'}")
    print(f"Saved: {output_dir / 'exp4_final_confusion_matrix.csv'}")
    print(f"Saved: {output_dir / 'exp4_campaign_ml_table.csv'}")


if __name__ == "__main__":
    main()
