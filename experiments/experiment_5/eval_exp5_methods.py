from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_core import build_model, ensure_features, select_model_input


BIN_LABELS = ["legit", "suspicious"]
MC_LABELS = ["legit", "phishing", "spam", "financial_fraud"]


def _file_key(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda x: Path(x).name)


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    yt = y_true.astype(str).str.lower().eq("suspicious").astype(int)
    yp = y_pred.astype(str).str.lower().eq("suspicious").astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return {
        "rows": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "mcc": float(matthews_corrcoef(yt, yp)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _multiclass_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    labels = sorted(set(y_true.astype(str)).union(set(y_pred.astype(str))))
    return {
        "rows": int(len(y_true)),
        "mc_accuracy": float(accuracy_score(y_true, y_pred)),
        "mc_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mc_f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "mc_f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "num_classes": int(len(labels)),
    }


def _heuristic_pred_binary(score: pd.Series, threshold: int) -> pd.Series:
    out = pd.Series(["legit"] * len(score), index=score.index, dtype=str)
    out.loc[score >= threshold] = "suspicious"
    return out


def _heuristic_pred_multiclass(binary_pred: pd.Series) -> pd.Series:
    out = pd.Series(["legit"] * len(binary_pred), index=binary_pred.index, dtype=str)
    out.loc[binary_pred.astype(str) == "suspicious"] = "phishing"
    return out


def _hybrid_a_binary(
    ml_mc_pred: pd.Series, score: pd.Series, high_threshold: int
) -> pd.Series:
    base = np.where(ml_mc_pred.astype(str).eq("legit"), "legit", "suspicious")
    out = pd.Series(base, index=ml_mc_pred.index, dtype=str)
    out.loc[(score >= high_threshold) & out.eq("legit")] = "suspicious"
    return out


def _hybrid_b_multiclass(
    heur_binary: pd.Series,
    ml_mc_pred: pd.Series,
    suspicious_fallback: str,
) -> pd.Series:
    out = pd.Series(["legit"] * len(ml_mc_pred), index=ml_mc_pred.index, dtype=str)
    suspicious_idx = heur_binary.astype(str).eq("suspicious")
    out.loc[suspicious_idx] = ml_mc_pred.loc[suspicious_idx].astype(str)
    out.loc[suspicious_idx & out.eq("legit")] = suspicious_fallback
    return out


def _per_class_rows(
    y_true: pd.Series,
    y_pred: pd.Series,
    method: str,
    split_name: str,
) -> list[dict]:
    rows = []
    labels = sorted(set(y_true.astype(str)).union(set(y_pred.astype(str))))
    for cls in labels:
        yt = y_true.astype(str).eq(cls).astype(int)
        yp = y_pred.astype(str).eq(cls).astype(int)
        rows.append(
            {
                "split": split_name,
                "method": method,
                "label": cls,
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "f1": float(f1_score(yt, yp, zero_division=0)),
                "support": int(yt.sum()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Exp5 methods: heuristic, ML, and hybrid variants"
    )
    parser.add_argument(
        "--benchmark-csv",
        default="results/experiment_5/exp5_unified_benchmark.csv",
    )
    parser.add_argument(
        "--processed-dir",
        default="dataset/processed/scenario_m_anti_template_feature_regularized",
    )
    parser.add_argument("--model", default="hybrid_logreg")
    parser.add_argument("--input-mode", default="text_plus_features")
    parser.add_argument(
        "--external-ml-pred-csv",
        default="",
        help="Optional external ML prediction CSV with file, ml_pred, ml_pred_binary.",
    )
    parser.add_argument("--heur-threshold", type=int, default=20)
    parser.add_argument("--hybrid-a-high-threshold", type=int, default=35)
    parser.add_argument("--hybrid-b-fallback", default="phishing")
    parser.add_argument("--max-train-rows", type=int, default=30000)
    parser.add_argument(
        "--output-binary-csv",
        default="results/experiment_5/exp5_method_compare_binary.csv",
    )
    parser.add_argument(
        "--output-multiclass-csv",
        default="results/experiment_5/exp5_method_compare_multiclass.csv",
    )
    parser.add_argument(
        "--output-per-class-csv",
        default="results/experiment_5/exp5_per_class_multiclass.csv",
    )
    parser.add_argument(
        "--output-confusion-binary-csv",
        default="results/experiment_5/exp5_confusion_binary_counts.csv",
    )
    parser.add_argument(
        "--output-confusion-multiclass-csv",
        default="results/experiment_5/exp5_confusion_multiclass_long.csv",
    )
    parser.add_argument(
        "--output-source-breakdown-csv",
        default="results/experiment_5/exp5_source_breakdown.csv",
    )
    parser.add_argument(
        "--output-group-breakdown-csv",
        default="results/experiment_5/exp5_group_breakdown.csv",
    )
    parser.add_argument(
        "--output-ranking-csv",
        default="results/experiment_5/exp5_method_ranking_binary.csv",
    )
    parser.add_argument(
        "--output-qualitative-csv",
        default="results/experiment_5/exp5_qualitative_examples.csv",
    )
    args = parser.parse_args()

    benchmark = pd.read_csv(args.benchmark_csv, low_memory=False)
    benchmark = ensure_features(benchmark)
    benchmark["y_true_binary"] = benchmark["y_true_binary"].astype(str).str.lower()
    benchmark["y_true_multiclass"] = (
        benchmark["y_true_multiclass"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": pd.NA, "<na>": pd.NA, "": pd.NA, "none": pd.NA})
    )

    if args.external_ml_pred_csv:
        pred_df = pd.read_csv(args.external_ml_pred_csv, low_memory=False).copy()
        pred_df["file_key"] = _file_key(pred_df["file"])
        join_cols = ["file"] if "file" in pred_df.columns and "file" in benchmark.columns else ["file_key"]
        pred_small = pred_df[[c for c in [*join_cols, "ml_pred", "ml_pred_binary"] if c in pred_df.columns]].drop_duplicates(subset=join_cols)
        benchmark = benchmark.copy()
        benchmark["file_key"] = _file_key(benchmark["file"])
        benchmark = benchmark.merge(pred_small, on=join_cols, how="left")
        ml_pred_mc = benchmark["ml_pred"].astype(str).replace({"nan": "", "<NA>": ""})
        missing_mc = ml_pred_mc.eq("")
        if missing_mc.any():
            raise RuntimeError(f"Missing external multiclass predictions for {int(missing_mc.sum())} benchmark rows")
        ml_pred_mc = ml_pred_mc.astype(str)
    else:
        train_path = Path(args.processed_dir) / "train.csv"
        train_df = ensure_features(
            pd.read_csv(
                train_path,
                low_memory=False,
                nrows=args.max_train_rows if args.max_train_rows > 0 else None,
            )
        )
        le = LabelEncoder()
        y_train = le.fit_transform(train_df["label"].astype(str))
        model = build_model(args.model, input_mode=args.input_mode)
        model.fit(select_model_input(train_df, args.input_mode), y_train)

        x_eval = select_model_input(benchmark, args.input_mode)
        ml_pred_idx = model.predict(x_eval)
        ml_pred_mc = pd.Series(le.inverse_transform(ml_pred_idx), index=benchmark.index)

    score = pd.to_numeric(benchmark["heuristic_score"], errors="coerce").fillna(0.0)
    heur_bin = _heuristic_pred_binary(score, args.heur_threshold)
    heur_mc = _heuristic_pred_multiclass(heur_bin)

    if args.external_ml_pred_csv and "ml_pred_binary" in benchmark.columns:
        ml_bin = benchmark["ml_pred_binary"].astype(str).replace({"nan": "", "<NA>": ""})
        missing_bin = ml_bin.eq("")
        if missing_bin.any():
            ml_bin = pd.Series(
                np.where(ml_pred_mc.eq("legit"), "legit", "suspicious"),
                index=benchmark.index,
            )
        else:
            ml_bin = ml_bin.astype(str)
    else:
        ml_bin = pd.Series(
            np.where(ml_pred_mc.eq("legit"), "legit", "suspicious"),
            index=benchmark.index,
        )

    hyb_a_mc = ml_pred_mc.copy()
    hyb_a_bin = _hybrid_a_binary(
        ml_pred_mc,
        score,
        high_threshold=args.hybrid_a_high_threshold,
    )

    hyb_b_bin = heur_bin.copy()
    hyb_b_mc = _hybrid_b_multiclass(
        hyb_b_bin,
        ml_pred_mc,
        suspicious_fallback=args.hybrid_b_fallback,
    )

    benchmark_out = benchmark.copy()
    benchmark_out["pred_heuristic_binary"] = heur_bin
    benchmark_out["pred_heuristic_multiclass"] = heur_mc
    benchmark_out["pred_ml_binary"] = ml_bin
    benchmark_out["pred_ml_multiclass"] = ml_pred_mc
    benchmark_out["pred_hybrid_a_binary"] = hyb_a_bin
    benchmark_out["pred_hybrid_a_multiclass"] = hyb_a_mc
    benchmark_out["pred_hybrid_b_binary"] = hyb_b_bin
    benchmark_out["pred_hybrid_b_multiclass"] = hyb_b_mc

    methods_binary = {
        "heuristic_only": heur_bin,
        "ml_only": ml_bin,
        "hybrid_a_conservative": hyb_a_bin,
        "hybrid_b_gated": hyb_b_bin,
    }
    methods_multiclass = {
        "heuristic_only": heur_mc,
        "ml_only": ml_pred_mc,
        "hybrid_a_conservative": hyb_a_mc,
        "hybrid_b_gated": hyb_b_mc,
    }

    mc_mask = benchmark_out["y_true_multiclass"].notna()

    binary_rows = []
    multiclass_rows = []
    per_class_rows = []
    confusion_rows = []

    scopes: list[tuple[str, pd.DataFrame]] = [("all", benchmark_out)]
    scopes.extend(
        [
            (f"group:{group}", part)
            for group, part in benchmark_out.groupby("benchmark_group")
        ]
    )
    scopes.extend(
        [
            (f"source:{source}", part)
            for source, part in benchmark_out.groupby("benchmark_source")
        ]
    )

    for split_name, part in scopes:
        for method, col in {
            "heuristic_only": "pred_heuristic_binary",
            "ml_only": "pred_ml_binary",
            "hybrid_a_conservative": "pred_hybrid_a_binary",
            "hybrid_b_gated": "pred_hybrid_b_binary",
        }.items():
            m = _binary_metrics(part["y_true_binary"], part[col])
            binary_rows.append({"split": split_name, "method": method, **m})

        part_mc_mask = part["y_true_multiclass"].notna()
        if not part_mc_mask.any():
            continue
        part_true = part.loc[part_mc_mask, "y_true_multiclass"].astype(str)
        for method, col in {
            "heuristic_only": "pred_heuristic_multiclass",
            "ml_only": "pred_ml_multiclass",
            "hybrid_a_conservative": "pred_hybrid_a_multiclass",
            "hybrid_b_gated": "pred_hybrid_b_multiclass",
        }.items():
            part_pred = part.loc[part_mc_mask, col].astype(str)
            m = _multiclass_metrics(part_true, part_pred)
            multiclass_rows.append({"split": split_name, "method": method, **m})
            per_class_rows.extend(
                _per_class_rows(
                    part_true, part_pred, method=method, split_name=split_name
                )
            )

            labels = sorted(set(part_true).union(set(part_pred)))
            cm = confusion_matrix(part_true, part_pred, labels=labels)
            for i, true_label in enumerate(labels):
                for j, pred_label in enumerate(labels):
                    confusion_rows.append(
                        {
                            "split": split_name,
                            "method": method,
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "count": int(cm[i, j]),
                        }
                    )

    binary_df = pd.DataFrame(binary_rows)
    multiclass_df = pd.DataFrame(multiclass_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    confusion_mc_df = pd.DataFrame(confusion_rows)

    source_rows = []
    for source, part in benchmark_out.groupby("benchmark_source"):
        for method, col in {
            "heuristic_only": "pred_heuristic_binary",
            "ml_only": "pred_ml_binary",
            "hybrid_a_conservative": "pred_hybrid_a_binary",
            "hybrid_b_gated": "pred_hybrid_b_binary",
        }.items():
            m = _binary_metrics(part["y_true_binary"], part[col])
            source_rows.append(
                {
                    "benchmark_source": source,
                    "rows": int(len(part)),
                    "method": method,
                    **m,
                }
            )
    source_df = pd.DataFrame(source_rows)

    group_rows = []
    for group, part in benchmark_out.groupby("benchmark_group"):
        for method, col in {
            "heuristic_only": "pred_heuristic_binary",
            "ml_only": "pred_ml_binary",
            "hybrid_a_conservative": "pred_hybrid_a_binary",
            "hybrid_b_gated": "pred_hybrid_b_binary",
        }.items():
            m = _binary_metrics(part["y_true_binary"], part[col])
            group_rows.append(
                {
                    "benchmark_group": group,
                    "rows": int(len(part)),
                    "method": method,
                    **m,
                }
            )
    group_df = pd.DataFrame(group_rows)

    ranking_df = binary_df[["split", "method", "f1", "balanced_accuracy", "mcc"]].copy()
    ranking_df["rank_f1"] = ranking_df.groupby("split")["f1"].rank(
        ascending=False, method="dense"
    )
    ranking_df["rank_balanced_accuracy"] = ranking_df.groupby("split")[
        "balanced_accuracy"
    ].rank(ascending=False, method="dense")
    ranking_df["rank_mcc"] = ranking_df.groupby("split")["mcc"].rank(
        ascending=False, method="dense"
    )
    ranking_df = ranking_df.sort_values(["split", "rank_mcc", "rank_f1"]).reset_index(
        drop=True
    )

    confusion_bin_df = binary_df[
        ["split", "method", "tn", "fp", "fn", "tp", "rows"]
    ].copy()

    benchmark_out["is_ml_correct_mc"] = (
        benchmark_out["pred_ml_multiclass"].astype(str)
        == benchmark_out["y_true_multiclass"].astype(str)
    ).astype(int)
    benchmark_out["is_hyb_b_correct_mc"] = (
        benchmark_out["pred_hybrid_b_multiclass"].astype(str)
        == benchmark_out["y_true_multiclass"].astype(str)
    ).astype(int)
    benchmark_out["is_heur_correct_mc"] = (
        benchmark_out["pred_heuristic_multiclass"].astype(str)
        == benchmark_out["y_true_multiclass"].astype(str)
    ).astype(int)
    benchmark_out["is_hyb_a_correct_mc"] = (
        benchmark_out["pred_hybrid_a_multiclass"].astype(str)
        == benchmark_out["y_true_multiclass"].astype(str)
    ).astype(int)

    qual_parts = []
    cond_heur_wrong_ml_correct = (
        mc_mask
        & (benchmark_out["is_heur_correct_mc"] == 0)
        & (benchmark_out["is_ml_correct_mc"] == 1)
    )
    q1 = benchmark_out[cond_heur_wrong_ml_correct].copy()
    q1["error_type"] = "heuristic_wrong_ml_correct"
    qual_parts.append(q1.sort_values("heuristic_score", ascending=False).head(4))

    cond_ml_wrong_hyb_b_correct = (
        mc_mask
        & (benchmark_out["is_ml_correct_mc"] == 0)
        & (benchmark_out["is_hyb_b_correct_mc"] == 1)
    )
    q2 = benchmark_out[cond_ml_wrong_hyb_b_correct].copy()
    q2["error_type"] = "ml_wrong_hybrid_b_correct"
    qual_parts.append(q2.sort_values("heuristic_score", ascending=False).head(4))

    cond_ml_wrong_hyb_a_correct = (
        mc_mask
        & (benchmark_out["is_ml_correct_mc"] == 0)
        & (benchmark_out["is_hyb_a_correct_mc"] == 1)
    )
    q3 = benchmark_out[cond_ml_wrong_hyb_a_correct].copy()
    q3["error_type"] = "ml_wrong_hybrid_a_correct"
    qual_parts.append(q3.sort_values("heuristic_score", ascending=False).head(4))

    cond_all_wrong = (
        mc_mask
        & (benchmark_out["is_heur_correct_mc"] == 0)
        & (benchmark_out["is_ml_correct_mc"] == 0)
        & (benchmark_out["is_hyb_b_correct_mc"] == 0)
    )
    q4 = benchmark_out[cond_all_wrong].copy()
    q4["error_type"] = "all_wrong"
    qual_parts.append(q4.sort_values("heuristic_score", ascending=False).head(4))

    if qual_parts:
        qual = pd.concat(qual_parts, ignore_index=True).drop_duplicates(
            subset=["benchmark_source", "file", "error_type"]
        )
    else:
        qual = benchmark_out.head(0).copy()
        qual["error_type"] = pd.Series(dtype=str)

    out_dir = Path(args.output_binary_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    binary_df.to_csv(args.output_binary_csv, index=False)
    multiclass_df.to_csv(args.output_multiclass_csv, index=False)
    per_class_df.to_csv(args.output_per_class_csv, index=False)
    confusion_bin_df.to_csv(args.output_confusion_binary_csv, index=False)
    confusion_mc_df.to_csv(args.output_confusion_multiclass_csv, index=False)
    source_df.to_csv(args.output_source_breakdown_csv, index=False)
    group_df.to_csv(args.output_group_breakdown_csv, index=False)
    ranking_df.to_csv(args.output_ranking_csv, index=False)
    qual[
        [
            "error_type",
            "benchmark_group",
            "benchmark_source",
            "file",
            "subject",
            "y_true_binary",
            "y_true_multiclass",
            "heuristic_score",
            "heuristic_reasons",
            "is_heur_correct_mc",
            "is_ml_correct_mc",
            "is_hyb_a_correct_mc",
            "is_hyb_b_correct_mc",
            "pred_heuristic_binary",
            "pred_heuristic_multiclass",
            "pred_ml_binary",
            "pred_ml_multiclass",
            "pred_hybrid_a_binary",
            "pred_hybrid_a_multiclass",
            "pred_hybrid_b_binary",
            "pred_hybrid_b_multiclass",
        ]
    ].to_csv(args.output_qualitative_csv, index=False)

    benchmark_out.to_csv(out_dir / "exp5_benchmark_with_predictions.csv", index=False)

    print(f"Saved binary comparison: {args.output_binary_csv}")
    print(f"Saved multiclass comparison: {args.output_multiclass_csv}")
    print(f"Saved per-class metrics: {args.output_per_class_csv}")
    print(f"Saved binary confusion counts: {args.output_confusion_binary_csv}")
    print(f"Saved multiclass confusion long: {args.output_confusion_multiclass_csv}")
    print(f"Saved source breakdown: {args.output_source_breakdown_csv}")
    print(f"Saved group breakdown: {args.output_group_breakdown_csv}")
    print(f"Saved method ranking: {args.output_ranking_csv}")
    print(f"Saved qualitative examples: {args.output_qualitative_csv}")
    print(
        f"Saved enriched benchmark: {out_dir / 'exp5_benchmark_with_predictions.csv'}"
    )


if __name__ == "__main__":
    main()
