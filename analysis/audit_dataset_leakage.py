from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _norm(text: object) -> str:
    if text is None or pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _pick_text_col(df: pd.DataFrame) -> str:
    for col in ["text_input", "text", "body", "body_text"]:
        if col in df.columns:
            return col
    raise ValueError(f"No text column found in {df.columns.tolist()}")


def _exact_overlap(source: pd.DataFrame, target: pd.DataFrame, source_text_col: str, target_text_col: str) -> dict[str, int]:
    src_subjects = {_norm(v) for v in source.get("subject", pd.Series(dtype=str)).tolist() if _norm(v)}
    src_texts = {_norm(v) for v in source[source_text_col].tolist() if _norm(v)}

    tgt_subjects = [_norm(v) for v in target.get("subject", pd.Series(dtype=str)).tolist()]
    tgt_texts = [_norm(v) for v in target[target_text_col].tolist()]

    return {
        "subject_exact_overlap": int(sum(1 for value in tgt_subjects if value and value in src_subjects)),
        "text_exact_overlap": int(sum(1 for value in tgt_texts if value and value in src_texts)),
    }


def _top_similarity_rows(source: pd.DataFrame, target: pd.DataFrame, source_text_col: str, target_text_col: str, top_k: int = 5) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    src_rows = [
        {
            "subject": _norm(row.get("subject", "")),
            "text": _norm(row.get(source_text_col, "")),
            "label": str(row.get("label", row.get("attack_type", ""))),
        }
        for _, row in source.iterrows()
    ]
    tgt_rows = [
        {
            "subject": _norm(row.get("subject", "")),
            "text": _norm(row.get(target_text_col, "")),
            "label": str(row.get("label", row.get("attack_type", ""))),
        }
        for _, row in target.iterrows()
    ]

    for tgt in tgt_rows:
        best_ratio = 0.0
        best_src: dict[str, object] | None = None
        for src in src_rows:
            subj_ratio = SequenceMatcher(None, tgt["subject"], src["subject"]).ratio() if tgt["subject"] and src["subject"] else 0.0
            text_ratio = SequenceMatcher(None, tgt["text"], src["text"]).ratio() if tgt["text"] and src["text"] else 0.0
            ratio = max(subj_ratio, text_ratio)
            if ratio > best_ratio:
                best_ratio = ratio
                best_src = src
        if best_src is not None:
            rows.append(
                {
                    "target_label": tgt["label"],
                    "source_label": best_src["label"],
                    "best_similarity": float(best_ratio),
                    "target_subject": tgt["subject"][:120],
                    "source_subject": best_src["subject"][:120],
                }
            )
    rows = sorted(rows, key=lambda item: item["best_similarity"], reverse=True)
    return rows[:top_k]


def main() -> None:
    scenario_train_path = PROJECT_ROOT / "dataset" / "processed" / "scenario_m_anti_template_feature_regularized" / "train.csv"
    adaptation_path = PROJECT_ROOT / "results" / "adaptation" / "campaign_style_train_adaptation.csv"
    eval_paths = {
        "exp1_campaign": PROJECT_ROOT / "results" / "experiment_5" / "exp1_campaign_model_input.csv",
        "exp2_seed": PROJECT_ROOT / "results" / "experiment_2_clean" / "gophish_seed_input_header_focused_clean.csv",
        "exp3_seed": PROJECT_ROOT / "results" / "experiment_3_clean" / "gophish_seed_input_url_focused_clean.csv",
    }

    out_dir = PROJECT_ROOT / "results" / "dataset_leakage_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_train = _load_csv(scenario_train_path)
    adaptation = _load_csv(adaptation_path)
    scenario_text_col = _pick_text_col(scenario_train)
    adaptation_text_col = _pick_text_col(adaptation)

    summary_rows = []
    detail_payload: dict[str, object] = {}

    for name, eval_path in eval_paths.items():
        eval_df = _load_csv(eval_path)
        eval_text_col = _pick_text_col(eval_df)

        train_exact = _exact_overlap(scenario_train, eval_df, scenario_text_col, eval_text_col)
        adapt_exact = _exact_overlap(adaptation, eval_df, adaptation_text_col, eval_text_col)
        top_near = _top_similarity_rows(adaptation, eval_df, adaptation_text_col, eval_text_col, top_k=8)
        max_near = max((row["best_similarity"] for row in top_near), default=0.0)

        summary_rows.append(
            {
                "eval_set": name,
                "scenario_train_subject_exact_overlap": train_exact["subject_exact_overlap"],
                "scenario_train_text_exact_overlap": train_exact["text_exact_overlap"],
                "adaptation_subject_exact_overlap": adapt_exact["subject_exact_overlap"],
                "adaptation_text_exact_overlap": adapt_exact["text_exact_overlap"],
                "adaptation_max_near_similarity": float(max_near),
            }
        )
        detail_payload[name] = {"top_near_matches": top_near}

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "detail.json").write_text(json.dumps(detail_payload, indent=2), encoding="utf-8")
    print(f"Saved dataset leakage audit to: {out_dir}")


if __name__ == "__main__":
    main()
