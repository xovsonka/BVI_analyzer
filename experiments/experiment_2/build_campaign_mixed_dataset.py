from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGET_BY_CLASS = {
    "legit": 20,
    "phishing": 35,
    "spam": 20,
    "financial_fraud": 25,
}

LLM_TARGET_TOTAL = 50
TOTAL_TARGET = sum(TARGET_BY_CLASS.values())


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "attack_type" not in df.columns:
        raise ValueError(f"Missing attack_type in {path}")

    def _col(name: str, fallback: str = "") -> pd.Series:
        if name in df.columns:
            return df[name].fillna("").astype(str)
        return pd.Series([fallback] * len(df), index=df.index, dtype=str)

    out = pd.DataFrame()
    out["subject"] = _col("subject")
    out["text"] = _col("body") if "body" in df.columns else _col("text")
    out["sender"] = _col("from_email") if "from_email" in df.columns else _col("sender")
    out["attack_type"] = (
        df["attack_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"spoofing": "phishing"})
    )
    out["source"] = _col("source", "exp2")
    return out


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) >= n:
        return df.sample(n=n, random_state=seed)
    return df.sample(n=n, random_state=seed, replace=True)


def _dedup_pool(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dedup_key"] = (
        out["subject"].astype(str).str.strip().str.lower()
        + "||"
        + out["sender"].astype(str).str.strip().str.lower()
        + "||"
        + out["text"].astype(str).str.strip().str.lower()
    )
    return out.drop_duplicates(subset="dedup_key").drop(columns=["dedup_key"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Exp2 mixed 50/50 campaign CSV")
    parser.add_argument(
        "--llm-csv",
        default="results/experiment_1/llm_drafts/templates.csv",
    )
    parser.add_argument(
        "--mutated-csv",
        default="results/experiment_1/mutated/mutated_templates.csv",
    )
    parser.add_argument(
        "--real-edits-csv",
        default="results/experiment_1/multi_real_edits/multi_real_edited_templates.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="results/experiment_2/gophish_seed_input_mixed_50_50.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    llm_pool = _dedup_pool(_load(Path(args.llm_csv)))
    llm_pool["generation_kind"] = "llm"
    llm_pool["generation_family"] = "llm"

    mut_pool = _dedup_pool(_load(Path(args.mutated_csv)))
    mut_pool["generation_kind"] = "mutated"
    mut_pool["generation_family"] = "non_llm"

    real_pool = _dedup_pool(_load(Path(args.real_edits_csv)))
    real_pool["generation_kind"] = "real_edit"
    real_pool["generation_family"] = "non_llm"

    rule_pool = pd.concat([mut_pool, real_pool], ignore_index=True)

    rows = []
    next_id = 1
    classes = list(TARGET_BY_CLASS.keys())
    llm_quota = {cls: TARGET_BY_CLASS[cls] // 2 for cls in classes}
    current_llm = sum(llm_quota.values())
    remainder = LLM_TARGET_TOTAL - current_llm
    if remainder > 0:
        # deterministic fill to hit exact 50/50 split
        for cls in sorted(classes, key=lambda c: TARGET_BY_CLASS[c], reverse=True):
            if remainder <= 0:
                break
            if llm_quota[cls] < TARGET_BY_CLASS[cls]:
                llm_quota[cls] += 1
                remainder -= 1

    for cls, target in TARGET_BY_CLASS.items():
        llm_n = llm_quota[cls]
        rule_n = target - llm_n

        llm_cls = llm_pool[llm_pool["attack_type"] == cls]
        rule_cls = rule_pool[rule_pool["attack_type"] == cls]
        if llm_cls.empty:
            raise RuntimeError(f"No LLM rows for class '{cls}'")
        if rule_cls.empty:
            raise RuntimeError(f"No rule-based rows for class '{cls}'")

        llm_take = _sample(llm_cls, llm_n, args.seed)
        rule_take = _sample(rule_cls, rule_n, args.seed + 1)

        for _, row in pd.concat([llm_take, rule_take], ignore_index=True).iterrows():
            rows.append(
                {
                    "id": next_id,
                    "text": str(row["text"]),
                    "body": str(row["text"]),
                    "label": 0 if cls == "legit" else 1,
                    "source": f"exp2_{row['generation_kind']}",
                    "subject": str(row["subject"]),
                    "sender": str(row["sender"]),
                    "receiver": f"user{next_id}@local.lab",
                    "attack_type": cls,
                    "generation_kind": row["generation_kind"],
                    "generation_family": row.get("generation_family", "non_llm"),
                }
            )
            next_id += 1

    out = pd.DataFrame(rows)
    out = out[
        out["subject"].astype(str).str.strip().ne("")
        & out["text"].astype(str).str.strip().ne("")
        & out["sender"].astype(str).str.strip().ne("")
    ].copy()
    out["dedup_key"] = (
        out["subject"].astype(str).str.strip().str.lower()
        + "||"
        + out["sender"].astype(str).str.strip().str.lower()
        + "||"
        + out["text"].astype(str).str.strip().str.lower()
    )
    duplicate_rows = int(out["dedup_key"].duplicated().sum())
    out = out.drop(columns=["dedup_key"])
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    out["id"] = range(1, len(out) + 1)
    out["receiver"] = out["id"].map(lambda i: f"user{i}@local.lab")

    if len(out) != TOTAL_TARGET:
        raise RuntimeError(
            f"Dataset has {len(out)} rows after dedup, expected {TOTAL_TARGET}. "
            "Adjust source pools or seed to keep exact campaign size."
        )

    class_counts = out["attack_type"].value_counts().to_dict()
    for cls, target in TARGET_BY_CLASS.items():
        got = int(class_counts.get(cls, 0))
        if got != target:
            raise RuntimeError(f"Class '{cls}' count is {got}, expected {target}")

    family_counts = out["generation_family"].value_counts().to_dict()
    if int(family_counts.get("llm", 0)) != LLM_TARGET_TOTAL:
        raise RuntimeError(
            f"LLM family count is {int(family_counts.get('llm', 0))}, expected {LLM_TARGET_TOTAL}"
        )
    if int(family_counts.get("non_llm", 0)) != TOTAL_TARGET - LLM_TARGET_TOTAL:
        raise RuntimeError(
            "non_llm family count is "
            f"{int(family_counts.get('non_llm', 0))}, expected {TOTAL_TARGET - LLM_TARGET_TOTAL}"
        )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Saved {len(out)} rows to: {out_path}")
    print("Class distribution:")
    print(out["attack_type"].value_counts().to_string())
    print("Generation mix:")
    print(out["generation_kind"].value_counts().to_string())
    print("Generation family:")
    print(out["generation_family"].value_counts().to_string())
    print(f"Potential duplicate messages (subject+sender+text): {duplicate_rows}")


if __name__ == "__main__":
    main()
