from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def normalize_text(text: str) -> str:
    out = (text or "").lower()
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"\s+", " ", out).strip()
    return out


def exact_overlap(a: pd.Series, b: pd.Series) -> dict:
    set_a = set(a.tolist())
    set_b = set(b.tolist())
    inter = set_a.intersection(set_b)
    rows_a = int(a.isin(inter).sum())
    rows_b = int(b.isin(inter).sum())
    return {
        "exact_unique_overlap": int(len(inter)),
        "exact_rows_a": rows_a,
        "exact_rows_b": rows_b,
        "exact_pct_a": float(rows_a / len(a)) if len(a) else 0.0,
        "exact_pct_b": float(rows_b / len(b)) if len(b) else 0.0,
    }


def near_overlap(a: pd.Series, b: pd.Series, threshold: float) -> dict:
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=120000,
        sublinear_tf=True,
    )
    ab = pd.concat(
        [a.reset_index(drop=True), b.reset_index(drop=True)], ignore_index=True
    )
    x_all = vec.fit_transform(ab)
    x_a = x_all[: len(a)]
    x_b = x_all[len(a) :]

    nn_ab = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn_ab.fit(x_b)
    d_ab, idx_ab = nn_ab.kneighbors(x_a)
    sim_ab = 1.0 - d_ab[:, 0]

    nn_ba = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn_ba.fit(x_a)
    d_ba, idx_ba = nn_ba.kneighbors(x_b)
    sim_ba = 1.0 - d_ba[:, 0]

    mask_ab = sim_ab >= threshold
    mask_ba = sim_ba >= threshold

    samples = []
    for i in np.where(mask_ab)[0].tolist()[:50]:
        j = int(idx_ab[i][0])
        samples.append(
            {
                "dir": "a_to_b",
                "a_idx": int(i),
                "b_idx": j,
                "similarity": float(sim_ab[i]),
                "a_preview": str(a.iloc[i])[:220],
                "b_preview": str(b.iloc[j])[:220],
            }
        )

    return {
        "near_rows_a": int(np.sum(mask_ab)),
        "near_rows_b": int(np.sum(mask_ba)),
        "near_pct_a": float(np.mean(mask_ab)) if len(mask_ab) else 0.0,
        "near_pct_b": float(np.mean(mask_ba)) if len(mask_ba) else 0.0,
        "mean_nn_sim_a_to_b": float(np.mean(sim_ab)) if len(sim_ab) else 0.0,
        "mean_nn_sim_b_to_a": float(np.mean(sim_ba)) if len(sim_ba) else 0.0,
        "samples": samples,
    }


def load_norm(path: Path, text_col: str, max_rows: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if text_col not in df.columns:
        text_col = "text"
    out = df[[text_col]].copy()
    out[text_col] = out[text_col].fillna("").astype(str).map(normalize_text)
    out = out[out[text_col] != ""].reset_index(drop=True)
    if max_rows > 0 and len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return out


def build_paths(processed: Path, mode: str, scenario: str) -> dict[str, Path]:
    if mode == "scenario":
        scen = processed / scenario
        return {
            "train": scen / "train.csv",
            "val": scen / "val.csv",
            "test": scen / "test.csv",
        }

    shared = processed / "shared"
    return {
        "train_pool": shared / "train_pool.csv",
        "val_iid": shared / "val_iid.csv",
        "test_iid": shared / "test_iid.csv",
        "test_hard_source": shared / "test_hard_source.csv",
        "test_hard_cluster": shared / "test_hard_cluster.csv",
        "test_deployment": shared / "test_deployment.csv",
    }


def build_pairs(mode: str) -> list[tuple[str, str]]:
    if mode == "scenario":
        return [("train", "val"), ("train", "test"), ("val", "test")]

    return [
        ("train_pool", "val_iid"),
        ("train_pool", "test_iid"),
        ("train_pool", "test_hard_source"),
        ("train_pool", "test_hard_cluster"),
        ("train_pool", "test_deployment"),
        ("val_iid", "test_iid"),
        ("val_iid", "test_deployment"),
        ("test_iid", "test_deployment"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full split duplicate matrix (exact + near)"
    )
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument(
        "--mode",
        default="shared",
        choices=["shared", "scenario"],
        help="shared = audit shared multi-eval splits; scenario = legacy train/val/test audit",
    )
    parser.add_argument("--scenario", default="scenario_g_source_balanced")
    parser.add_argument("--text-col", default="text")
    parser.add_argument(
        "--max-rows-per-split",
        type=int,
        default=30000,
        help="0 = no sampling (full split)",
    )
    parser.add_argument(
        "--near-thresholds",
        nargs="+",
        type=float,
        default=[0.90],
        help="One or more near-duplicate thresholds (e.g. 0.85 0.90 0.95)",
    )
    parser.add_argument("--output-csv", default="results/split_dup_matrix.csv")
    parser.add_argument("--output-json", default="results/split_dup_matrix.json")
    parser.add_argument("--output-samples-csv", default="results/split_dup_samples.csv")
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    paths = build_paths(processed, mode=args.mode, scenario=args.scenario)
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    data = {
        name: load_norm(path, args.text_col, args.max_rows_per_split)
        for name, path in paths.items()
    }

    pairs = build_pairs(args.mode)
    rows: list[dict] = []
    samples: list[dict] = []

    for a_name, b_name in pairs:
        a = data[a_name][args.text_col]
        b = data[b_name][args.text_col]
        ex = exact_overlap(a, b)
        for th in args.near_thresholds:
            nr = near_overlap(a, b, threshold=float(th))
            rows.append(
                {
                    "pair": f"{a_name}_vs_{b_name}",
                    "threshold": float(th),
                    "rows_a": int(len(a)),
                    "rows_b": int(len(b)),
                    **ex,
                    **{k: v for k, v in nr.items() if k != "samples"},
                }
            )

            for s in nr["samples"]:
                samples.append(
                    {"pair": f"{a_name}_vs_{b_name}", "threshold": float(th), **s}
                )

    out_df = pd.DataFrame(rows)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    out_samples = Path(args.output_samples_csv)
    out_samples.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(samples).to_csv(out_samples, index=False)

    payload = {
        "scenario": args.scenario,
        "mode": args.mode,
        "text_col": args.text_col,
        "near_thresholds": [float(x) for x in args.near_thresholds],
        "pairs": rows,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved CSV: {out_csv}")
    print(f"Saved samples: {out_samples}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
