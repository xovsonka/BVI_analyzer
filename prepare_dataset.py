from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "dataset" / "source"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "campaign_input.csv"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
VAL_FILE = PROCESSED_DIR / "val.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"


TEXT_PRIORITY = ["text", "text_combined", "message", "body"]


def safe_read_csv(file_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, low_memory=False, encoding="latin1")


def normalize_text(df: pd.DataFrame) -> pd.Series:
    cols_lower = {c.lower(): c for c in df.columns}

    if "subject" in cols_lower and "body" in cols_lower:
        subject_col = cols_lower["subject"]
        body_col = cols_lower["body"]
        return (
            df[subject_col].fillna("").astype(str).str.strip()
            + "\n\n"
            + df[body_col].fillna("").astype(str).str.strip()
        ).str.strip()

    for key in TEXT_PRIORITY:
        if key in cols_lower:
            return df[cols_lower[key]].fillna("").astype(str).str.strip()

    raise ValueError("No supported text column found")


def normalize_label(df: pd.DataFrame, file_path: Path) -> pd.Series:
    cols_lower = {c.lower(): c for c in df.columns}

    if "label" in cols_lower:
        label_col = cols_lower["label"]
        labels = pd.to_numeric(df[label_col], errors="coerce")
        return labels.fillna(0).astype(int)

    if "enron" in file_path.parts:
        return pd.Series([0] * len(df), index=df.index)

    return pd.Series([pd.NA] * len(df), index=df.index)


def source_name(file_path: Path) -> str:
    return file_path.stem


def build_dataset() -> pd.DataFrame:
    csv_files = sorted(SOURCE_DIR.rglob("*.csv"))
    rows = []

    for file_path in csv_files:
        df = safe_read_csv(file_path)

        try:
            text_series = normalize_text(df)
        except ValueError:
            print(f"Skipping (missing text): {file_path}")
            continue

        label_series = normalize_label(df, file_path)

        prepared = pd.DataFrame(
            {
                "text": text_series,
                "label": label_series,
                "source": source_name(file_path),
            }
        )

        rows.append(prepared)
        print(f"Loaded {file_path} -> {len(prepared)} rows")

    if not rows:
        raise RuntimeError("No usable CSV files found")

    dataset = pd.concat(rows, ignore_index=True)
    dataset = dataset.dropna(subset=["label"])
    dataset["label"] = dataset["label"].astype(int)
    dataset["text"] = dataset["text"].fillna("").astype(str).str.strip()
    dataset = dataset[dataset["text"] != ""]
    dataset = dataset.drop_duplicates(subset=["text", "label"])
    dataset.insert(0, "id", range(1, len(dataset) + 1))

    return dataset


def print_split_stats(name: str, df: pd.DataFrame) -> None:
    print(f"{name}: {len(df)} rows")
    print(df["label"].value_counts().sort_index().to_string())


def create_splits(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 80/10/10 stratified split
    train_df, temp_df = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify=dataset["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _label_percentage(df: pd.DataFrame, label_value: int) -> float:
    return (df["label"] == label_value).mean() * 100


def _text_hash_set(df: pd.DataFrame) -> set[int]:
    text_hashes = pd.util.hash_pandas_object(df["text"], index=False)
    return set(text_hashes.astype("uint64").tolist())


def verify_split_quality(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    print("Split Quality Checks")
    print("- Label stratification (%)")
    for name, df in (("Train", train_df), ("Validation", val_df), ("Test", test_df)):
        p0 = _label_percentage(df, 0)
        p1 = _label_percentage(df, 1)
        print(f"  {name}: label0={p0:.2f}%, label1={p1:.2f}%")

    print("- Cross-split duplicate check (text overlap)")
    train_hash = _text_hash_set(train_df)
    val_hash = _text_hash_set(val_df)
    test_hash = _text_hash_set(test_df)
    print(f"  train ∩ val: {len(train_hash & val_hash)}")
    print(f"  train ∩ test: {len(train_hash & test_hash)}")
    print(f"  val ∩ test: {len(val_hash & test_hash)}")

    print("- Source balance (%)")
    combined = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    source_table = (
        pd.crosstab(combined["split"], combined["source"], normalize="index") * 100
    ).round(2)
    print(source_table.to_string())


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    dataset.to_csv(OUTPUT_FILE, index=False)
    train_df, val_df, test_df = create_splits(dataset)
    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VAL_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print("\nDone")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(dataset)}")
    print("Label distribution:")
    print(dataset["label"].value_counts().sort_index().to_string())
    print()
    print_split_stats("Train", train_df)
    print()
    print_split_stats("Validation", val_df)
    print()
    print_split_stats("Test", test_df)
    print()
    print(f"Saved: {TRAIN_FILE}")
    print(f"Saved: {VAL_FILE}")
    print(f"Saved: {TEST_FILE}")
    print()
    verify_split_quality(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
