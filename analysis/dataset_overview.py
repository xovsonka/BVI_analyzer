from pathlib import Path
import pandas as pd

BASE = Path("dataset/source")
csv_files = sorted(BASE.rglob("*.csv"))

print(f"Nasiel som {len(csv_files)} CSV suborov.\n")

source_counts = {}

for f in csv_files:
    source_name = f.parent.name
    source_counts[source_name] = source_counts.get(source_name, 0) + 1

    print("=" * 80)
    print(f"Subor: {f}")

    # pokus o nacitanie (bezpecnejsie pre starsie datasety)
    try:
        df = pd.read_csv(f, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(f, low_memory=False, encoding="latin1")

    print(f"Riadky: {len(df):,}")
    print(f"Stlpce: {len(df.columns)}")
    print("Nazvy stlpcov:", list(df.columns))

    # ukazka prvych 2 riadkov (skratene)
    print("\nUkazka:")
    print(df.head(2).to_string(index=False, max_colwidth=80))

    # missing values
    na_ratio = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    print("\nTop missing hodnoty (%):")
    print(na_ratio.head(5).round(2).to_string())

    # ak existuje label-like stlpec, vypis distribuciu
    label_candidates = [
        c for c in df.columns if c.lower() in {"label", "class", "type", "category"}
    ]
    if label_candidates:
        c = label_candidates[0]
        print(f"\nDistribucia '{c}':")
        print(df[c].value_counts(dropna=False).head(10).to_string())

print("\nPrehlad podla zdroja:")
for source_name, count in sorted(source_counts.items()):
    print(f"- {source_name}: {count} CSV")

print("\nHotovo.")
