import os
import glob
import pandas as pd

RAW_DIR = "/home/zammorak/thesis/data/raw/news_headlines/"
OUT_PATH = "/home/zammorak/thesis/data/interim/news_headlines_master.csv"

def standardize_columns(cols):
    return (
        pd.Index(cols)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

def combine_all_csvs_same_schema(
    folder_path: str,
    output_path: str,
    recursive: bool = False,
) -> pd.DataFrame:
    # Find CSVs
    pattern = "**/*.csv" if recursive else "*.csv"
    files = sorted(glob.glob(os.path.join(folder_path, pattern), recursive=recursive))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder_path}")

    # Pass 1: discover the FULL set of columns across all files (after standardization)
    all_cols = []
    for fp in files:
        # read just header (fast)
        header_df = pd.read_csv(fp, nrows=0)
        cols = standardize_columns(header_df.columns)
        all_cols.append(cols)

    # Canonical column order:
    # - start with common NVDA-news-like columns if present, then append the rest sorted
    union_cols = pd.Index([])

    for cols in all_cols:
        union_cols = union_cols.union(cols)

    preferred_order = [
        "symbol_target",
        "uuid",
        "published_at",
        "date",
        "title",
        "description",
        "keywords",
        "snippet",
        "url",
        "source",
        "image_url",
        "entities_json",
        "raw_json",
    ]
    preferred_present = [c for c in preferred_order if c in union_cols]
    remaining = sorted([c for c in union_cols if c not in preferred_present])
    canonical_cols = preferred_present + remaining

    # Pass 2: read all files, force the SAME columns + SAME index style
    frames = []
    for fp in files:
        df = pd.read_csv(fp)

        # standardize column names
        df.columns = standardize_columns(df.columns)

        # ensure identical schema (same columns, same order)
        df = df.reindex(columns=canonical_cols)

        # optional: keep traceability
        df["source_file"] = os.path.basename(fp)
        if "source_file" not in canonical_cols:
            # keep source_file at end (and consistent)
            canonical_cols = canonical_cols + ["source_file"]
            df = df.reindex(columns=canonical_cols)

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)  # RangeIndex 0..N-1 (consistent index)

    # Light cleaning that won't break schema
    if "published_at" in combined.columns:
        combined["published_at"] = pd.to_datetime(combined["published_at"], errors="coerce", utc=True)

    # De-dupe (optional but usually good)
    if "uuid" in combined.columns:
        combined = combined.drop_duplicates(subset=["uuid"], keep="first")
    else:
        combined = combined.drop_duplicates()

    # Reset index to be clean + consistent
    combined = combined.reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False)

    return combined


if __name__ == "__main__":
    df_master = combine_all_csvs_same_schema(
        folder_path=RAW_DIR,
        output_path=OUT_PATH,
        recursive=False,   # set True if there are subfolders
    )

    print("✅ Combined CSVs from:", RAW_DIR)
    print("✅ Rows:", len(df_master))
    print("✅ Columns:", len(df_master.columns))
    print("✅ Saved to:", OUT_PATH)
