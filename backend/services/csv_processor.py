import pandas as pd
import s3fs
import re
import sys
import argparse
from pathlib import Path


RAW_CSV_DIR = "s3://nexo-storage-ca/input_data/csv/eod_csv"
PROCESSED_DIR = "s3://nexo-storage-ca/input_data/csv/processed_csv"


def find_latest_csv():

    fs = s3fs.S3FileSystem()
    candidates = fs.glob(f"{RAW_CSV_DIR}/*.csv")
    if not candidates:
        print("No CSV files found in", RAW_CSV_DIR)
        return None

    # s3fs may return paths with or without the s3:// prefix. normalize to s3://
    chosen = sorted(candidates)[-1]
    if not str(chosen).startswith("s3://"):
        chosen = "s3://" + str(chosen).lstrip("/")
    return chosen


def process_csv(src_path: str, verbose=False):

    # Ensure output dir exists (PROCESSED_DIR might be an s3 path; mkdir only valid for local Path)
    try:
        src_str = str(src_path)
        is_s3 = src_str.startswith("s3://")
        if not is_s3 and src_str.startswith("nexo-storage-ca/"):
            src_str = "s3://" + src_str.lstrip("/")
            is_s3 = True

        if is_s3:
            try:
                df = pd.read_csv(
                    src_str,
                    dtype=str,
                    keep_default_na=False,
                    low_memory=False,
                    storage_options={"anon": False},
                )
            except Exception:
                df = pd.read_csv(
                    src_str,
                    dtype=str,
                    keep_default_na=False,
                    encoding="latin1",
                    low_memory=False,
                    storage_options={"anon": False},
                )
        else:
            try:
                df = pd.read_csv(
                    src_str,
                    dtype=str,
                    keep_default_na=False,
                    low_memory=False,
                )
            except Exception:
                df = pd.read_csv(
                    src_str,
                    dtype=str,
                    keep_default_na=False,
                    encoding="latin1",
                    low_memory=False,
                )
    except Exception as e:
        print("ERROR reading CSV:", e)
        raise

    df.columns = [c.strip() for c in df.columns]

    if verbose:
        print("DEBUG: Raw columns:", list(df.columns))

    drop_cols = [c for c in df.columns if "currency" in c.lower()]
    df = df.drop(columns=drop_cols, errors="ignore")

    def to_float(x):
        try:
            if x in ("", None):
                return None
            return float(str(x).replace(",", "").strip())
        except Exception:
            return None

    def to_pct(x):
        try:
            if x in ("", None):
                return None
            return round(float(str(x).replace(",", "").strip()), 2)
        except Exception:
            return None

    out = pd.DataFrame()
    out["symbol"] = df.get("Symbol")
    out["company_name"] = df.get("Description")

    out["mcap_rs_cr"] = (
        df["Market capitalization"]
        .map(to_float)
        .map(lambda v: round(v / 1e7, 2) if v else None)
    )
    out["price"] = df["Price"].map(to_float).map(lambda v: round(v, 2) if v else None)
    out["all_time_high"] = (
        df["High All Time"].map(to_float).map(lambda v: round(v, 2) if v else None)
    )

    out["change_1d_pct"] = df["Price Change % 1 day"].map(to_pct)
    out["change_1w_pct"] = df["Price Change % 1 week"].map(to_pct)

    out["volume_24h_rs_cr"] = (
        df["Price * Volume (Turnover) 1 day"]
        .map(to_float)
        .map(lambda v: round(v / 1e7, 2) if v else None)
    )
    out["vwap"] = df["Volume Weighted Average Price 1 day"].map(to_float).map(
        lambda v: round(v, 2) if v else None
    )
    out["atr_pct"] = df["Average True Range % (14) 1 day"].map(to_pct)
    out["volatility"] = df["Volatility 1 day"].map(to_pct)
    out["vol_change_pct"] = df["Volume Change % 1 day"].map(to_pct)
    out["relative_vol"] = df["Relative Volume 1 day"].map(to_float).map(
        lambda v: round(v, 2) if v else None
    )

    before = len(out)
    out = out.drop_duplicates(subset=["symbol"], keep="first")
    after = len(out)
    if verbose and before != after:
        print(f"Deduplicated {before - after} rows")

    out = out.sort_values(
        by="mcap_rs_cr", ascending=False, na_position="last"
    ).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))

    safe_name = "processed_" + re.sub(
        r"[^A-Za-z0-9._-]", "_", Path(src_path).name
    )
    out_path = f"{PROCESSED_DIR}/{safe_name}"

    try:
        out.to_csv(out_path, index=False, encoding="utf-8", storage_options={"anon": False})
    except Exception as e:
        print("ERROR writing processed CSV:", e)
        raise

    print(f"Wrote processed CSV: {out_path} (rows={len(out)})")
    return out_path


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="Path to specific CSV file", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.src:
        src_path = args.src
    else:
        src_path = find_latest_csv()

    if not src_path:
        print("No source CSV found.")
        sys.exit(1)

    if args.verbose:
        print("Processing file:", src_path)

    process_csv(src_path, verbose=args.verbose)


if __name__ == "__main__":
    main()
