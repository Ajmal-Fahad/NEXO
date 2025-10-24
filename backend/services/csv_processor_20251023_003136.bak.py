#!/usr/bin/env python3
"""
backend/services/csv_processor.py
Supports local and S3 sources. Uses .env S3_* paths if present.
"""
from __future__ import annotations
import os
import re
import sys
import argparse
from pathlib import Path
# --- fix sys.path when run directly ---
sys.path.append(str(Path(__file__).resolve().parents[2]))

# third-party
import pandas as pd
# s3fs is optional but expected if using S3 paths
try:
    import s3fs
except Exception:
    s3fs = None

# load .env from repo root so S3 paths / creds are visible when running locally
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
except Exception:
    # python-dotenv not available — rely on environment variables only
    pass

# --- paths: prefer local directories for local testing; allow S3 via .env ---
BASE = Path(__file__).resolve().parents[1]
LOCAL_RAW_DIR = BASE / "input_data" / "csv" / "eod_csv"
LOCAL_PROCESSED_DIR = BASE / "input_data" / "csv" / "processed_csv"

# env overrides (may be empty)
S3_RAW_CSV_PATH = os.getenv("S3_RAW_CSV_PATH", "").strip()
S3_PROCESSED_CSV_PATH = os.getenv("S3_PROCESSED_CSV_PATH", "").strip()

# Effective runtime paths (strings). If S3 env is present use that; else local path strings.
RAW_CSV_DIR = S3_RAW_CSV_PATH if S3_RAW_CSV_PATH else str(LOCAL_RAW_DIR)
PROCESSED_DIR = S3_PROCESSED_CSV_PATH if S3_PROCESSED_CSV_PATH else str(LOCAL_PROCESSED_DIR)

# --- helpers ----------------------------------------------------------------
def _is_s3_path(p: str) -> bool:
    return isinstance(p, str) and p.startswith("s3://")

# ---- file discovery (S3-first, local-fallback) --------------------------------
def _s3_list_latest_csv(s3_path: str) -> str | None:
    if s3fs is None:
        print("s3fs not installed — cannot list S3. Install s3fs in your venv.")
        return None
    fs = s3fs.S3FileSystem(anon=False)
    try:
        candidates = fs.glob(f"{s3_path.rstrip('/') }/*.csv")
        if not candidates:
            return None
        # normalize and pick last lexicographically (or you can sort by key)
        chosen = sorted(candidates)[-1]
        if not str(chosen).startswith("s3://"):
            chosen = "s3://" + str(chosen).lstrip("/")
        return str(chosen)
    except Exception as e:
        print("S3 listing error:", e)
        return None

def find_latest_csv() -> str | None:
    """
    Return latest CSV path.
      - If RAW_CSV_DIR is an S3 URL (or env-specified) and s3fs present, use S3.
      - Otherwise if local dir exists and has CSVs, return local path (Path object string).
    Returns a path string (either 's3://...' or local filesystem path) or None.
    """
    # 1) Try S3 first if RAW_CSV_DIR is S3-like
    if _is_s3_path(RAW_CSV_DIR) or _is_s3_path(S3_RAW_CSV_PATH):
        s3_latest = _s3_list_latest_csv(RAW_CSV_DIR)
        if s3_latest:
            return s3_latest

    # 2) Fallback to local files if directory exists and has CSVs
    try:
        local_dir = Path(RAW_CSV_DIR) if not _is_s3_path(RAW_CSV_DIR) else Path(LOCAL_RAW_DIR)
        if local_dir.exists() and local_dir.is_dir():
            candidates = list(local_dir.glob("*.csv"))
            if candidates:
                # prefer dated filenames then mtime
                def date_key(p):
                    m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", p.name)
                    if m:
                        try:
                            y, mo, d = map(int, m.groups())
                            return (y, mo, d, p.stat().st_mtime)
                        except Exception:
                            pass
                    return (0, 0, 0, p.stat().st_mtime)
                chosen = sorted(candidates, key=date_key, reverse=True)[0]
                return str(chosen)
    except Exception:
        pass

    # no files found
    return None

# ---- processing ------------------------------------------------------------
def process_csv(src_path: str, verbose: bool = False) -> str:
    """
    Read src_path (local path or s3://...) and write processed CSV to PROCESSED_DIR.
    Returns output path string.
    """
    src_str = str(src_path)
    is_s3 = _is_s3_path(src_str)
    # allow paths like 'nexo-storage-ca/..' to be treated as s3
    if not is_s3 and src_str.startswith("nexo-storage-ca/"):
        src_str = "s3://" + src_str.lstrip("/")
        is_s3 = True

    # read into pandas
    try:
        if is_s3:
            storage_options = {"anon": False}
            try:
                df = pd.read_csv(src_str, dtype=str, keep_default_na=False, low_memory=False, storage_options=storage_options)
            except Exception:
                df = pd.read_csv(src_str, dtype=str, keep_default_na=False, encoding="latin1", low_memory=False, storage_options=storage_options)
        else:
            try:
                df = pd.read_csv(src_str, dtype=str, keep_default_na=False, low_memory=False)
            except Exception:
                df = pd.read_csv(src_str, dtype=str, keep_default_na=False, encoding="latin1", low_memory=False)
    except Exception as e:
        print("ERROR reading CSV:", e)
        raise

    # normalize headers
    df.columns = [c.strip() for c in df.columns]

    if verbose:
        print("DEBUG: Raw columns:", list(df.columns))

    # drop currency columns (example)
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

    if "Market capitalization" in df.columns:
        out["mcap_rs_cr"] = (
            df["Market capitalization"].map(to_float).map(lambda v: round(v / 1e7, 2) if v else None)
        )
    else:
        out["mcap_rs_cr"] = None

    out["price"] = df.get("Price").map(to_float).map(lambda v: round(v, 2) if v else None)
    out["all_time_high"] = df.get("High All Time").map(to_float).map(lambda v: round(v, 2) if v else None)

    out["change_1d_pct"] = df.get("Price Change % 1 day").map(to_pct)
    out["change_1w_pct"] = df.get("Price Change % 1 week").map(to_pct)

    out["volume_24h_rs_cr"] = (
        df.get("Price * Volume (Turnover) 1 day", pd.Series()).map(to_float)
        .map(lambda v: round(v / 1e7, 2) if v else None)
    )
    out["vwap"] = df.get("Volume Weighted Average Price 1 day").map(to_float).map(lambda v: round(v, 2) if v else None)
    out["atr_pct"] = df.get("Average True Range % (14) 1 day").map(to_pct)
    out["volatility"] = df.get("Volatility 1 day").map(to_pct)
    out["vol_change_pct"] = df.get("Volume Change % 1 day").map(to_pct)
    out["relative_vol"] = df.get("Relative Volume 1 day").map(to_float).map(lambda v: round(v, 2) if v else None)

    before = len(out)
    out = out.drop_duplicates(subset=["symbol"], keep="first")
    after = len(out)
    if verbose and before != after:
        print(f"Deduplicated {before - after} rows")

    out = out.sort_values(by="mcap_rs_cr", ascending=False, na_position="last").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))

    safe_name = "processed_" + re.sub(r"[^A-Za-z0-9._-]", "_", Path(src_path).name)
    # decide output path
    if _is_s3_path(PROCESSED_DIR):
        out_path = f"{PROCESSED_DIR.rstrip('/')}/{safe_name}"
        try:
            out.to_csv(out_path, index=False, encoding="utf-8", storage_options={"anon": False})
        except Exception as e:
            print("ERROR writing processed CSV to S3:", e)
            raise
    else:
        out_dir = Path(PROCESSED_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / safe_name)
        try:
            out.to_csv(out_path, index=False, encoding="utf-8")
        except Exception as e:
            print("ERROR writing processed CSV locally:", e)
            raise

    print(f"Wrote processed CSV: {out_path} (rows={len(out)})")
    return out_path

# ---- CLI -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="Path to specific CSV file (local path or s3://...)", default=None)
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
