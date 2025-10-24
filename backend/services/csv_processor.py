#!/usr/bin/env python3
"""
backend/services/csv_processor.py

Behaviour:
 - Automatically prefers S3 (if configured and reachable via fsspec)
 - Falls back silently to local CSV folders when S3 unavailable or missing
 - Uses .env S3_* paths for cloud awareness (no manual switch)
 - Produces processed CSV to S3 or local paths consistently

Environment variables used:
 - S3_RAW_CSV_PATH
 - S3_PROCESSED_CSV_PATH
"""

from __future__ import annotations
import os
import re
import sys
import argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

# Optional dependency
try:
    import fsspec
except Exception:
    fsspec = None

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
LOCAL_RAW_DIR = BASE / "input_data" / "csv" / "eod_csv"
LOCAL_PROCESSED_DIR = BASE / "input_data" / "csv" / "processed_csv"

S3_RAW_CSV_PATH = os.getenv("S3_RAW_CSV_PATH", "").strip()
S3_PROCESSED_CSV_PATH = os.getenv("S3_PROCESSED_CSV_PATH", "").strip()

RAW_CSV_DIR = S3_RAW_CSV_PATH or str(LOCAL_RAW_DIR)
PROCESSED_DIR = S3_PROCESSED_CSV_PATH or str(LOCAL_PROCESSED_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.lower().startswith("s3://")


def _list_s3_csvs(s3_prefix: str) -> list[str]:
    """Return list of CSV keys under an S3 prefix using fsspec (silent fallback)."""
    if not fsspec or not _is_s3_uri(s3_prefix):
        return []
    try:
        fs = fsspec.filesystem("s3")
        objs = fs.glob(f"{s3_prefix.rstrip('/')}/*.csv")
        return sorted(str(o) if str(o).startswith("s3://") else f"s3://{o}" for o in objs)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Core: Discover latest CSV (S3-first)
# ---------------------------------------------------------------------------
def find_latest_csv() -> str | None:
    """
    Find the latest CSV file automatically:
      1️⃣ Try S3 first if configured
      2️⃣ Fallback to local directory if S3 fails or empty
    """
    # --- Try S3 first ---
    if _is_s3_uri(RAW_CSV_DIR):
        s3_candidates = _list_s3_csvs(RAW_CSV_DIR)
        if s3_candidates:
            chosen = s3_candidates[-1]
            print(f"✅ Found latest CSV on S3: {chosen}")
            return chosen
        else:
            print("ℹ️ No S3 CSVs found — falling back to local folder.")

    # --- Local fallback ---
    local_dir = Path(LOCAL_RAW_DIR)
    if local_dir.exists() and local_dir.is_dir():
        candidates = list(local_dir.glob("*.csv"))
        if candidates:
            def date_key(p: Path):
                m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", p.name)
                if m:
                    try:
                        y, mo, d = map(int, m.groups())
                        return (y, mo, d, p.stat().st_mtime)
                    except Exception:
                        pass
                return (0, 0, 0, p.stat().st_mtime)
            chosen = sorted(candidates, key=date_key, reverse=True)[0]
            print(f"✅ Using local CSV: {chosen}")
            return str(chosen)

    print("⚠️ No CSV files found (S3 or local).")
    return None


# ---------------------------------------------------------------------------
# CSV Processor
# ---------------------------------------------------------------------------
def process_csv(src_path: str, verbose: bool = False) -> str:
    """
    Process a CSV file (local or S3) into the processed directory.
    Automatically writes to S3 if configured, otherwise local.
    """
    src_str = str(src_path)
    is_s3 = _is_s3_uri(src_str)

    try:
        if fsspec and is_s3:
            df = pd.read_csv(src_str, dtype=str, keep_default_na=False, storage_options={"anon": False})
        else:
            df = pd.read_csv(src_str, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        raise

    # --- Normalize Columns ---
    df.columns = [c.strip() for c in df.columns]
    if verbose:
        print("Columns:", list(df.columns))

    def _to_float(x):
        try:
            if x in ("", None):
                return None
            return float(str(x).replace(",", "").strip())
        except Exception:
            return None

    def _to_pct(x):
        try:
            if x in ("", None):
                return None
            return round(float(str(x).replace(",", "").strip()), 2)
        except Exception:
            return None

    out = pd.DataFrame()
    out["symbol"] = df.get("Symbol")
    out["company_name"] = df.get("Description")

    out["mcap_rs_cr"] = df.get("Market capitalization", pd.Series()).map(_to_float).map(
        lambda v: round(v / 1e7, 2) if v else None
    )
    out["price"] = df.get("Price").map(_to_float)
    out["change_1d_pct"] = df.get("Price Change % 1 day").map(_to_pct)
    out["change_1w_pct"] = df.get("Price Change % 1 week").map(_to_pct)
    out["vwap"] = df.get("Volume Weighted Average Price 1 day").map(_to_float)
    out["mcap_rs_cr"] = out["mcap_rs_cr"].fillna(0)

    out = out.drop_duplicates(subset=["symbol"], keep="first").sort_values(
        by="mcap_rs_cr", ascending=False
    ).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))

    safe_name = "processed_" + re.sub(r"[^A-Za-z0-9._-]", "_", Path(src_path).name)
    output_path = f"{PROCESSED_DIR.rstrip('/')}/{safe_name}"

    # --- Write processed file (S3 or local fallback) ---
    try:
        if fsspec and _is_s3_uri(PROCESSED_DIR):
            fs = fsspec.filesystem("s3")
            with fs.open(output_path, "w") as f:
                out.to_csv(f, index=False, encoding="utf-8")
            print(f"✅ Processed CSV uploaded to S3: {output_path}")
        else:
            Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
            out.to_csv(output_path, index=False, encoding="utf-8")
            print(f"✅ Processed CSV saved locally: {output_path}")
    except Exception as e:
        print(f"⚠️ Failed to write processed CSV: {e}")
        raise

    return output_path


# ---------------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Process latest or given EOD CSV (S3-aware).")
    parser.add_argument("--src", help="Optional path to CSV (S3 or local). If omitted, auto-detects latest.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    csv_path = args.src or find_latest_csv()
    if not csv_path:
        print("No CSV source found. Exiting.")
        sys.exit(1)

    process_csv(csv_path, verbose=args.verbose)


if __name__ == "__main__":
    main()