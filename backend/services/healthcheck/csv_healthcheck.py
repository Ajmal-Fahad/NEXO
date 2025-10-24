#!/usr/bin/env python3
"""
Healthcheck for CSV pipeline (csv_processor/csv_utils).
Verifies .env, S3 access for RAW/PROCESSED CSV paths and that at least one CSV exists.
Exits 0 on success, non-zero on failure.
"""
import os, sys, importlib
from pathlib import Path
from dotenv import load_dotenv

# load .env from repo root
env_path = Path(__file__).resolve().parents[2] / ".env"
if env_path.exists():
    load_dotenv(env_path)

# prefer RAW then PROCESSED for existence check
s3_raw = os.getenv("S3_RAW_CSV_PATH", "").strip()
s3_proc = os.getenv("S3_PROCESSED_CSV_PATH", "").strip()

s3_target = s3_raw or s3_proc
if not s3_target:
    print("⚠️  No S3 CSV path configured (S3_RAW_CSV_PATH / S3_PROCESSED_CSV_PATH). Assuming local-only mode.")
    sys.exit(0)

# check for fsspec / s3fs
try:
    fsspec = importlib.import_module("fsspec")
except Exception as e:
    print(f"❌ fsspec not installed: {e}")
    sys.exit(2)

try:
    fs = fsspec.filesystem("s3")
except Exception as e:
    print(f"❌ Failed to instantiate s3 filesystem: {e}")
    sys.exit(3)

prefix = s3_target.rstrip("/")
try:
    # prefer a glob search for CSVs under prefix
    pattern = prefix + "/*.csv"
    matched = fs.glob(pattern)
    if not matched:
        # also try listing the prefix itself
        exists = fs.exists(prefix)
        if not exists:
            print(f"❌ S3 CSV prefix not found or inaccessible: {prefix}")
            sys.exit(4)
        # prefix exists but no csv found
        print(f"❌ No CSV files found under: {prefix}")
        sys.exit(5)
    print(f"✅ S3 CSV path OK — found {len(matched)} CSV(s) under: {prefix}")
    sys.exit(0)
except Exception as e:
    print(f"❌ S3 check failed for {prefix}: {e}")
    sys.exit(6)
