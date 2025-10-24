#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=".."
export S3_PROCESSED_CSV_PATH="s3://nexo-storage-ca/input_data/csv/processed_csv"
export S3_STATIC_CSV_PATH="s3://nexo-storage-ca/input_data/csv/static"

echo "1) find_latest_processed_eod:"
python3 - <<'PY'
from backend.services import csv_utils
print("FOUND:", repr(csv_utils.find_latest_processed_eod()))
PY

echo "2) load_processed_df:"
python3 - <<'PY'
from backend.services import csv_utils
print("load:", csv_utils.load_processed_df() is not None)
PY

echo "3) sample S3 read:"
python3 - <<'PY'
from backend.services import csv_utils
import pandas as pd, s3fs
p = csv_utils.find_latest_processed_eod()
if p:
    fs = s3fs.S3FileSystem(anon=False)
    with fs.open(str(p),'rb') as f:
        df = pd.read_csv(f, nrows=3)
    print("rows:", len(df))
else:
    echo "no p"
PY

echo "4) get_market_snapshot RELIANCE:"
python3 - <<'PY'
from backend.services import csv_utils
print(csv_utils.get_market_snapshot("RELIANCE"))
PY

echo "5) CLI JSON:"
.venv/bin/python services/csv_utils.py --symbol RELIANCE --json

echo "SMOKE TESTS COMPLETED"
