#!/usr/bin/env python3
"""
update_market_snapshots.py
Utility script to refresh market_snapshot in all announcement JSON files,
based on the latest processed EOD CSV.
"""

import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "input_data" / "csv" / "processed_csv" / "processed_app_based_eod_2025-09-24.csv"
ANN_DIR = BASE_DIR / "data" / "announcements"

def load_market_data():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return {}
    df = pd.read_csv(CSV_PATH)
    return {row["symbol"]: row.to_dict() for _, row in df.iterrows()}

def update_jsons(market_data: dict):
    for json_file in ANN_DIR.rglob("*.json"):
        with open(json_file, "r+", encoding="utf-8") as f:
            data = json.load(f)
        symbol = data.get("canonical_symbol") or data.get("symbol")
        if symbol and symbol in market_data:
            data["market_snapshot"] = market_data[symbol]
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Updated market data in {json_file}")

if __name__ == "__main__":
    market_data = load_market_data()
    if not market_data:
        print("No market data found, exiting.")
    else:
        update_jsons(market_data)
