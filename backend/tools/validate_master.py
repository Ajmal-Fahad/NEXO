#!/usr/bin/env python3
"""
validate_master.py
Scan announcement JSON files under backend/data/announcements and validate required fields.
Usage: python3 backend/tools/validate_master.py
Exits with code 0 if no errors, 2 if issues found.
"""

import os, json, sys, glob
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")
ANN_DIR = os.path.abspath(os.path.join(ROOT, "data", "announcements"))

# Top-level required keys in announcement JSON (subset relevant to UI)
REQUIRED_TOP = [
    "id",
    "canonical_symbol",
    "canonical_company_name",
    "summary_60",
    "announcement_datetime_iso",
    # either banner_image or banner_exists should be present (checked separately)
]

# Required keys inside market_snapshot
REQUIRED_MS = [
    "symbol",
    "price",
    "change_1d_pct",
    "mcap_rs_cr",
    "all_time_high",
    "market_snapshot_date",
    "rank"
]

errors = defaultdict(list)
file_count = 0
checked_files = []

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"__load_error__": str(e)}

# Walk announcement subfolders, support nested year/date structure
pattern = os.path.join(ANN_DIR, "**", "*.json")
all_files = sorted(glob.glob(pattern, recursive=True))

if not all_files:
    print("[WARN] No announcement JSON files found under:", ANN_DIR)
    sys.exit(0)

for p in all_files:
    file_count += 1
    checked_files.append(p)
    obj = load_json(p)
    if "__load_error__" in obj:
        errors[p].append("JSON_LOAD_ERROR: " + obj["__load_error__"])
        continue

    # Top-level required fields
    for k in REQUIRED_TOP:
        if k not in obj or obj.get(k) in (None, ""):
            errors[p].append(f"MISSING_TOP:{k}")

    # banner_image or banner_exists check
    if not obj.get("banner_image") and ("banner_exists" not in obj):
        errors[p].append("MISSING_BANNER_INFO: banner_image AND banner_exists absent")
    else:
        # if banner_exists is True but banner_image missing -> warning
        if obj.get("banner_exists") is True and not obj.get("banner_image"):
            errors[p].append("BANNER_EXISTS_TRUE_BUT_NO_banner_image")

    # market_snapshot existence
    ms = obj.get("market_snapshot")
    if not isinstance(ms, dict):
        errors[p].append("MISSING_MARKET_SNAPSHOT")
    else:
        for k in REQUIRED_MS:
            if k not in ms or ms.get(k) in (None, ""):
                errors[p].append(f"MISSING_MS:{k}")

    # basic type/content sanity checks (non-exhaustive)
    try:
        # price numeric?
        if isinstance(ms, dict):
            price = ms.get("price")
            if price is not None:
                try:
                    _ = float(price)
                except:
                    errors[p].append("INVALID_MS:price_not_numeric")
            # change_1d_pct numeric?
            ch = ms.get("change_1d_pct")
            if ch is not None:
                try:
                    _ = float(ch)
                except:
                    errors[p].append("INVALID_MS:change_1d_pct_not_numeric")
    except Exception:
        pass

# Summarize results
total_errors = sum(len(v) for v in errors.values())
print("Validation run summary")
print("----------------------")
print("Announcement files checked:", file_count)
print("Files with errors:", len(errors))
print("Total error items:", total_errors)
print("")

if errors:
    # Show up to first 50 problematic files with their issues
    shown = 0
    for f, issues in errors.items():
        print(f"FILE: {f}")
        for iss in issues:
            print("   -", iss)
        shown += 1
        if shown >= 50:
            break
    print("")
    print("Tip: run this script again after fixing files. Use tools/regen_announcement.py to regenerate single files (if available).")
    sys.exit(2)

print("No issues found. All checked announcement files contain the required keys.")
sys.exit(0)
