#!/usr/bin/env python3
"""
regen_announcement.py

Usage:
  python3 backend/tools/regen_announcement.py --path <path/to/ann.json>
  python3 backend/tools/regen_announcement.py --id <announcement_id>

What it does:
 - Loads the specified announcement JSON.
 - Attempts safe, deterministic fixes for missing fields:
    * announcement_datetime_iso <- parsed from announcement_datetime_human if missing
    * summary_60 <- from summary_raw or headline_ai (truncate ~60 words) if missing
    * banner_exists <- checks backend/input_data/images/processed_images/processed_banners/<banner_image> or <canonical_symbol>_banner.png
    * For missing market_snapshot.mcap_rs_cr: adds a 'ms_fix_note' flag (does not guess)
 - Writes back the file if changes were made and prints a changelog.

Note: This script is conservative and non-destructive. Review changes in git/diff before committing.
"""
import argparse, json, os, glob, sys, re
from datetime import datetime
try:
    from dateutil import parser as dateparser
except Exception:
    dateparser = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANN_DIR = os.path.join(BASE_DIR, "data", "announcements")
BANNER_DIR = os.path.join(BASE_DIR, "input_data", "images", "processed_images", "processed_banners")

def find_file_by_id(ann_id):
    pattern = os.path.join(ANN_DIR, "**", f"*{ann_id}*.json")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None

def safe_load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def parse_human_datetime(human):
    if not human:
        return None
    # Try dateutil if available
    if dateparser:
        try:
            dt = dateparser.parse(human)
            return dt.isoformat()
        except Exception:
            pass
    # Fallback: common patterns e.g., "02 Oct 2025, 12:45 PM"
    try:
        # remove commas
        s = re.sub(r",", "", human)
        dt = datetime.strptime(s, "%d %b %Y %I:%M %p")
        return dt.isoformat()
    except Exception:
        return None

def truncate_to_words(text, word_limit=60):
    if not text:
        return None
    words = re.findall(r"\S+", text)
    if len(words) <= word_limit:
        return text.strip()
    return " ".join(words[:word_limit]).strip() + "..."

def check_banner_exists(obj):
    # Prefer explicit banner_image field
    banner = obj.get("banner_image")
    if banner:
        # if absolute or relative path, try to check local file
        basename = os.path.basename(banner)
        candidate = os.path.join(BANNER_DIR, basename)
        return os.path.exists(candidate)
    # fallback to canonical_symbol_banner.png
    sym = obj.get("canonical_symbol") or obj.get("symbol")
    if sym:
        candidate = os.path.join(BANNER_DIR, f"{sym}_banner.png")
        if os.path.exists(candidate):
            # also set banner_image relative path for convenience
            obj.setdefault("banner_image", os.path.join("input_data","images","processed_images","processed_banners", f"{sym}_banner.png"))
            return True
    return False

def regen_file(path):
    changes = []
    obj = safe_load(path)
    # 1. announcement_datetime_iso
    if not obj.get("announcement_datetime_iso") and obj.get("announcement_datetime_human"):
        parsed = parse_human_datetime(obj.get("announcement_datetime_human"))
        if parsed:
            obj["announcement_datetime_iso"] = parsed
            changes.append("SET announcement_datetime_iso from announcement_datetime_human")
    # 2. summary_60
    if not obj.get("summary_60"):
        # try summary_raw
        s = obj.get("summary_raw") or obj.get("summary_ai") or obj.get("headline_ai") or obj.get("headline_final")
        if s:
            truncated = truncate_to_words(s, 60)
            obj["summary_60"] = truncated
            changes.append("FILLED summary_60 from summary_raw/headline (truncated)")
    # 3. banner_exists
    if "banner_exists" not in obj or obj.get("banner_exists") is None:
        exists = check_banner_exists(obj)
        obj["banner_exists"] = bool(exists)
        changes.append(f"SET banner_exists={exists}")
    # 4. market_snapshot checks
    ms = obj.get("market_snapshot")
    if not isinstance(ms, dict):
        obj["market_snapshot"] = {}
        ms = obj["market_snapshot"]
        changes.append("CREATED empty market_snapshot dict")
    # If mcap missing, add a fix note (we don't guess numeric values)
    if "mcap_rs_cr" not in ms or ms.get("mcap_rs_cr") in (None, ""):
        ms.setdefault("ms_fix_note", "mcap_rs_cr_missing: manual review required")
        changes.append("ADD ms_fix_note for missing mcap_rs_cr")
    # write back if changes
    if changes:
        safe_write(path, obj)
    return changes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to announcement JSON file (absolute or relative)")
    parser.add_argument("--id", help="Announcement id (will search files containing this id)")
    args = parser.parse_args()
    if not args.path and not args.id:
        print("Error: supply --path <file> or --id <announcement_id>")
        sys.exit(1)
    path = args.path
    if not path and args.id:
        found = find_file_by_id(args.id)
        if not found:
            print("ERROR: No file found for id:", args.id)
            sys.exit(2)
        path = found
    if not os.path.exists(path):
        print("ERROR: file not found:", path)
        sys.exit(3)
    print("Processing:", path)
    changes = regen_file(path)
    if changes:
        print("Changes applied:")
        for c in changes:
            print(" -", c)
        print("File updated:", path)
    else:
        print("No changes necessary for:", path)

if __name__ == "__main__":
    main()
