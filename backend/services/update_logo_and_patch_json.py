#!/usr/bin/env python3
"""
services/update_logo_and_patch_json.py

Usage:
  .venv/bin/python services/update_logo_and_patch_json.py --file PATH/TO/YESBANK.png

This:
 - copies the input image -> input_data/images/processed_images/processed_logos/<SYMBOL>_logo.<ext>
 - updates company_logo in JSON files under data/announcements whose symbol matches.
"""
from __future__ import annotations
import argparse, sys, json, shutil, re
from pathlib import Path

def derive_symbol_from_filename(name: str) -> str:
    stem = Path(name).stem
    # remove trailing "-logo" or "_logo" if present
    stem = re.sub(r'(?i)([-_]?logo)$', '', stem)
    # strip non-alnum and uppercase
    return re.sub(r'[^A-Za-z0-9]', '', stem).upper()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to image file to install as logo")
    ap.add_argument("--outdir", default="input_data/images/processed_images/processed_logos", help="Processed logos dir")
    args = ap.parse_args()

    img_path = Path(args.file)
    if not img_path.exists() or not img_path.is_file():
        print("ERROR: Image not found:", img_path, file=sys.stderr)
        sys.exit(2)

    base = Path.cwd()
    processed_dir = (base / args.outdir).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    symbol = derive_symbol_from_filename(img_path.name)
    if not symbol:
        print("ERROR: failed to derive symbol from filename:", img_path.name, file=sys.stderr)
        sys.exit(3)

    ext = "." + img_path.suffix.lstrip(".").lower()
    dst_name = f"{symbol}_logo{ext}"
    dst_path = processed_dir / dst_name

    try:
        shutil.copy2(str(img_path), str(dst_path))
        print("Copied image to:", dst_path)
    except Exception as e:
        print("ERROR: failed to copy image:", e, file=sys.stderr)
        sys.exit(4)

    static_rel = f"/static/images/processed_logos/{dst_name}"

    ann_dir = (base / "data" / "announcements")
    if not ann_dir.exists():
        print("ERROR: announcements dir missing:", ann_dir, file=sys.stderr)
        sys.exit(5)

    checked = 0
    updated = 0
    for p in sorted(ann_dir.rglob("*.json")):
        checked += 1
        try:
            txt = p.read_text(encoding="utf-8")
            j = json.loads(txt)
        except Exception:
            # skip invalid json
            continue
        if not isinstance(j, dict):
            continue
        # derive candidate symbol from JSON (canonical_symbol or symbol)
        cand = ""
        if isinstance(j.get("canonical_symbol"), str) and j.get("canonical_symbol").strip():
            cand = j.get("canonical_symbol").strip().upper()
        elif isinstance(j.get("symbol"), str) and j.get("symbol").strip():
            cand = j.get("symbol").strip().upper()

        if cand == symbol:
            if j.get("company_logo") != static_rel:
                j["company_logo"] = static_rel
                try:
                    p.write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")
                    updated += 1
                    print("Updated:", p.relative_to(base), "->", static_rel)
                except Exception as e:
                    print("ERROR: failed to write", p, ":", e, file=sys.stderr)

    print(f"Done. scanned={checked} json files, updated={updated}.")
    if updated == 0:
        print("Note: no JSON matched the symbol. Check canonical_symbol/symbol fields in your JSONs.")

if __name__ == "__main__":
    main()
