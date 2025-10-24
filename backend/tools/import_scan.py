#!/usr/bin/env python3
# tools/import_scan.py
import importlib
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # project root (NEXO/)
SERVICES_DIR = ROOT / "backend" / "services"

if not SERVICES_DIR.exists():
    print("ERROR: services dir not found at", SERVICES_DIR)
    sys.exit(2)

sys.path.insert(0, str(ROOT))

py_files = sorted([p for p in SERVICES_DIR.glob("*.py") if p.is_file() and p.name != "__init__.py"])

ordered = []
img = "image_processor.py"
if (SERVICES_DIR / img).exists():
    ordered.append(SERVICES_DIR / img)
for p in py_files:
    if p.name == img:
        continue
    ordered.append(p)

failures = []
print("=== Import scan starting ===")
for p in ordered:
    mod_name = p.stem
    fq = f"backend.services.{mod_name}"
    print(f"\nImporting {fq} ...", end=" ")
    try:
        if fq in sys.modules:
            importlib.reload(sys.modules[fq])
        else:
            importlib.import_module(fq)
        print("OK")
    except Exception:
        print("FAILED")
        tb = traceback.format_exc()
        print(tb)
        failures.append((fq, tb))

print("\n=== Summary ===")
print(f"Total modules attempted: {len(ordered)}")
print(f"Failures: {len(failures)}")
if failures:
    for fq, tb in failures:
        print(f"\n---- {fq} FAILED ----")
        print(tb)
    sys.exit(1)
else:
    print("All imports succeeded ✅")
    sys.exit(0)
