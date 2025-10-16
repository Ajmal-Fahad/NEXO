#!/usr/bin/env python3
"""
latest_marketdata_json.py

Generates two synchronized files:
 - latest_marketdata.json  (pretty, human-readable)
 - latest_marketdata_min.json  (minified, for frontend use)
Also keeps rolling 3-day backups for both versions.
"""

from __future__ import annotations
import sys, os, json, shutil, glob, importlib.util
from pathlib import Path
from datetime import datetime
from typing import Any

# Base backend directories
BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "data" / "master_data"
OUT_PATH = OUT_DIR / "latest_marketdata.json"
OUT_PATH_MIN = OUT_DIR / "latest_marketdata_min.json"


# ---------------- Robust import of csv_utils ----------------
def import_csv_utils():
    try:
        from backend.services import csv_utils as cu  # type: ignore
        return cu
    except Exception:
        pass

    try:
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))
        from services import csv_utils as cu  # type: ignore
        return cu
    except Exception:
        pass

    try:
        fpath = BASE_DIR / "services" / "csv_utils.py"
        if fpath.exists():
            spec = importlib.util.spec_from_file_location("csv_utils_local", str(fpath))
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)  # type: ignore
            return module
    except Exception:
        pass

    return None


cu = import_csv_utils()
if cu is None:
    print("❌ ERROR: Unable to import backend.services.csv_utils.", file=sys.stderr)
    sys.exit(2)


# ---------------- Helpers ----------------
def _make_serializable(v: Any) -> Any:
    """Convert pandas/numpy scalars and unusual objects to JSON-safe types."""
    try:
        pd = getattr(cu, "pd", None)
        if pd is not None and pd.isna(v):
            return None
    except Exception:
        pass

    try:
        import numpy as _np
        if isinstance(v, getattr(_np, "generic", ())):
            return v.item()
    except Exception:
        pass

    try:
        if hasattr(v, "item") and not isinstance(v, (dict, list, str, bytes)):
            return v.item()
    except Exception:
        pass

    if isinstance(v, (str, bool, int, float)) or v is None:
        return v

    if isinstance(v, dict):
        return {k: _make_serializable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_make_serializable(x) for x in v]

    try:
        return str(v)
    except Exception:
        return None


def _normalize_records(records):
    return [_make_serializable(r) for r in records]


# ---------------- Main Flow ----------------
def main():
    print("\n🔍 Locating latest processed EOD CSV...")
    try:
        latest_csv = cu.find_latest_processed_eod()
    except Exception as e:
        print(f"❌ cu.find_latest_processed_eod() raised: {e}", file=sys.stderr)
        latest_csv = None

    if not latest_csv:
        print("❌ No processed CSV found.", file=sys.stderr)
        sys.exit(1)

    latest_csv = Path(latest_csv)
    print(f"✅ Using CSV: {latest_csv}")

    try:
        df = cu.load_processed_df(force_reload=True)
    except TypeError:
        try:
            df = cu.load_processed_df()
        except Exception:
            df = getattr(cu, "_EOD_DF", None)
    except Exception:
        df = getattr(cu, "_EOD_DF", None)

    records = []
    if df is not None:
        try:
            # Convert numeric NaN -> None for clean JSON
            df = df.where(cu.pd.notnull(df), None)
            for col in df.columns:
                if cu.pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].apply(
                        lambda x: None if (x is None or str(x).lower() in ("nan", "none", "")) else x
                    )
            records_raw = df.to_dict(orient="records")
            records = _normalize_records(records_raw)
        except Exception as e:
            print(f"⚠️ Warning: converting DataFrame to records failed: {e}", file=sys.stderr)
            records = []

    if not records:
        print("❌ No market records produced.", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------- Backup rotation for both files --------------
    try:
        date_stamp = datetime.now().strftime("%Y%m%d")
        if OUT_PATH.exists():
            shutil.copy2(OUT_PATH, OUT_DIR / f"latest_marketdata_{date_stamp}.json")
            print(f"🗄️  Backup created: latest_marketdata_{date_stamp}.json")
        if OUT_PATH_MIN.exists():
            shutil.copy2(OUT_PATH_MIN, OUT_DIR / f"latest_marketdata_min_{date_stamp}.json")
            print(f"🗄️  Backup created: latest_marketdata_min_{date_stamp}.json")

        backups = sorted(
            [p for p in OUT_DIR.glob("latest_marketdata_*.json")],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        backups_min = sorted(
            [p for p in OUT_DIR.glob("latest_marketdata_min_*.json")],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        for old in backups[3:]:
            try:
                old.unlink()
            except Exception:
                pass
        for old in backups_min[3:]:
            try:
                old.unlink()
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ Backup rotation failed: {e}", file=sys.stderr)

    # ------------- Write new JSONs --------------
    payload = {
        "source_csv": str(latest_csv),
        "record_count": len(records),
        "generated_at": datetime.now().isoformat(),
        "data": records,
    }

    try:
        with open(OUT_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        with open(OUT_PATH_MIN, "w", encoding="utf-8") as fh_min:
            json.dump(payload, fh_min, ensure_ascii=False, separators=(",", ":"))
    except Exception as e:
        print(f"❌ Failed writing output JSON: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n✅ Wrote latest market JSON: {OUT_PATH}")
    print(f"⚡ Minified JSON: {OUT_PATH_MIN}")
    print(f"📈 Records exported: {len(records)}")
    print(f"🕒 generated_at: {payload['generated_at']}\n")


if __name__ == "__main__":
    main()
