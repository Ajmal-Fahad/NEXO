#!/usr/bin/env python3
"""
csv_utils.py - helper for processed EOD CSV + indices lookup (dynamic CLI)

Usage examples (from backend/):
  .venv/bin/python services/csv_utils.py --symbol NDTV
  .venv/bin/python services/csv_utils.py --list
  .venv/bin/python services/csv_utils.py --symbol RELIANCE --json

Expectations:
 - Processed EOD CSVs in: input_data/csv/processed_csv/
 - Indices CSV in:         input_data/csv/static/
 - Processed CSV should have canonical columns (but we try to be permissive).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Try to import s3fs/fsspec lazily inside functions; don't fail import here if missing.
# Keep base paths relative to the project.
BASE = Path(__file__).resolve().parents[1]

LOCAL_PROCESSED_DIR = BASE / "input_data" / "csv" / "processed_csv"
LOCAL_STATIC_DIR = BASE / "input_data" / "csv" / "static"

# Environment overrides (allow empty -> None)
_s3_proc_env = os.getenv("S3_PROCESSED_CSV_PATH", "")
S3_PROCESSED_CSV_PATH: Optional[str] = _s3_proc_env.strip() or None
_s3_static_env = os.getenv("S3_STATIC_CSV_PATH", "")
S3_STATIC_CSV_PATH: Optional[str] = _s3_static_env.strip() or None

# Keep PROCESSED_DIR/STATIC_DIR as either Path or string (for S3).
if S3_PROCESSED_CSV_PATH and S3_PROCESSED_CSV_PATH.lower().startswith("s3://"):
    PROCESSED_S3 = True
    PROCESSED_DIR: Union[str, Path] = S3_PROCESSED_CSV_PATH.rstrip("/")
else:
    PROCESSED_S3 = False
    PROCESSED_DIR = LOCAL_PROCESSED_DIR

if S3_STATIC_CSV_PATH and S3_STATIC_CSV_PATH.lower().startswith("s3://"):
    STATIC_S3 = True
    STATIC_DIR: Union[str, Path] = S3_STATIC_CSV_PATH.rstrip("/")
else:
    STATIC_S3 = False
    STATIC_DIR = LOCAL_STATIC_DIR

# caches
_EOD_DF: Optional[pd.DataFrame] = None
_EOD_PATH: Optional[Union[str, Path]] = None
_INDICES_DF: Optional[pd.DataFrame] = None
_INDICES_PATH: Optional[Union[str, Path]] = None

# Date regex matches: 2025-09-24 or 20250924 or 2025_09_24
DATE_RE = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")

# broad index priority (used to pick a preferred broad index from indices CSV)
BROAD_PRIORITY = [
    "NIFTY50",
    "NIFTYNEXT50",
    "NIFTY100",
    "NIFTY200",
    "NIFTY500",
    "NIFTYTOTALMARKET",
    "NIFTYMIDCAP50",
    "NIFTYMIDCAP150",
]


def _log(*args: Any, **kwargs: Any) -> None:
    """Helper to print logs to stderr."""
    print(*args, file=sys.stderr, **kwargs)


# ---- helpers ----------------------------------------------------------------
def _ensure_s3_uri(candidate: Union[str, Path]) -> str:
    """Return a normalized s3://... URI string for a candidate that may be a bare key."""
    s = str(candidate)
    s = s.strip().lstrip("/")
    if not s.lower().startswith("s3://"):
        return "s3://" + s
    return s


def _filename_from_path(p: Union[str, Path]) -> str:
    """Return the filename portion from a pathlib.Path or an s3://... string."""
    if hasattr(p, "name"):
        # pathlib.Path-like
        return p.name
    s = str(p).rstrip("/")
    return s.split("/")[-1]


# ---- file discovery / loading ----------------------------------------------
def find_latest_processed_eod() -> Optional[Union[str, Path]]:
    """
    Return newest processed_*.csv in PROCESSED_DIR or S3 path set by env var.

    If an S3 env is configured, this will attempt to list that S3 location
    (via fsspec) and return a full 's3://...' URI string for the newest match.
    Otherwise it returns a pathlib.Path to the local file.
    """
    # prefer explicit env var if present (robust to missing scheme)
    s3_env = (os.getenv("S3_PROCESSED_CSV_PATH") or "").strip()
    if s3_env and not s3_env.lower().startswith("s3://") and "nexo-storage-ca" in s3_env:
        s3_env = "s3://" + s3_env.lstrip("/")

    if s3_env:
        try:
            # import lazily so module import doesn't fail when fsspec missing
            import fsspec  # type: ignore

            fs = fsspec.filesystem("s3")
            prefix = s3_env.rstrip("/")
            # look for processed_*.csv first, then any .csv as fallback
            candidates = fs.glob(f"{prefix}/processed_*.csv") or fs.glob(f"{prefix}/*.csv")
            if not candidates:
                return None

            # try to parse dates from filenames and pick the latest date if present
            dated: List[Tuple[pd.Timestamp, str]] = []
            for key in candidates:
                name = key.split("/")[-1]
                m = DATE_RE.search(name)
                if m:
                    try:
                        y, mo, d = map(int, m.groups())
                        ts = pd.Timestamp(year=y, month=mo, day=d)
                        dated.append((ts, key))
                    except Exception:
                        # ignore parse errors on filename
                        continue
            if dated:
                dated.sort(key=lambda x: x[0], reverse=True)
                ret = dated[0][1]
                return _ensure_s3_uri(ret)

            # fallback: lexicographically newest candidate
            ret = sorted(candidates)[-1]
            return _ensure_s3_uri(ret)
        except Exception as exc:  # pragma: no cover - defensive
            _log("S3 listing failed:", exc)
            # fall through to local checks

    # Local filesystem fallback
    local_processed_dir = Path(PROCESSED_DIR) if isinstance(PROCESSED_DIR, str) else PROCESSED_DIR
    if not local_processed_dir.exists():
        return None

    candidates = list(local_processed_dir.glob("processed_*.csv"))
    if not candidates:
        candidates = list(local_processed_dir.glob("*.csv"))
        if not candidates:
            return None

    dated_local: List[Tuple[pd.Timestamp, Path]] = []
    for p in candidates:
        m = DATE_RE.search(p.name)
        if m:
            try:
                y, mo, d = map(int, m.groups())
                ts = pd.Timestamp(year=y, month=mo, day=d)
                dated_local.append((ts, p))
            except Exception:
                continue
    if dated_local:
        dated_local.sort(key=lambda x: x[0], reverse=True)
        return dated_local[0][1]

    # fallback to newest modified file
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_processed_df(force_reload: bool = False) -> Optional[pd.DataFrame]:
    """
    Load and cache the latest processed EOD CSV into a pandas DataFrame.

    Supports both local Path and s3://... string returned by find_latest_processed_eod().
    """
    global _EOD_DF, _EOD_PATH
    if _EOD_DF is not None and not force_reload:
        return _EOD_DF

    p = find_latest_processed_eod()
    if not p:
        _EOD_DF = None
        _EOD_PATH = None
        return None

    _log(f"Loading processed EOD CSV: {p}")

    try:
        # s3 URI handling
        if isinstance(p, str) and p.startswith("s3://"):
            storage_opts = {"anon": False}
            try:
                # try modern pandas with storage_options
                df = pd.read_csv(p, low_memory=False, storage_options=storage_opts)
            except TypeError:
                # older pandas; open via fsspec file-like
                import fsspec  # type: ignore

                fs = fsspec.filesystem("s3")
                with fs.open(p, "rb") as fh:
                    df = pd.read_csv(fh, low_memory=False)
        else:
            # local file path (Path)
            try:
                df = pd.read_csv(p, low_memory=False)
            except Exception:
                # fallback encoding
                df = pd.read_csv(p, encoding="latin1", low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive
        _log("Primary read failed:", exc)
        # final fallback using latin1 and fsspec if needed
        try:
            if isinstance(p, str) and p.startswith("s3://"):
                import fsspec  # type: ignore

                fs = fsspec.filesystem("s3")
                with fs.open(p, "rb") as fh:
                    df = pd.read_csv(fh, encoding="latin1", low_memory=False)
            else:
                df = pd.read_csv(p, encoding="latin1", low_memory=False)
        except Exception as exc2:  # pragma: no cover - defensive
            _log("Final read attempt failed:", exc2)
            _EOD_DF = None
            _EOD_PATH = None
            return None

    # tidy headers and cache
    df.columns = [c.strip() for c in df.columns]
    _EOD_DF = df
    _EOD_PATH = p
    return _EOD_DF


def _extract_date_from_filename(path: Optional[Union[str, Path]]) -> Optional[str]:
    """
    Return date string like '24-Sep-2025' extracted from filename, or None.

    Accepts a pathlib.Path or a string (including s3://... URIs) and is robust to both.
    """
    if not path:
        return None
    try:
        name = _filename_from_path(path)
    except Exception:
        return None
    if not name:
        return None
    m = DATE_RE.search(name)
    if not m:
        return None
    try:
        y, mo, d = map(int, m.groups())
        return pd.Timestamp(year=y, month=mo, day=d).strftime("%d-%b-%Y")
    except Exception:
        return None


def _normalize_value(v: Any) -> Any:
    """Return None for NA-like values, otherwise the original value."""
    if pd.isna(v):
        return None
    return v


# ---- market snapshot --------------------------------------------------------
def get_market_snapshot(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Lookup a symbol or company text in the processed EOD DataFrame.
    Returns a dict with canonical keys and market_snapshot_date extracted from filename.
    """
    if not symbol:
        return None

    df = load_processed_df()
    if df is None:
        return None

    s = str(symbol).strip().upper()

    # heuristics for column names
    symbol_cols = [c for c in df.columns if c.strip().lower() in ("symbol", "sym", "ticker")]
    comp_cols = [c for c in df.columns if c.strip().lower() in ("company_name", "company", "description", "company name")]

    rows = pd.DataFrame()
    if symbol_cols:
        for sc in symbol_cols:
            try:
                rows = df[df[sc].astype(str).str.upper() == s]
            except Exception:
                rows = df[df[sc].astype(str).str.upper() == s]
            if not rows.empty:
                break

    # fallback: partial match on company name/description
    if rows.empty and comp_cols:
        for cc in comp_cols:
            mask = df[cc].astype(str).str.upper().str.contains(re.escape(s))
            rows = df[mask]
            if not rows.empty:
                break

    if rows.empty:
        return None

    row = rows.iloc[0]

    def tryget(*names: str) -> Any:
        for n in names:
            for c in df.columns:
                if c.strip().lower() == n.strip().lower():
                    return _normalize_value(row.get(c))
        return None

    snapshot: Dict[str, Any] = {
        "symbol": tryget("symbol", "Symbol"),
        "company_name": tryget("company_name", "company name", "description", "Company"),
        "rank": tryget("rank"),
        "price": tryget("price", "close", "last"),
        "change_1d_pct": tryget("change_1d_pct", "change % (24 hrs)", "change_1d"),
        "change_1w_pct": tryget("change_1w_pct", "change % (1w)", "change_1w"),
        "vwap": tryget("vwap", "vwap (24 hrs)"),
        "mcap_rs_cr": tryget("mcap_rs_cr", "market cap (rs. cr.)", "market capitalization"),
        "volume_24h_rs_cr": tryget("volume_24h_rs_cr", "volume (rs cr.)", "turnover"),
        "all_time_high": tryget("all_time_high", "all time high (rs.)"),
        "atr_pct": tryget("atr_pct", "atr% (24 hrs)"),
        "relative_vol": tryget("relative_vol", "relative vol"),
        "vol_change_pct": tryget("vol_change_pct", "vol. change (24 hrs)"),
        "volatility": tryget("volatility"),
    }

    # attach snapshot date from filename when possible
    snapshot["market_snapshot_date"] = _extract_date_from_filename(_EOD_PATH) if _EOD_PATH else None

    # attach image urls (optional, best-effort)
    try:
        from services.image_utils import get_logo_path, get_banner_path  # type: ignore
    except Exception:
        def get_logo_path(*a: Any, **k: Any) -> Optional[str]:
            return None
        def get_banner_path(*a: Any, **k: Any) -> Optional[str]:
            return None

    snapshot["logo_url"] = get_logo_path(snapshot.get("symbol"), snapshot.get("company_name"))
    snapshot["banner_url"] = get_banner_path(snapshot.get("symbol"), snapshot.get("company_name"))

    # coerce numerics where sensible
    numeric_keys = [
        "price", "vwap", "mcap_rs_cr", "volume_24h_rs_cr", "all_time_high",
        "atr_pct", "relative_vol", "vol_change_pct", "volatility",
    ]
    for k in numeric_keys:
        if k in snapshot and snapshot[k] is not None:
            try:
                snapshot[k] = float(snapshot[k])
            except Exception:
                # leave raw if conversion fails
                pass

    return snapshot


# ---- indices loading / lookup ----------------------------------------------
def load_indices_df(force_reload: bool = False) -> Optional[pd.DataFrame]:
    """
    Load and cache the broad_and_sector indices CSV from STATIC_DIR.

    Supports local filesystem and S3 prefix (via fsspec) seamlessly.
    """
    global _INDICES_DF, _INDICES_PATH
    if _INDICES_DF is not None and not force_reload:
        return _INDICES_DF

    local_static_dir = Path(STATIC_DIR) if isinstance(STATIC_DIR, str) else STATIC_DIR

    candidates = [
        local_static_dir / "broad_and_sector_indices.csv",
        local_static_dir / "BROAD_AND_SECTOR_INDICES.csv",
        local_static_dir / "broad and sector indices.csv",
    ]

    found: Optional[Union[Path, str]] = None
    for p in candidates:
        if isinstance(p, Path) and p.exists():
            found = p
            break

    # fallback local glob
    if found is None and isinstance(local_static_dir, Path) and local_static_dir.exists():
        for p in local_static_dir.glob("*.csv"):
            if "sector" in p.name.lower() or "index" in p.name.lower():
                found = p
                break

    # fallback S3
    if found is None and S3_STATIC_CSV_PATH:
        try:
            import fsspec  # type: ignore

            fs = fsspec.filesystem("s3")
            prefix = S3_STATIC_CSV_PATH.rstrip("/")
            entries = fs.glob(f"{prefix}/*.csv") or []
            for key in entries:
                name = key.split("/")[-1]
                if "sector" in name.lower() or "index" in name.lower():
                    found = _ensure_s3_uri(key)
                    break
        except Exception as exc:  # pragma: no cover - defensive
            _log("S3 static listing failed:", exc)

    if not found:
        return None

    try:
        if isinstance(found, str) and found.startswith("s3://"):
            df = pd.read_csv(found, low_memory=False, storage_options={"anon": False})
        else:
            try:
                df = pd.read_csv(found, low_memory=False)
            except Exception:
                df = pd.read_csv(found, encoding="latin1", low_memory=False)
    except Exception:
        if isinstance(found, str) and found.startswith("s3://"):
            df = pd.read_csv(found, encoding="latin1", low_memory=False, storage_options={"anon": False})
        else:
            df = pd.read_csv(found, encoding="latin1", low_memory=False)

    df.columns = [c.strip() for c in df.columns]
    _INDICES_DF = df
    _INDICES_PATH = found
    return _INDICES_DF


def get_indices_for_symbol(symbol: str) -> Tuple[str, str]:
    """
    Return (BroadIndex, SectorialIndex) for a given symbol.
    BroadIndex is chosen by priority list; SectorialIndex is the sector name (first token).
    """
    df = load_indices_df()
    if df is None:
        return ("Uncategorised Index", "Uncategorised Sector")

    s = str(symbol).strip().upper()

    # prefer exact match on 'Symbol' column when present
    results = pd.DataFrame()
    if "Symbol" in df.columns:
        results = df[df["Symbol"].astype(str).str.upper() == s]

    if results.empty and "Description" in df.columns:
        results = df[df["Description"].astype(str).str.upper().str.contains(re.escape(s))]

    if results.empty:
        # try other common columns
        symbol_cols = [c for c in df.columns if c.strip().lower() in ("symbol", "sym", "ticker")]
        desc_cols = [c for c in df.columns if c.strip().lower() in ("description", "company", "name")]
        if symbol_cols:
            for sc in symbol_cols:
                results = df[df[sc].astype(str).str.upper() == s]
                if not results.empty:
                    break
        if results.empty and desc_cols:
            for dc in desc_cols:
                results = df[df[dc].astype(str).str.upper().str.contains(re.escape(s))]
                if not results.empty:
                    break

    if results.empty:
        return ("Uncategorised Index", "Uncategorised Sector")

    row = results.iloc[0]

    sector_raw = None
    for cand in ("SectorialIndex", "Sector", "Sector Name", "Sectorial", "sector"):
        if cand in row.index:
            sector_raw = row.get(cand)
            break
    if pd.isna(sector_raw) or not str(sector_raw).strip():
        sector = "Uncategorised Sector"
    else:
        sector = str(sector_raw).split(",")[0].strip() or "Uncategorised Sector"

    broad = "Uncategorised Index"
    for b in BROAD_PRIORITY:
        if b in row.index and str(row.get(b)).strip().lower() in ("yes", "y", "true", "1"):
            broad = b
            break

    return (broad, sector)


# ---- formatting helper (UI display) ----------------------------------------
def format_snapshot_for_display(symbol: str) -> str:
    """Return a pretty string for quick terminal/UI debugging."""
    snap = get_market_snapshot(symbol)
    if not snap:
        return f"{symbol}: No market snapshot available."

    broad, sector = get_indices_for_symbol(symbol)

    def arrow(val: Any) -> str:
        try:
            f = float(val)
            if f > 0:
                return "⬆️"
            if f < 0:
                return "🔻"
        except Exception:
            pass
        return ""

    price = snap.get("price")
    change1d = snap.get("change_1d_pct")
    change1w = snap.get("change_1w_pct")

    s = f\"{snap.get('symbol')} | {snap.get('company_name')}\\n\"
    s += f\"{broad} | {sector}\\n\\n\"
    s += f\"📊 Market Snapshot: |{snap.get('market_snapshot_date')}, EOD|\\n\"
    s += f\"Price: ₹{price} | {change1d}% (1D) {arrow(change1d)} | {change1w}% (1W) {arrow(change1w)}\\n\"
    s += f\"Volume (24 Hrs): ₹{snap.get('volume_24h_rs_cr')} Cr\\n\"
    s += f\"Mcap: ₹{snap.get('mcap_rs_cr')} Cr | Rank: #{snap.get('rank')} by Mcap\\n\\n\"
    s += f\"VWAP: ₹{snap.get('vwap')} | ATR (14D): {snap.get('atr_pct')}%\\n\"
    s += f\"Relative Vol: {snap.get('relative_vol')} | Vol Change: {snap.get('vol_change_pct')}%\\n\"
    s += f\"Volatility: {snap.get('volatility')}%\\n\"
    return s.strip()


def list_symbols(limit: Optional[int] = None) -> List[str]:
    """Return list of symbols from the latest processed CSV (best-effort)."""
    df = load_processed_df()
    if df is None:
        return []
    symbol_cols = [c for c in df.columns if c.strip().lower() in ("symbol", "sym", "ticker")]
    if symbol_cols:
        s = df[symbol_cols[0]].astype(str).str.strip().unique().tolist()
    else:
        comp_cols = [c for c in df.columns if c.strip().lower() in ("company_name", "company", "description")]
        if comp_cols:
            s = df[comp_cols[0]].astype(str).str.strip().unique().tolist()
        else:
            s = []
    if limit:
        return s[:limit]
    return s


# ---- CLI -------------------------------------------------------------------
def _cli() -> None:
    parser = argparse.ArgumentParser(description="csv_utils - fetch market snapshot from processed EOD CSV")
    parser.add_argument("--symbol", help="Stock symbol or company name to lookup")
    parser.add_argument("--list", action="store_true", help="List available symbols (first 200)")
    parser.add_argument("--json", action="store_true", help="Output JSON for --symbol lookup")
    parser.add_argument("--reload", action="store_true", help="Force reload cached CSVs")
    parser.add_argument("--limit", type=int, default=200, help="Limit for --list output")
    args = parser.parse_args()

    if args.reload:
        load_processed_df(force_reload=True)
        load_indices_df(force_reload=True)

    if args.list:
        syms = list_symbols(limit=args.limit)
        for x in syms:
            print(x)
        return

    if args.symbol:
        if args.json:
            snap = get_market_snapshot(args.symbol)
            if not snap:
                print(json.dumps({"error": "not found", "symbol": args.symbol}))
                return
            broad, sector = get_indices_for_symbol(args.symbol)
            snap["broad_index"], snap["sector_index"] = broad, sector
            # ensure JSON serializable (numpy types)
            print(json.dumps(snap, default=lambda o: (None if pd.isna(o) else (int(o) if hasattr(o, "astype") else str(o)))))
            return

        # pretty display
        print(format_snapshot_for_display(args.symbol))
        return

    parser.print_help()


if __name__ == "__main__":
    _cli()
