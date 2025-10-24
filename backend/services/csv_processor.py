#!/usr/bin/env python3
"""
backend/services/csv_processor.py

Robust CSV processor (finalized):

Features
	•	S3-first discovery and write, with automatic fallback to local filesystem.
	•	Threaded timeouts + retry/backoff for blocking S3 ops.
	•	Optional S3 storage options via JSON env var (S3_STORAGE_OPTIONS).
	•	Schema validation for required core columns.
	•	Column alias mapping to requested output columns:
Input -> Output
"Volume Change % 1 day" -> "volume_change_24h_pct" (percentage)
"Price * Volume (Turnover) 1 day" -> "Volume_24H" (numeric, converted to Crores)
"Average True Range % (14) 1 day" -> "atr_14d" (percentage)
	•	Company / symbol fallbacks and retention.
	•	Round all numeric output columns to 2 decimals (except 'rank').
	•	Both synchronous process_csv() and async_process_csv() wrapper for FastAPI safe usage.
	•	Keeps CLI compatibility: can be executed as module to auto-detect latest CSV.
	•	Emits structured logging (no print statements).
	•	Exposes a small metric counter for S3->local fallback events.

Notes for maintainers
	•	This module is purposely defensive: it avoids import-time side-effects (no load_dotenv).
If you want .env loading, call your app-level load_env() (e.g. services.load_env()) prior
to invoking functions that rely on environment variables.
	•	If you call process_csv() from a FastAPI route, use async_process_csv() or
run_in_threadpool to avoid blocking the event loop.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Optional dependency for S3 operations (fsspec + s3fs)
try:
    import fsspec  # type: ignore
except Exception:
    fsspec = None  # type: ignore

# Logger / basic config
logger = logging.getLogger(__name__)
if not logger.handlers:
    _lvl = os.getenv("LOG_LEVEL", "INFO")
    logging_level = getattr(logging, _lvl.upper(), logging.INFO)
    logging.basicConfig(level=logging_level)
    logger = logging.getLogger(__name__)

# Paths & env configuration
BASE = Path(__file__).resolve().parents[1]
LOCAL_RAW_DIR = BASE / "input_data" / "csv" / "eod_csv"
LOCAL_PROCESSED_DIR = BASE / "input_data" / "csv" / "processed_csv"

S3_RAW_CSV_PATH = os.getenv("S3_RAW_CSV_PATH", "").strip() or None
S3_PROCESSED_CSV_PATH = os.getenv("S3_PROCESSED_CSV_PATH", "").strip() or None

# Validate S3 URI format early (warn only)
if S3_RAW_CSV_PATH and not S3_RAW_CSV_PATH.startswith("s3://"):
    logger.warning("S3_RAW_CSV_PATH '%s' does not start with 's3://' — this may disable S3-first logic", S3_RAW_CSV_PATH)
if S3_PROCESSED_CSV_PATH and not S3_PROCESSED_CSV_PATH.startswith("s3://"):
    logger.warning("S3_PROCESSED_CSV_PATH '%s' does not start with 's3://' — this may disable S3 writes", S3_PROCESSED_CSV_PATH)

# Optional JSON-encoded storage options (credentials, client_kwargs, anon, etc.)
_S3_STORAGE_OPTIONS_RAW = os.getenv("S3_STORAGE_OPTIONS", "").strip()
try:
    S3_STORAGE_OPTIONS: Dict[str, Any] = json.loads(_S3_STORAGE_OPTIONS_RAW) if _S3_STORAGE_OPTIONS_RAW else {}
except Exception:
    S3_STORAGE_OPTIONS = {}
    logger.warning("Failed to parse S3_STORAGE_OPTIONS JSON; continuing without storage options")

# Timeouts & retries for S3 operations
S3_OP_TIMEOUT = int(os.getenv("S3_OP_TIMEOUT", "30"))
S3_RETRIES = int(os.getenv("S3_RETRIES", "2"))
S3_RETRY_BACKOFF = float(os.getenv("S3_RETRY_BACKOFF", "1.0"))

# Derived effective dirs (S3 URIs kept as strings; local as Path)
RAW_CSV_DIR = S3_RAW_CSV_PATH or str(LOCAL_RAW_DIR)
PROCESSED_DIR = S3_PROCESSED_CSV_PATH or str(LOCAL_PROCESSED_DIR)

# Small metric: count of fallback events from S3 -> local writes
_fallback_count = 0

def _increment_fallback_metric() -> None:
    """Increment the in-process fallback counter and log the event."""
    global _fallback_count
    _fallback_count += 1
    logger.info("Fallback to local storage occurred (total=%d)", _fallback_count)

# Helpers: concurrency, retry, timeouts
def _is_s3_uri(path: str | Path) -> bool:
    """Return True if the given path looks like an S3 URI (s3://...)."""
    return isinstance(path, str) and path.lower().startswith("s3://")

def _run_with_timeout(fn, *args, timeout: int = 30, **kwargs):
    """
    Execute a blocking function in a thread and enforce a timeout.
    Returns the function result or raises concurrent.futures.TimeoutError.
    """
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=timeout)
        finally:
            try:
                fut.cancel()
            except Exception:
                # best-effort; cannot kill C-level blocking calls
                pass

def _retry_call(fn, args, retries: int = 2, backoff: float = 1.0, **kwargs):
    """
    Simple retry wrapper. Retries on any Exception, waiting backoff*attempt between retries.
    Returns the first successful result or raises the last exception.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, retries, getattr(fn, "__name__", "call"), exc)
            if attempt < retries:
                time.sleep(backoff * attempt)
    logger.exception("All %d attempts failed for %s", retries, getattr(fn, "__name__", "call"))
    raise last_exc

# S3 helpers
def _list_s3_csvs(s3_prefix: str) -> List[str]:
    """
    List CSV files under an S3 prefix using fsspec. Returns a sorted list of s3:// URIs (or empty list).
    This function applies retry + timeout wrappers.
    """
    if not fsspec or not _is_s3_uri(s3_prefix):
        return []
    try:
        def _inner(prefix: str) -> List[str]:
            fs = fsspec.filesystem("s3", **S3_STORAGE_OPTIONS)
            objs = fs.glob(f"{prefix.rstrip('/')}/*.csv")
            out: List[str] = []
            for o in objs:
                s = str(o)
                if s.startswith("s3://"):
                    out.append(s)
                else:
                    out.append(f"s3://{s}")
            return sorted(out)

        return _retry_call(lambda p: _run_with_timeout(_inner, p, timeout=S3_OP_TIMEOUT),
                           (s3_prefix,), retries=S3_RETRIES, backoff=S3_RETRY_BACKOFF)
    except FuturesTimeout:
        logger.warning("S3 list operation timed out after %s seconds for prefix %s", S3_OP_TIMEOUT, s3_prefix)
        return []
    except Exception as exc:
        logger.exception("Error listing S3 CSVs for %s: %s", s3_prefix, exc)
        return []

def _write_s3_csv(output_path: str, df: pd.DataFrame) -> None:
    """
    Write DataFrame to S3 using fsspec. Uses retry + timeout wrappers.
    Raises exceptions on failure.
    """
    if not fsspec:
        raise RuntimeError("fsspec is not available for S3 writes")
    def _inner(path: str, _df: pd.DataFrame) -> None:
        fs = fsspec.filesystem("s3", **S3_STORAGE_OPTIONS)
        with fs.open(path, "w") as f:
            _df.to_csv(f, index=False, encoding="utf-8")
    # apply retry + timeout
    return _retry_call(lambda p, d: _run_with_timeout(_inner, p, d, timeout=S3_OP_TIMEOUT),
                       (output_path, df), retries=S3_RETRIES, backoff=S3_RETRY_BACKOFF)

def _read_csv(src: str) -> pd.DataFrame:
    """
    Read CSV from local path or S3 (via fsspec). Raises on fatal errors.
    Returns a pandas.DataFrame with raw strings (no NA coercion).
    """
    if _is_s3_uri(src):
        if not fsspec:
            raise RuntimeError("S3 URI provided but fsspec is not installed")
        def _inner(path: str):
            fs = fsspec.filesystem("s3", **S3_STORAGE_OPTIONS)
            with fs.open(path, "r") as f:
                return pd.read_csv(f, dtype=str, keep_default_na=False)
        try:
            return _run_with_timeout(_inner, src, timeout=S3_OP_TIMEOUT)
        except FuturesTimeout:
            raise TimeoutError(f"S3 read operation timed out after {S3_OP_TIMEOUT}s for {src}")
    else:
        # local file (pandas will raise FileNotFoundError if missing)
        return pd.read_csv(src, dtype=str, keep_default_na=False)

# Discovery: find latest CSV (S3-first then local)
def find_latest_csv() -> Optional[str]:
    """
    Auto-detect the latest CSV to process.
    - Tries S3 first (if RAW_CSV_DIR is an s3:// prefix and fsspec available).
    - Falls back to the local raw directory.
    Returns an S3 URI or local file path string, or None if nothing found.
    """
    # S3-first
    if _is_s3_uri(RAW_CSV_DIR):
        s3_candidates = _list_s3_csvs(RAW_CSV_DIR)
        if s3_candidates:
            chosen = s3_candidates[-1]
            logger.info("Found latest CSV on S3: %s", chosen)
            return chosen
        else:
            logger.info("No S3 CSVs found — falling back to local folder")

    # Local fallback
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
            logger.info("Using local CSV: %s", chosen)
            return str(chosen)

    logger.warning("No CSV files found (S3 or local).")
    return None

# Validation utilities
_REQUIRED_COLUMNS = {
    "Symbol": {"reason": "needed for 'symbol'"},
    "Description": {"reason": "needed for 'company_name'"},
    "Price": {"reason": "needed for 'price'"},
}

def _validate_columns(df: pd.DataFrame) -> None:
    """Ensure required core columns are present; raise ValueError if missing."""
    missing = [c for c in _REQUIRED_COLUMNS.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

# Data processing (mapping + normalization)
def process_csv(src_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Process a CSV (local or S3) and write processed CSV to configured processed dir.

    Returns
    -------
    dict
        {"path": output_path, "s3": bool, "rows": int}

    Raises
    ------
    Exception on fatal errors (reading/writing). Callers should catch/log as desired.
    """
    src_str = str(src_path)

    # Read source
    try:
        df = _read_csv(src_str)
    except Exception as e:
        logger.exception("Failed to read CSV %s: %s", src_str, e)
        raise

    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    if verbose:
        logger.debug("Source CSV columns: %s", list(df.columns))

    # Validate required columns
    try:
        _validate_columns(df)
    except Exception as exc:
        logger.error("CSV validation failed: %s", exc)
        raise

    # Conversion helpers
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
            # strip percent sign and commas
            return float(str(x).replace("%", "").replace(",", "").strip())
        except Exception:
            return None

    # Build output DataFrame
    out = pd.DataFrame()
    # Basic mapping: symbol & company_name (keep original strings, but coerce later)
    out["symbol"] = df.get("Symbol")
    out["company_name"] = df.get("Description")

    # Market cap to crores (existing behavior)
    out["mcap_rs_cr"] = df.get("Market capitalization", pd.Series(dtype="float")).map(_to_float).map(
        lambda v: round(v / 1e7, 2) if v else None
    )

    # Price
    out["price"] = df.get("Price").map(_to_float)

    # Existing percent columns
    out["change_1d_pct"] = df.get("Price Change % 1 day").map(_to_pct)
    out["change_1w_pct"] = df.get("Price Change % 1 week").map(_to_pct)

    # VWAP
    out["vwap"] = df.get("Volume Weighted Average Price 1 day").map(_to_float)

    # -------------------------
    # NEW requested mappings (column aliases)
    # -------------------------
    # Input -> Output
    # "Volume Change % 1 day" -> "volume_change_24h_pct" (percentage)
    # "Price * Volume (Turnover) 1 day" -> "Volume_24H" (numeric), convert to Crores
    # "Average True Range % (14) 1 day" -> "atr_14d" (percentage)
    out["volume_change_24h_pct"] = df.get("Volume Change % 1 day").map(_to_pct)
    # read turnover; convert to float then to Crores (divide by 1e7) — keep None when missing
    out["Volume_24H"] = df.get("Price * Volume (Turnover) 1 day").map(_to_float).map(lambda v: round(v / 1e7, 2) if v is not None else None)
    out["atr_14d"] = df.get("Average True Range % (14) 1 day").map(_to_pct)

    # Ensure mcap not NaN
    out["mcap_rs_cr"] = out["mcap_rs_cr"].fillna(0.0)

    # Remove duplicates (by symbol) and sort by market cap (desc)
    out = out.drop_duplicates(subset=["symbol"], keep="first").sort_values(
        by="mcap_rs_cr", ascending=False
    ).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))

    # -------------------------
    # Rounding: all numeric columns except 'rank' to 2 decimals
    # If coercion to numeric fails, preserve original value (best-effort)
    # -------------------------
    # Round only numeric columns (exclude text fields)
    numeric_candidates = [
        c for c in out.columns if c not in ("rank", "symbol", "company_name")
    ]
    for col in numeric_candidates:
        try:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
        except Exception:
            # keep original values if rounding fails
            logger.debug(f"Skipping rounding for column {col} (non-numeric)")

    # Re-apply symbol/company_name fallback if entirely missing
    # If symbol missing but Description contains "SYMBOL - NAME" style, try to split
    if out["symbol"].isnull().all() and "Description" in df.columns:
        # heuristic split: "SYM - Company Name" or "SYM: Company Name"
        possible = df["Description"].astype(str).fillna("")
        sym_candidates = possible.str.extract(r"^\s*([A-Z0-9\.\-]{1,10})\s*[-:]\s*(.+)$", expand=True)
        if sym_candidates.shape[1] == 2:
            out["symbol"] = sym_candidates[0]
            out["company_name"] = sym_candidates[1]

    # -------------------------
    # Output path selection (S3 first if configured)
    # -------------------------
    safe_name = "processed_" + re.sub(r"[^A-Za-z0-9._-]", "_", Path(src_str).name)
    if _is_s3_uri(PROCESSED_DIR):
        output_path = f"{PROCESSED_DIR.rstrip('/')}/{safe_name}"
    else:
        local_out_dir = Path(PROCESSED_DIR)
        local_out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(local_out_dir / safe_name)

    # -------------------------
    # Write processed CSV (try S3 then fallback to local)
    # -------------------------
    try:
        if _is_s3_uri(output_path):
            if not fsspec:
                raise RuntimeError("Attempted to write to S3 but fsspec is not installed")
            _write_s3_csv(output_path, out)
            logger.info("Processed CSV uploaded to S3: %s", output_path)
        else:
            out.to_csv(output_path, index=False, encoding="utf-8")
            logger.info("Processed CSV saved locally: %s", output_path)
    except Exception as e:
        logger.exception("Failed to write processed CSV to %s: %s", output_path, e)
        # If we attempted S3, try local fallback and increment metric
        if _is_s3_uri(output_path):
            try:
                local_out_dir = Path(LOCAL_PROCESSED_DIR)
                local_out_dir.mkdir(parents=True, exist_ok=True)
                fallback = str(local_out_dir / safe_name)
                out.to_csv(fallback, index=False, encoding="utf-8")
                _increment_fallback_metric()
                logger.warning("S3 write failed — fell back to local path: %s", fallback)
                return {"path": fallback, "s3": False, "rows": len(out)}
            except Exception as exc2:
                logger.exception("Fallback local write also failed: %s", exc2)
                raise
        raise

    return {"path": output_path, "s3": _is_s3_uri(output_path), "rows": len(out)}

# Async wrapper for FastAPI integration
async def async_process_csv(src_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Non-blocking wrapper for FastAPI endpoints: runs process_csv in an executor.
    Usage within FastAPI route:
    result = await async_process_csv(path)
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, process_csv, src_path, verbose)

# CLI for direct invocation
def main():
    parser = argparse.ArgumentParser(description="Process latest or given EOD CSV (S3-aware).")
    parser.add_argument("--src", help="Optional path to CSV (S3 or local). If omitted, auto-detects latest.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    csv_path = args.src or find_latest_csv()
    if not csv_path:
        logger.error("No CSV source found. Exiting.")
        sys.exit(1)

    result = process_csv(csv_path, verbose=args.verbose)
    logger.info("Processing result: %s", result)

__all__ = ["find_latest_csv", "process_csv", "async_process_csv", "main"]

if __name__ == "__main__":
    main()
