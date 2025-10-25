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
	•	Both synchronous process_csv_sync() and async_process_csv() wrapper for FastAPI safe usage.
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

import atexit
import argparse
import asyncio
import contextvars
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, TypedDict
import threading
from functools import partial

import pandas as pd

# Optional dependency for S3 operations (fsspec + s3fs)
try:
    import fsspec  # type: ignore
except (ImportError, ModuleNotFoundError):
    fsspec = None  # type: ignore

# Logger: Do NOT configure handlers here — main.py should configure logging
logger = logging.getLogger(__name__)
# NOTE: Do not call logger.setLevel(...) here — leave logging configuration to the application.

# ---------------------------------------------------------------------
# Configuration via Pydantic Settings (optional but consistent with main.py)
# ---------------------------------------------------------------------
try:
    from pydantic_settings import BaseSettings
except Exception:
    try:
        from pydantic import BaseSettings
    except Exception:
        BaseSettings = object  # fallback dummy class if dependency missing

class CSVSettings(BaseSettings):
    S3_RAW_CSV_PATH: str | None = None
    S3_PROCESSED_CSV_PATH: str | None = None
    S3_STORAGE_OPTIONS: str | None = None
    S3_OP_TIMEOUT: int = 30
    S3_RETRIES: int = 2
    S3_RETRY_BACKOFF: float = 1.0
    LOG_LEVEL: str = "INFO"

    model_config = dict(env_file=None, case_sensitive=False)

csv_settings = CSVSettings()
# Do not change logger configuration here; the app should control global logging.

def _sanitize_config(config: dict) -> dict:
    """Redact sensitive fields from config dict for safe logging."""
    sanitized = config.copy()
    sensitive_keys = {'key', 'secret', 'token', 'password', 'credential'}
    for key in list(sanitized.keys()):
        if any(s in key.lower() for s in sensitive_keys):
            sanitized[key] = '***REDACTED***'
    return sanitized

logger.debug(
    "CSVSettings loaded: %s",
    _sanitize_config(csv_settings.model_dump() if hasattr(csv_settings, "model_dump") else vars(csv_settings)),
)

# Paths & env configuration
BASE = Path(__file__).resolve().parents[1]
LOCAL_RAW_DIR = BASE / "input_data" / "csv" / "eod_csv"
LOCAL_PROCESSED_DIR = BASE / "input_data" / "csv" / "processed_csv"

S3_RAW_CSV_PATH = (csv_settings.S3_RAW_CSV_PATH or "").strip() or None
S3_PROCESSED_CSV_PATH = (csv_settings.S3_PROCESSED_CSV_PATH or "").strip() or None

if S3_RAW_CSV_PATH and not S3_RAW_CSV_PATH.startswith("s3://"):
    logger.warning(
        "S3_RAW_CSV_PATH '%s' does not start with 's3://' — this may disable S3-first logic",
        S3_RAW_CSV_PATH,
    )
if S3_PROCESSED_CSV_PATH and not S3_PROCESSED_CSV_PATH.startswith("s3://"):
    logger.warning(
        "S3_PROCESSED_CSV_PATH '%s' does not start with 's3://' — this may disable S3 writes",
        S3_PROCESSED_CSV_PATH,
    )

_S3_STORAGE_OPTIONS_RAW = (csv_settings.S3_STORAGE_OPTIONS or "").strip() if isinstance(csv_settings.S3_STORAGE_OPTIONS, str) else ""
try:
    S3_STORAGE_OPTIONS: Dict[str, Any] = json.loads(_S3_STORAGE_OPTIONS_RAW) if _S3_STORAGE_OPTIONS_RAW else {}
except Exception:
    S3_STORAGE_OPTIONS = {}
    logger.warning("Failed to parse S3_STORAGE_OPTIONS JSON; continuing without storage options")
else:
    # safe debug of parsed S3 options (redact secrets)
    try:
        logger.debug("Parsed S3_STORAGE_OPTIONS: %s", _sanitize_config(dict(S3_STORAGE_OPTIONS)))
    except Exception:
        # never fail because of logging
        logger.debug("Parsed S3_STORAGE_OPTIONS (unprintable)")

# Expected keys in S3_STORAGE_OPTIONS (all optional):
#   - key: AWS access key ID
#   - secret: AWS secret access key
#   - endpoint_url: Custom S3 endpoint (for S3-compatible services)
#   - anon: Use anonymous access (bool, default False)
#   - client_kwargs: Additional boto3 client configuration (dict)
# Sensitive keys are automatically redacted in logs via _sanitize_config().

S3_OP_TIMEOUT = int(csv_settings.S3_OP_TIMEOUT)
S3_RETRIES = int(csv_settings.S3_RETRIES)
S3_RETRY_BACKOFF = float(csv_settings.S3_RETRY_BACKOFF)

RAW_CSV_DIR = S3_RAW_CSV_PATH or str(LOCAL_RAW_DIR)
PROCESSED_DIR = S3_PROCESSED_CSV_PATH or str(LOCAL_PROCESSED_DIR)

# Metrics and executor
_fallback_count = 0
_METRICS_COLLECTOR: Optional[Callable[[str, int], None]] = None
_METRICS_LOCK = threading.Lock()
# Executors: create lazily to avoid spawning threads at import time (helps tests / CLI).
_EXECUTOR: Optional[ThreadPoolExecutor] = None
_TIMEOUT_EXECUTOR: Optional[ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()
_shutting_down = False
# Cache for fsspec filesystem (avoid recreating for each operation)
_FSSPEC_FS: Optional[Any] = None
_FSSPEC_LOCK = threading.Lock()

class ProcessResult(TypedDict):
    path: str
    s3: bool
    rows: int

def _get_executor() -> ThreadPoolExecutor:
    if _shutting_down:
        raise RuntimeError("Executor creation disallowed during shutdown")
    global _EXECUTOR
    if _EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _EXECUTOR is None:
                # CSV_THREADPOOL_SIZE: Controls the number of worker threads for CSV processing.
                # Recommended range: 4-32. Default: 4. Increase for high-throughput scenarios.
                _EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("CSV_THREADPOOL_SIZE", "4")))
    return _EXECUTOR  # type: ignore[return-value]

def _get_timeout_executor() -> ThreadPoolExecutor:
    if _shutting_down:
        raise RuntimeError("Executor creation disallowed during shutdown")
    global _TIMEOUT_EXECUTOR
    if _TIMEOUT_EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _TIMEOUT_EXECUTOR is None:
                _TIMEOUT_EXECUTOR = ThreadPoolExecutor(max_workers=2)
    return _TIMEOUT_EXECUTOR  # type: ignore[return-value]

def _get_s3_fs() -> Any:
    """Lazily create and cache an fsspec S3 filesystem instance."""
    if not fsspec:
        raise RuntimeError("fsspec is not installed")
    global _FSSPEC_FS
    if _FSSPEC_FS is None:
        with _FSSPEC_LOCK:
            if _FSSPEC_FS is None:
                _FSSPEC_FS = fsspec.filesystem("s3", **S3_STORAGE_OPTIONS)
    return _FSSPEC_FS

def reset_metrics() -> None:
    global _fallback_count, _METRICS_COLLECTOR
    with _METRICS_LOCK:
        _fallback_count = 0
        _METRICS_COLLECTOR = None

def set_metrics_collector(collector: Optional[Callable[[str, int], None]]) -> None:
    global _METRICS_COLLECTOR
    # Protect collector swap with same lock used by incrementer to avoid races.
    with _METRICS_LOCK:
        _METRICS_COLLECTOR = collector

def _increment_fallback_metric() -> None:
    global _fallback_count, _METRICS_COLLECTOR
    with _METRICS_LOCK:
        _fallback_count += 1
        current = _fallback_count
    logger.info("Fallback to local storage occurred (total=%d)", current)
    if _METRICS_COLLECTOR:
        try:
            _METRICS_COLLECTOR("csv_processor.fallback_to_local", current)
        except Exception:
            logger.debug("Metrics collector failed; continuing")

# Optional integration helper for prometheus_client.Counter
def install_prometheus_counter(counter: Any) -> None:
    """
    Install a prometheus_client.Counter-like object as metrics collector.
    The provided `counter` must support `.inc()` with no args.

    Example usage from app startup:
      from prometheus_client import Counter
      from backend.services.csv_processor import install_prometheus_counter
      install_prometheus_counter(
          Counter("csv_processor_fallback_to_local_total", "Number of S3→local fallbacks")
      )
    """
    def _collector(name: str, val: int):
        try:
            # Counter only supports increment; assume increments by 1
            counter.inc()
        except Exception:
            logger.exception("Prometheus collector failed")
    set_metrics_collector(_collector)

# Helpers
def _is_s3_uri(path: str | Path) -> bool:
    return isinstance(path, str) and path.lower().startswith("s3://")

def _run_with_timeout(fn, *args, timeout: int = 30, **kwargs) -> Any:
    """
    Execute a blocking function in a small dedicated executor and enforce a timeout.
    Returns the function result or raises concurrent.futures.TimeoutError.
    """
    fut = _get_timeout_executor().submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=timeout)
    finally:
        try:
            fut.cancel()
        except Exception:
            pass

def _retry_call_sync(fn, args: tuple = (), retries: int = 2, backoff: float = 1.0, **kwargs) -> Any:
    """
    Synchronous retry wrapper for blocking functions.
    Call patterns supported:
      - _retry_call_sync(fn, args=(arg1, arg2))
      - _retry_call_sync(lambda: fn(arg1, arg2))
    Retries on exception with exponential backoff.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt, retries, getattr(fn, "__name__", "call"), exc,
            )
            if attempt < retries:
                import time
                time.sleep(backoff * attempt)
    logger.exception("All %d attempts failed for %s", retries, getattr(fn, "__name__", "call"))
    raise last_exc

# Helper to return a column Series aligned to df.index for optional columns.
def _col_or_none(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] if present, else a Series of None with same index and object dtype."""
    if col in df.columns:
        return df[col]
    return pd.Series([None] * len(df), index=df.index, dtype="object")

# S3 helpers (sync)
def _list_s3_csvs_sync(s3_prefix: str) -> List[str]:
    if not fsspec or not _is_s3_uri(s3_prefix):
        return []
    def _inner(prefix: str) -> List[str]:
        fs = _get_s3_fs()
        objs = fs.glob(f"{prefix.rstrip('/')}/*.csv")
        out: List[str] = []
        for o in objs:
            s = str(o)
            if s.startswith("s3://"):
                out.append(s)
            else:
                out.append(f"s3://{s}")
        return sorted(out)
    try:
        return _retry_call_sync(lambda p: _run_with_timeout(_inner, p, timeout=S3_OP_TIMEOUT), args=(s3_prefix,), retries=S3_RETRIES, backoff=S3_RETRY_BACKOFF)
    except FuturesTimeout:
        logger.warning("S3 list operation timed out after %s seconds for prefix %s", S3_OP_TIMEOUT, s3_prefix)
        return []
    except Exception as exc:
        logger.exception("Error listing S3 CSVs for %s: %s", s3_prefix, exc)
        return []

def _write_s3_csv_sync(output_path: str, df: pd.DataFrame) -> None:
    if not fsspec:
        raise RuntimeError("fsspec is not available for S3 writes")
    def _inner(path: str, _df: pd.DataFrame) -> None:
        fs = _get_s3_fs()
        # open in text mode with explicit encoding so pandas writes plain text
        with fs.open(path, "w", encoding="utf-8") as f:
            _df.to_csv(f, index=False)
    # apply retry + timeout
    return _retry_call_sync(lambda p, d: _run_with_timeout(_inner, p, d, timeout=S3_OP_TIMEOUT),
                            args=(output_path, df), retries=S3_RETRIES, backoff=S3_RETRY_BACKOFF)

def _read_csv_sync(src: str) -> pd.DataFrame:
    if _is_s3_uri(src):
        if not fsspec:
            raise RuntimeError("S3 URI provided but fsspec is not installed")
        def _inner(path: str):
            fs = _get_s3_fs()
            with fs.open(path, "r", encoding="utf-8") as f:
                return pd.read_csv(f, dtype=str, keep_default_na=False)
        try:
            return _run_with_timeout(_inner, src, timeout=S3_OP_TIMEOUT)
        except FuturesTimeout as e:
            # Raise same exception type for consistency with other timeout handlers
            raise FuturesTimeout(f"S3 read operation timed out after {S3_OP_TIMEOUT}s for {src}") from e
    else:
        return pd.read_csv(src, dtype=str, keep_default_na=False)

# Discovery: find latest CSV (sync)
def find_latest_csv_sync() -> Optional[str]:
    if _is_s3_uri(RAW_CSV_DIR):
        s3_candidates = _list_s3_csvs_sync(RAW_CSV_DIR)
        if s3_candidates:
            chosen = s3_candidates[-1]
            logger.info("Found latest CSV on S3: %s", chosen)
            return chosen
        else:
            logger.info("No S3 CSVs found — falling back to local folder")

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
    missing = [c for c in _REQUIRED_COLUMNS.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

# Core synchronous processor
def process_csv_sync(src_path: str, verbose: bool = False) -> ProcessResult:
    # Consider replacing Dict[str, Any] with ProcessResult for stronger typing in the future.
    src_str = str(src_path)
    try:
        df = _read_csv_sync(src_str)
    except Exception as e:
        logger.exception("Failed to read CSV %s: %s", src_str, e)
        raise

    df.columns = [c.strip() for c in df.columns]
    if verbose:
        logger.debug("Source CSV columns: %s", list(df.columns))

    try:
        _validate_columns(df)
    except Exception as exc:
        logger.error("CSV validation failed: %s", exc)
        raise

    def _to_float(x) -> Optional[float]:
        try:
            if x in ("", None):
                return None
            return float(str(x).replace(",", "").strip())
        except Exception:
            return None

    def _to_pct(x) -> Optional[float]:
        try:
            if x in ("", None):
                return None
            return float(str(x).replace("%", "").replace(",", "").strip())
        except Exception:
            return None

    out = pd.DataFrame()
    out["symbol"] = df.get("Symbol")
    out["company_name"] = df.get("Description")

    out["mcap_rs_cr"] = _col_or_none(df, "Market capitalization").map(_to_float).map(
        lambda v: round(v / 1e7, 2) if v is not None else None
    )

    out["price"] = _col_or_none(df, "Price").map(_to_float)
    out["change_1d_pct"] = _col_or_none(df, "Price Change % 1 day").map(_to_pct)
    out["change_1w_pct"] = _col_or_none(df, "Price Change % 1 week").map(_to_pct)
    out["vwap"] = _col_or_none(df, "Volume Weighted Average Price 1 day").map(_to_float)

    out["volume_change_24h_pct"] = _col_or_none(df, "Volume Change % 1 day").map(_to_pct)
    out["Volume_24H"] = _col_or_none(df, "Price * Volume (Turnover) 1 day").map(_to_float).map(lambda v: round(v / 1e7, 2) if v is not None else None)
    out["atr_14d"] = _col_or_none(df, "Average True Range % (14) 1 day").map(_to_pct)

    # do not coerce missing mcap to 0 by default; keep None to indicate missing
    # out["mcap_rs_cr"] = out["mcap_rs_cr"].fillna(0.0)

    out = out.drop_duplicates(subset=["symbol"], keep="first")
    if not out.empty and out["mcap_rs_cr"].notna().any():
        out = out.sort_values(by="mcap_rs_cr", ascending=False, na_position='last')
    else:
        logger.warning("All market cap values are missing; skipping sort by market cap")
    out = out.reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))

    numeric_candidates = [
        c for c in out.columns if c not in ("rank", "symbol", "company_name")
    ]
    try:
        out[numeric_candidates] = out[numeric_candidates].apply(pd.to_numeric, errors="coerce").round(2)
    except (ValueError, TypeError) as e:
        logger.warning("Batch numeric conversion failed: %s. Trying column-by-column.", e)
        for col in numeric_candidates:
            try:
                out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
            except Exception:
                logger.debug("Skipping rounding for column %s (non-numeric)", col)

    if out["symbol"].isnull().all() and "Description" in df.columns:
        possible = df["Description"].astype(str).fillna("")
        sym_candidates = possible.str.extract(r"^\s*([A-Z0-9\.\-]{1,10})\s*[-:]\s*(.+)$", expand=True)
        if sym_candidates.shape[1] == 2:
            out["symbol"] = sym_candidates[0]
            out["company_name"] = sym_candidates[1]

    safe_name = "processed_" + re.sub(r"[^A-Za-z0-9._-]", "_", Path(src_str).name)
    if _is_s3_uri(PROCESSED_DIR):
        output_path = f"{PROCESSED_DIR.rstrip('/')}/{safe_name}"
    else:
        local_out_dir = Path(PROCESSED_DIR)
        local_out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(local_out_dir / safe_name)

    try:
        if _is_s3_uri(output_path):
            if not fsspec:
                raise RuntimeError("Attempted to write to S3 but fsspec is not installed")
            _write_s3_csv_sync(output_path, out)
            logger.info("Processed CSV uploaded to S3: %s", output_path)
        else:
            out.to_csv(output_path, index=False, encoding="utf-8")
            logger.info("Processed CSV saved locally: %s", output_path)
    except Exception as e:
        logger.exception("Failed to write processed CSV to %s: %s", output_path, e)
        if _is_s3_uri(output_path):
            try:
                local_out_dir = Path(LOCAL_PROCESSED_DIR)
                local_out_dir.mkdir(parents=True, exist_ok=True)
                fallback = str(local_out_dir / safe_name)
                out.to_csv(fallback, index=False, encoding="utf-8")
                _increment_fallback_metric()
                logger.warning("S3 write failed — fell back to local path: %s", fallback)
                return ProcessResult(path=fallback, s3=False, rows=len(out))
            except Exception as exc2:
                logger.exception("Fallback local write also failed: %s", exc2)
                raise
        raise

    return ProcessResult(path=output_path, s3=_is_s3_uri(output_path), rows=len(out))

# Async wrapper for FastAPI integration (ContextVar-aware)
async def async_process_csv(src_path: str, verbose: bool = False) -> ProcessResult:
    """
    Non-blocking wrapper for FastAPI endpoints: runs process_csv_sync in the shared executor.
    Preserves ContextVar values by copying context into executor call.
    """
    ctx = contextvars.copy_context()
    loop = asyncio.get_running_loop()
    func = partial(lambda _ctx, p, v: _ctx.run(process_csv_sync, p, v), ctx, src_path, verbose)
    return await loop.run_in_executor(_get_executor(), func)

# CLI for direct invocation
async def main():
    parser = argparse.ArgumentParser(description="Process latest or given EOD CSV (S3-aware).")
    parser.add_argument("--src", help="Optional path to CSV (S3 or local). If omitted, auto-detects latest.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    csv_path = args.src or find_latest_csv_sync()
    if not csv_path:
        logger.error("No CSV source found. Exiting.")
        sys.exit(1)

    # run sync core directly (CLI)
    result = process_csv_sync(csv_path, verbose=args.verbose)
    logger.info("Processing result: %s", result)

# Shutdown hook: clean up executors on exit
def _shutdown_executor():
    # Prevent new executors from being created while we shut down.
    global _shutting_down
    with _EXECUTOR_LOCK:
        _shutting_down = True
        logger.info("Shutting down executors...")
        try:
            if _EXECUTOR is not None:
                try:
                    _EXECUTOR.shutdown(wait=True, cancel_futures=True)
                except TypeError:
                    _EXECUTOR.shutdown(wait=True)
        except Exception:
            logger.exception("Error shutting down _EXECUTOR")

        try:
            if _TIMEOUT_EXECUTOR is not None:
                _TIMEOUT_EXECUTOR.shutdown(wait=True)
        except Exception:
            logger.exception("Error shutting down _TIMEOUT_EXECUTOR")

        # try to clean up cached fsspec FS if present
        try:
            global _FSSPEC_FS
            if _FSSPEC_FS is not None:
                try:
                    # some fs implementations have a .close() method
                    close_fn = getattr(_FSSPEC_FS, "close", None)
                    if callable(close_fn):
                        close_fn()
                except Exception:
                    logger.debug("Failed to close cached fsspec FS; ignoring")
                _FSSPEC_FS = None
        except Exception:
            logger.debug("Error during fsspec FS cleanup; ignoring")

atexit.register(_shutdown_executor)

__all__ = [
    "find_latest_csv_sync",
    "process_csv_sync",
    "async_process_csv",
    "set_metrics_collector",
    "reset_metrics",
    "install_prometheus_counter",
    "main",
]
if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(main())
