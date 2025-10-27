#!/usr/bin/env python3
"""
backend/services/csv_utils.py
v0.0.01

Enterprise-grade, read-only CSV utilities that reuse csv_processor.py primitives.
- Must be used with csv_processor.py (fail-fast if csv_processor not available).
- Thread-safe caching with double-checked locking (avoids holding locks during I/O).
- S3 reads via fsspec/_get_s3_fs with boto3 fallback.
- latin1 fallback for encoding issues.
- Path-safety checks to prevent traversal attacks.
- Health check, Prometheus metrics (optional), audit logging, CLI, and async wrappers.
"""

from __future__ import annotations

import argparse
import atexit
import asyncio
import contextvars
import json
import logging
import os
import re
import sys
import threading
import time
from functools import partial, lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

# load .env optionally in a robust way
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env")
except Exception:
    pass

# Strict dependency: csv_processor primitives (fail-fast)
try:
    from .csv_processor import (
        get_csv_settings,
        _get_s3_fs,
        _get_executor,
        _retry_call_sync,
        CircuitBreaker,
        correlation_id_var,
        logger,
        _audit_log,
        _increment_error_metric,
        _is_s3_uri,
        _sanitize_config,
        LOCAL_PROCESSED_DIR,
        LOCAL_STATIC_DIR,
        _parse_s3_storage_options,
        boto3,  # may be None if not installed but import succeeds
        # Additional imports for comprehensive testing
        _sanitize_csv_value,
        _robust_float_convert,
        _robust_percent_convert,
        find_latest_csv_sync,
        process_csv_sync,
        configure_logging_from_settings,
        ClientError,
    )
except Exception as exc:
    raise ImportError(
        "csv_utils requires backend.services.csv_processor to be importable. "
        "Fix the import error in csv_processor."
    ) from exc

# ===== Add / replace this config block in backend/services/csv_utils.py =====
# Read env-driven config (tests set these env vars)
S3_PROCESSED_CSV_PATH = os.getenv("S3_PROCESSED_CSV_PATH", "")
S3_STATIC_CSV_PATH = os.getenv("S3_STATIC_CSV_PATH", "")
AWS_S3_SECRETS_NAME = os.getenv("AWS_S3_SECRETS_NAME", "")
CSV_COLUMN_MAPPINGS = os.getenv("CSV_COLUMN_MAPPINGS", "")
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")

# Local directories (do NOT create on import — tests use pyfakefs and will create them)
# Use sensible defaults that are directories, not files.
LOCAL_PROCESSED_DIR = Path(os.getenv("LOCAL_PROCESSED_DIR", "/tmp/input_data/csv"))
LOCAL_STATIC_DIR = Path(os.getenv("LOCAL_STATIC_DIR", "/tmp/static/csv"))
LOCAL_RAW_DIR = Path(os.getenv("LOCAL_RAW_DIR", str(LOCAL_PROCESSED_DIR.parent)))

# Ensure these are Path objects and normalized:
LOCAL_PROCESSED_DIR = LOCAL_PROCESSED_DIR.expanduser().resolve() if not LOCAL_PROCESSED_DIR.is_absolute() else LOCAL_PROCESSED_DIR
LOCAL_STATIC_DIR = LOCAL_STATIC_DIR.expanduser().resolve() if not LOCAL_STATIC_DIR.is_absolute() else LOCAL_STATIC_DIR
LOCAL_RAW_DIR = LOCAL_RAW_DIR.expanduser().resolve() if not LOCAL_RAW_DIR.is_absolute() else LOCAL_RAW_DIR

def _safe_setup_logging():
    """
    Configure a very conservative logging format so tests that inject custom
    log records (without extra fields like `correlation_id`) won't blow up.
    Tests may rely on capturing logging output.
    """
    # Keep it minimal and safe — avoid formats that refer to non-standard LogRecord keys.
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    # Avoid reconfiguring handlers repeatedly in tests: use basicConfig only if root has no handlers.
    if not logging.getLogger().handlers:
        logging.basicConfig(format=fmt)
    else:
        # If handlers exist, adjust formatter of existing handlers conservatively
        for h in logging.getLogger().handlers:
            try:
                h.setFormatter(logging.Formatter(fmt))
            except Exception:
                # best-effort; don't fail import
                pass

# Initial safe setup
_safe_setup_logging()

def reset_cached_settings():
    """
    Re-read environment and reset any module-level cached configuration used by csv_utils.
    Tests call this between cases to avoid cross-test pollution.
    """
    global S3_PROCESSED_CSV_PATH, S3_STATIC_CSV_PATH, AWS_S3_SECRETS_NAME, CSV_COLUMN_MAPPINGS, LOG_FORMAT
    global LOCAL_PROCESSED_DIR, LOCAL_STATIC_DIR, LOCAL_RAW_DIR

    S3_PROCESSED_CSV_PATH = os.getenv("S3_PROCESSED_CSV_PATH", "")
    S3_STATIC_CSV_PATH = os.getenv("S3_STATIC_CSV_PATH", "")
    AWS_S3_SECRETS_NAME = os.getenv("AWS_S3_SECRETS_NAME", "")
    CSV_COLUMN_MAPPINGS = os.getenv("CSV_COLUMN_MAPPINGS", "")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "text")

    # Recompute local paths (but still do NOT create them on reset)
    LOCAL_PROCESSED_DIR = Path(os.getenv("LOCAL_PROCESSED_DIR", "/tmp/input_data/csv"))
    LOCAL_STATIC_DIR = Path(os.getenv("LOCAL_STATIC_DIR", "/tmp/static/csv"))
    LOCAL_RAW_DIR = Path(os.getenv("LOCAL_RAW_DIR", str(LOCAL_PROCESSED_DIR.parent)))

    # Normalize path objects
    LOCAL_PROCESSED_DIR = LOCAL_PROCESSED_DIR.expanduser().resolve() if not LOCAL_PROCESSED_DIR.is_absolute() else LOCAL_PROCESSED_DIR
    LOCAL_STATIC_DIR = LOCAL_STATIC_DIR.expanduser().resolve() if not LOCAL_STATIC_DIR.is_absolute() else LOCAL_STATIC_DIR
    LOCAL_RAW_DIR = LOCAL_RAW_DIR.expanduser().resolve() if not LOCAL_RAW_DIR.is_absolute() else LOCAL_RAW_DIR

    # Clear any lru_cache-decorated functions in this module (best effort)
    try:
        for value in list(globals().values()):
            # detect functools.cache_clear attribute (lru_cache decorated objects)
            if hasattr(value, "cache_clear"):
                try:
                    value.cache_clear()
                except Exception:
                    pass
    except Exception:
        pass

    # Reconfigure safe logging after resetting LOG_FORMAT
    _safe_setup_logging()
# ===== end block =====

__version__ = "v0.0.01"

# Prometheus optional integration
try:
    from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except Exception:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = generate_latest = CollectorRegistry = None

# --------------------------
# Constants / settings
# --------------------------
DATE_RE = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")
BROAD_PRIORITY = [
    "NIFTY50", "NIFTYNEXT50", "NIFTY100", "NIFTY200",
    "NIFTY500", "NIFTYTOTALMARKET", "NIFTYMIDCAP50", "NIFTYMIDCAP150",
]

# Caches & locks
_EOD_DF: Optional[pd.DataFrame] = None
_EOD_PATH: Optional[Union[Path, str]] = None
_EOD_CACHE_TS: float = 0.0

_INDICES_DF: Optional[pd.DataFrame] = None
_INDICES_PATH: Optional[Union[Path, str]] = None
_INDICES_CACHE_TS: float = 0.0

_eod_cache_lock = threading.Lock()
_indices_cache_lock = threading.Lock()

# Cache TTL (seconds)
_CACHE_TTL_SECONDS = 5 * 60

# Circuit breaker for read operations
_s3_read_circuit_breaker = CircuitBreaker()

# S3 concurrency semaphore
_s3_semaphore: Optional[threading.Semaphore] = None
_s3_semaphore_lock = threading.Lock()

# Metrics queue (simple) - we will export to Prometheus counters/histogram when available
_METRICS_QUEUE: "queue.Queue" = __import__("queue").Queue(maxsize=1000)
_METRICS_WORKER_THREAD: Optional[threading.Thread] = None
_METRICS_LOCK = threading.Lock()

# Prometheus metrics (if available)
if PROMETHEUS_AVAILABLE:
    _PROM_REG = CollectorRegistry()
    _PROM_COUNTER_OPS = Counter("csv_utils_operations_total", "CSV utils operations", ["op"], registry=_PROM_REG)
    _PROM_ERRORS = Counter("csv_utils_errors_total", "CSV utils errors", ["op"], registry=_PROM_REG)
    _PROM_DURATION = Histogram("csv_utils_duration_seconds", "Operation durations", ["op"], registry=_PROM_REG)
else:
    _PROM_REG = _PROM_COUNTER_OPS = _PROM_ERRORS = _PROM_DURATION = None

# --------------------------
# Helpers
# --------------------------
def _now() -> float:
    return time.time()

def _is_cache_fresh(ts: float) -> bool:
    return (ts > 0.0) and (_now() - ts) < _CACHE_TTL_SECONDS

def _is_path_safe(path: Union[str, Path], allowed_dirs: Iterable[Union[str, Path]]) -> bool:
    """
    Return True if `path` is located inside (or equal to) one of the `allowed_dirs`.
    This resolves symlinks and uses strict=False so it won't fail if file doesn't exist
    (important for tests that use fake filesystems).
    """
    try:
        p = Path(path)
        resolved = p.resolve(strict=False)
    except Exception:
        # fallback to absolute if resolve() raises unexpectedly
        resolved = Path(path).absolute()

    for base in allowed_dirs:
        try:
            base_p = Path(base).resolve(strict=False)
        except Exception:
            base_p = Path(base).absolute()

        # If the target path equals the allowed base or is below it, it's safe.
        if resolved == base_p or base_p in resolved.parents:
            return True

    return False

def _ensure_s3_uri(val: str) -> str:
    v = val.rstrip("/") if isinstance(val, str) else val
    if isinstance(v, str) and v.lower().startswith("s3://"):
        return v
    if isinstance(v, str) and "/" in v:
        return "s3://" + v.lstrip("/")
    return v

def _safe_metric(op: str, value: Union[int, float] = 1):
    """Push metric to queue and Prometheus if available."""
    try:
        _METRICS_QUEUE.put_nowait({"op": op, "value": value, "ts": _now()})
    except Exception:
        logger.debug("Metrics queue full/dropped: %s", op)
    if PROMETHEUS_AVAILABLE and _PROM_COUNTER_OPS:
        try:
            _PROM_COUNTER_OPS.labels(op=op).inc(value)
        except Exception:
            pass

# --------------------------
# Metrics worker
# --------------------------
def _metrics_worker():
    while True:
        try:
            item = _METRICS_QUEUE.get(timeout=1)
            if item is None:
                break
            # no-op: expand later to push to external exporter if needed
            logger.debug("csv_utils.metric: %s", item)
        except __import__("queue").Empty:
            continue
        except Exception:
            logger.exception("metrics worker error")

def _start_metrics_worker():
    global _METRICS_WORKER_THREAD
    if _METRICS_WORKER_THREAD is None or not _METRICS_WORKER_THREAD.is_alive():
        _METRICS_WORKER_THREAD = threading.Thread(target=_metrics_worker, daemon=True)
        _METRICS_WORKER_THREAD.start()

# --------------------------
# S3 semaphore helper
# --------------------------
def _get_s3_semaphore() -> threading.Semaphore:
    global _s3_semaphore
    if _s3_semaphore is None:
        with _s3_semaphore_lock:
            if _s3_semaphore is None:
                try:
                    conc = int(getattr(get_csv_settings(), "S3_CONCURRENCY", 4))
                except (ValueError, TypeError):
                    conc = 4
                _s3_semaphore = threading.Semaphore(max(1, conc))
    return _s3_semaphore

# --------------------------
# Resilient CSV read (fsspec preferred, boto3 fallback)
# --------------------------
def _read_csv_resilient(path: Union[str, Path]) -> pd.DataFrame:
    """
    Read a CSV from local or S3 with retries, latin1 fallback, and boto3 fallback.
    This function is synchronous and should be wrapped by _retry_call_sync by callers
    that need retries. However this file uses _retry_call_sync at call sites.
    """
    settings = get_csv_settings()

    def _op():
        pstr = str(path)
        if _is_s3_uri(pstr):
            sem = _get_s3_semaphore()
            acquired = sem.acquire(timeout=int(settings.S3_OP_TIMEOUT)) if sem else True
            try:
                try:
                    fs = _get_s3_fs()
                    with fs.open(pstr, "rb") as fh:
                        try:
                            return pd.read_csv(fh, low_memory=False)
                        except UnicodeDecodeError:
                            fh.seek(0)
                            return pd.read_csv(fh, encoding="latin1", low_memory=False)
                except Exception:
                    # boto3 fallback
                    if boto3:
                        m = re.match(r"s3://([^/]+)/(.+)", pstr)
                        if not m:
                            raise
                        bucket, key = m.groups()
                        client = boto3.client("s3")
                        resp = client.get_object(Bucket=bucket, Key=key)
                        body = resp["Body"].read()
                        bio = BytesIO(body)
                        try:
                            return pd.read_csv(bio, low_memory=False)
                        except UnicodeDecodeError:
                            bio.seek(0)
                            return pd.read_csv(bio, encoding="latin1", low_memory=False)
                    raise
            finally:
                if acquired and sem:
                    sem.release()
        else:
            try:
                return pd.read_csv(pstr, low_memory=False)
            except UnicodeDecodeError:
                return pd.read_csv(pstr, encoding="latin1", low_memory=False)

    return _retry_call_sync(
        _op,
        retries=settings.S3_RETRIES,
        backoff=settings.S3_RETRY_BACKOFF,
        timeout=settings.S3_OP_TIMEOUT,
        breaker=_s3_read_circuit_breaker
    )

# --------------------------
# Discovery: find latest processed EOD
# --------------------------
def find_latest_processed_eod() -> Optional[Union[Path, str]]:
    """
    Find the latest processed EOD CSV. Returns s3://... string or Path for local.
    """
    s = get_csv_settings()
    processed_dir = s.S3_PROCESSED_CSV_PATH or str(LOCAL_PROCESSED_DIR)

    # S3 branch
    if isinstance(processed_dir, str) and _is_s3_uri(processed_dir):
        try:
            fs = _get_s3_fs()
            prefix = processed_dir.rstrip("/")
            candidates = fs.glob(f"{prefix}/processed_*.csv") or fs.glob(f"{prefix}/*.csv")
            candidates = [(_ensure_s3_uri(str(x))) for x in candidates]
            if candidates:
                dated: List[Tuple[pd.Timestamp, str]] = []
                for p in candidates:
                    name = str(p).split("/")[-1]
                    m = DATE_RE.search(name)
                    if m:
                        try:
                            y, mo, d = map(int, m.groups())
                            dated.append((pd.Timestamp(year=y, month=mo, day=d), p))
                        except Exception:
                            continue
                if dated:
                    dated.sort(key=lambda x: x[0], reverse=True)
                    return dated[0][1]
                return sorted(candidates)[-1]
        except Exception:
            logger.exception("S3 listing for processed CSVs failed; falling back to local")

    # local fallback
    local_dir = Path(LOCAL_PROCESSED_DIR)
    if not local_dir.exists():
        return None
    if not _is_path_safe(local_dir, [Path(LOCAL_PROCESSED_DIR)]):
        logger.error("Local processed dir not allowed: %s", local_dir)
        return None

    candidates = list(local_dir.glob("processed_*.csv"))
    if not candidates:
        candidates = list(local_dir.glob("*.csv"))
        if not candidates:
            return None

    dated_local: List[Tuple[pd.Timestamp, Path]] = []
    for p in candidates:
        m = DATE_RE.search(p.name)
        if m:
            try:
                y, mo, d = map(int, m.groups())
                dated_local.append((pd.Timestamp(year=y, month=mo, day=d), p))
            except Exception:
                continue
    if dated_local:
        dated_local.sort(key=lambda x: x[0], reverse=True)
        return dated_local[0][1]
    return max(candidates, key=lambda p: p.stat().st_mtime)

# --------------------------
# Load processed EOD DataFrame (double-checked locking & cache TTL)
# --------------------------
def load_processed_df(force_reload: bool = False) -> Optional[pd.DataFrame]:
    """
    Load and cache the latest processed EOD CSV.

    Args:
        force_reload: If True, bypass cache and reload.

    Returns:
        pd.DataFrame or None on failure.

    Raises:
        ValueError if the CSV is missing required columns.
    """
    global _EOD_DF, _EOD_PATH, _EOD_CACHE_TS

    # Fast-path
    if _EOD_DF is not None and not force_reload and _is_cache_fresh(_EOD_CACHE_TS):
        return _EOD_DF

    # Determine path and perform IO outside lock
    p = find_latest_processed_eod()
    if not p:
        _EOD_DF = None
        _EOD_PATH = None
        _EOD_CACHE_TS = 0.0
        _audit_log("load_processed_df", "eod_csv", "failure", {"reason": "no file"})
        return None

    try:
        df = _read_csv_resilient(p)
    except Exception as e:
        _increment_error_metric("csv_utils.load_processed_df.read_failure")
        logger.exception("Failed to read processed EOD CSV %s", p)
        return None

    cols = {c.strip() for c in df.columns}
    # Check for required column types (more flexible than exact names)
    has_symbol_col = any(c.strip().lower() in ("symbol", "sym", "ticker") for c in df.columns)
    has_desc_col = any(c.strip().lower() in ("company_name", "company", "description", "company name") for c in df.columns)
    
    if not (has_symbol_col and has_desc_col):
        missing = []
        if not has_symbol_col:
            missing.append("symbol-like column")
        if not has_desc_col:
            missing.append("description/company-like column")
        _increment_error_metric("csv_utils.load_processed_df.validation")
        raise ValueError(f"Processed EOD CSV missing required column types: {missing}. Found columns: {sorted(list(cols))}")

    # Commit to cache under lock
    with _eod_cache_lock:
        if _EOD_DF is None or force_reload or not _is_cache_fresh(_EOD_CACHE_TS):
            df.columns = [c.strip() for c in df.columns]
            _EOD_DF = df
            _EOD_PATH = p
            _EOD_CACHE_TS = _now()
            _safe_metric("load_processed_df", 1)
            _audit_log("load_processed_df.finish", str(p), "success", {"rows": len(df)})
    return _EOD_DF

# --------------------------
# Market snapshot helpers
# --------------------------
def _normalize_value(v):
    return None if pd.isna(v) else v

def _extract_date_from_filename(path: Optional[Union[Path, str]]) -> Optional[str]:
    if not path:
        return None
    try:
        name = path.name if isinstance(path, Path) else str(path).rstrip("/").split("/")[-1]
    except Exception:
        name = str(path).rstrip("/").split("/")[-1]
    m = DATE_RE.search(name)
    if not m:
        return None
    try:
        y, mo, d = map(int, m.groups())
        return pd.Timestamp(year=y, month=mo, day=d).strftime("%d-%b-%Y")
    except Exception:
        return None

def get_market_snapshot(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Lookup a symbol and return a market snapshot dict, or None if not found.
    """
    if not symbol:
        return None
    df = load_processed_df()
    if df is None:
        return None

    s = str(symbol).strip().upper()
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

    if rows.empty and comp_cols:
        for cc in comp_cols:
            mask = df[cc].astype(str).str.upper().str.contains(re.escape(s))
            rows = df[mask]
            if not rows.empty:
                break

    if rows.empty:
        return None

    row = rows.iloc[0]

    def tryget(*names):
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
        "volume_24h_rs_cr": tryget("volume_24h_rs_cr", "volume_24h", "Volume_24H", "volume (rs cr.)", "turnover"),
        "all_time_high": tryget("all_time_high", "all time high (rs.)"),
        "atr_pct": tryget("atr_pct", "atr_14d", "atr% (24 hrs)"),
        "relative_vol": tryget("relative_vol", "relative vol"),
        "vol_change_pct": tryget("vol_change_pct", "volume_change_24h_pct", "vol. change (24 hrs)"),
        "volatility": tryget("volatility"),
    }

    snapshot["market_snapshot_date"] = _extract_date_from_filename(_EOD_PATH) if _EOD_PATH else None

    for k in ["price", "vwap", "mcap_rs_cr", "volume_24h_rs_cr", "all_time_high", "atr_pct", "relative_vol", "vol_change_pct", "volatility"]:
        if snapshot.get(k) is not None:
            try:
                snapshot[k] = float(snapshot[k])
            except Exception:
                pass

    _safe_metric("get_market_snapshot", 1)
    _audit_log("get_market_snapshot", snapshot.get("symbol") or str(symbol), "success")
    return snapshot

# --------------------------
# Indices loader & lookup
# --------------------------
def load_indices_df(force_reload: bool = False) -> Optional[pd.DataFrame]:
    """
    Load and cache indices CSV used for mapping symbols -> indices.
    Validates presence of 'Symbol' and 'Description' columns.
    """
    global _INDICES_DF, _INDICES_PATH, _INDICES_CACHE_TS

    if _INDICES_DF is not None and not force_reload and _is_cache_fresh(_INDICES_CACHE_TS):
        return _INDICES_DF

    settings = get_csv_settings()
    static_dir = settings.S3_STATIC_CSV_PATH or str(LOCAL_STATIC_DIR)
    path_to_load: Optional[Union[str, Path]] = None

    if isinstance(static_dir, str) and _is_s3_uri(static_dir):
        try:
            fs = _get_s3_fs()
            prefix = static_dir.rstrip("/")
            entries = fs.glob(f"{prefix}/*.csv")
            for e in entries:
                name = str(e).split("/")[-1]
                if "sector" in name.lower() or "index" in name.lower():
                    path_to_load = _ensure_s3_uri(str(e))
                    break
        except Exception:
            logger.exception("S3 listing failed for static indices")
    else:
        local_dir = Path(static_dir)
        if local_dir.is_dir() and _is_path_safe(local_dir, [Path(LOCAL_STATIC_DIR)]):
            for p in local_dir.glob("*.csv"):
                if "sector" in p.name.lower() or "index" in p.name.lower():
                    path_to_load = p
                    break

    if not path_to_load:
        _increment_error_metric("csv_utils.load_indices_df.not_found")
        return None

    try:
        df = _read_csv_resilient(path_to_load)
    except Exception:
        _increment_error_metric("csv_utils.load_indices_df.read_failure")
        logger.exception("Failed to read indices CSV %s", path_to_load)
        return None

    cols = {c.strip() for c in df.columns}
    if "Symbol" not in cols or "Description" not in cols:
        _increment_error_metric("csv_utils.load_indices_df.validation")
        raise ValueError("Indices CSV missing required 'Symbol' and/or 'Description' columns")

    with _indices_cache_lock:
        if _INDICES_DF is None or force_reload or not _is_cache_fresh(_INDICES_CACHE_TS):
            df.columns = [c.strip() for c in df.columns]
            _INDICES_DF = df
            _INDICES_PATH = path_to_load
            _INDICES_CACHE_TS = _now()
            _safe_metric("load_indices_df", 1)
            _audit_log("load_indices_df.finish", str(path_to_load), "success", {"rows": len(df)})
    return _INDICES_DF

def get_indices_for_symbol(symbol: str) -> Tuple[str, str]:
    """
    Return (BroadIndex, Sector) for a symbol.
    """
    df = load_indices_df()
    if df is None:
        return ("Uncategorised Index", "Uncategorised Sector")
    s = str(symbol).strip().upper()

    rows = df[df['Symbol'].astype(str).str.upper() == s] if "Symbol" in df.columns else pd.DataFrame()
    if rows.empty and "Description" in df.columns:
        rows = df[df['Description'].astype(str).str.upper().str.contains(re.escape(s))]

    if rows.empty:
        symbol_cols = [c for c in df.columns if c.strip().lower() in ("symbol", "sym", "ticker")]
        desc_cols = [c for c in df.columns if c.strip().lower() in ("description", "company", "name")]
        if symbol_cols:
            for sc in symbol_cols:
                rows = df[df[sc].astype(str).str.upper() == s]
                if not rows.empty:
                    break
        if rows.empty and desc_cols:
            for dc in desc_cols:
                rows = df[df[dc].astype(str).str.upper().str.contains(re.escape(s))]
                if not rows.empty:
                    break

    if rows.empty:
        return ("Uncategorised Index", "Uncategorised Sector")

    row = rows.iloc[0]
    sector_raw = None
    for cand in ("SectorialIndex", "Sector", "Sector Name", "Sectorial", "sector"):
        if cand in row.index:
            sector_raw = row.get(cand)
            break
    sector = "Uncategorised Sector"
    if pd.notna(sector_raw) and str(sector_raw).strip():
        sector = str(sector_raw).split(",")[0].strip() or "Uncategorised Sector"

    broad = "Uncategorised Index"
    for b in BROAD_PRIORITY:
        if b in row.index and pd.notna(row.get(b)) and str(row.get(b)).strip().lower() in ("yes", "y", "true", "1"):
            broad = b
            break

    _safe_metric("get_indices_for_symbol", 1)
    _audit_log("get_indices_for_symbol", symbol, "success")
    return (broad, sector)

# --------------------------
# Convenience
# --------------------------
def list_symbols(limit: Optional[int] = None) -> List[str]:
    """
    List unique symbols (or company names if symbol column missing).
    """
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
    return s[:limit] if limit else s

def format_snapshot_for_display(symbol: str) -> str:
    """
    Pretty-print a market snapshot for display/CLI.
    """
    snap = get_market_snapshot(symbol)
    if not snap:
        return f"{symbol}: No market snapshot available."
    broad, sector = get_indices_for_symbol(symbol)

    def arrow(val):
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

    s = f"{snap.get('symbol')} | {snap.get('company_name')}\n"
    s += f"{broad} | {sector}\n\n"
    s += f"📊 Market Snapshot: |{snap.get('market_snapshot_date')}, EOD|\n"
    s += f"Price: ₹{price} | {change1d}% (1D) {arrow(change1d)} | {change1w}% (1W) {arrow(change1w)}\n"
    s += f"Volume (24 Hrs): ₹{snap.get('volume_24h_rs_cr')} Cr\n"
    s += f"Mcap: ₹{snap.get('mcap_rs_cr')} Cr | Rank: #{snap.get('rank')}\n\n"
    s += f"VWAP: ₹{snap.get('vwap')} | ATR (14D): {snap.get('atr_pct')}%\n"
    s += f"Relative Vol: {snap.get('relative_vol')} | Vol Change: {snap.get('vol_change_pct')}%\n"
    s += f"Volatility: {snap.get('volatility')}%\n"
    return s.strip()

# --------------------------
# Async wrappers
# --------------------------
async def async_load_processed_df(force_reload: bool = False) -> Optional[pd.DataFrame]:
    """
    Async wrapper for load_processed_df preserving correlation_id context.
    """
    if correlation_id_var.get() == 'standalone':
        correlation_id_var.set(f"task-{int(time.time())}-{os.getpid()}")
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func = partial(load_processed_df, force_reload)
    return await loop.run_in_executor(_get_executor(), ctx.run, func)

async def async_get_market_snapshot(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Async wrapper for get_market_snapshot preserving correlation_id context.
    """
    if correlation_id_var.get() == 'standalone':
        correlation_id_var.set(f"task-{int(time.time())}-{os.getpid()}")
    loop = asyncio.get_running_loop()
    ctx = __import__("contextvars").copy_context()
    func = partial(get_market_snapshot, symbol)
    return await loop.run_in_executor(_get_executor(), ctx.run, func)

# --------------------------
# Health check
# --------------------------
def health_check() -> Dict[str, Any]:
    """
    Returns health report suitable for liveness/readiness checks.
    """
    report: Dict[str, Any] = {
        "status": "healthy",
        "version": __version__,
        "timestamp": _now(),
        "checks": {}
    }
    try:
        report["checks"]["eod_loaded"] = {"loaded": _EOD_DF is not None, "path": str(_EOD_PATH) if _EOD_PATH else None}
        report["checks"]["indices_loaded"] = {"loaded": _INDICES_DF is not None, "path": str(_INDICES_PATH) if _INDICES_PATH else None}
        report["checks"]["s3_circuit_breaker"] = {"state": _s3_read_circuit_breaker.state}
        report["checks"]["cache_ttl_seconds"] = _CACHE_TTL_SECONDS
        report["checks"]["metrics_worker_alive"] = _METRICS_WORKER_THREAD is not None and _METRICS_WORKER_THREAD.is_alive()
    except Exception as e:
        report["status"] = "unhealthy"
        report["error"] = str(e)
    return report

# --------------------------
# Lifecycle helpers
# --------------------------
def preload_settings():
    """
    Initialize resources on app startup. Delegates to csv_processor settings.
    """
    get_csv_settings()
    _start_metrics_worker()

def shutdown_csv_utils():
    """
    Minimal shutdown hook. Primary resource shutdown is handled by csv_processor.
    """
    try:
        # stop metrics worker
        if _METRICS_QUEUE:
            try:
                _METRICS_QUEUE.put_nowait(None)
            except Exception:
                pass
        if _METRICS_WORKER_THREAD:
            _METRICS_WORKER_THREAD.join(timeout=2)
    except Exception:
        logger.exception("Error during csv_utils shutdown")

atexit.register(shutdown_csv_utils)

# --------------------------
# CLI entrypoint
# --------------------------
def _cli():
    parser = argparse.ArgumentParser(description="csv_utils - helpers for processed EOD CSVs")
    parser.add_argument("--symbol", help="stock symbol or company name to lookup")
    parser.add_argument("--list", action="store_true", help="list available symbols (first 200)")
    parser.add_argument("--json", action="store_true", help="output JSON for --symbol")
    parser.add_argument("--reload", action="store_true", help="force reload cached CSVs")
    parser.add_argument("--limit", type=int, default=200, help="limit for --list")
    args = parser.parse_args()

    try:
        from .csv_processor import configure_logging_from_settings
        configure_logging_from_settings(force=True)
    except Exception:
        logging.basicConfig(level=logging.INFO)

    if args.reload:
        load_processed_df(force_reload=True)
        load_indices_df(force_reload=True)

    if args.list:
        for x in list_symbols(limit=args.limit):
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
            print(json.dumps(snap, default=lambda o: None if pd.isna(o) else (int(o) if hasattr(o, "astype") else str(o))))
            return
        print(format_snapshot_for_display(args.symbol))
        return

    parser.print_help()

# public API
__all__ = [
    "find_latest_processed_eod", "load_processed_df", "async_load_processed_df",
    "get_market_snapshot", "async_get_market_snapshot", "load_indices_df",
    "get_indices_for_symbol", "list_symbols", "format_snapshot_for_display",
    "preload_settings", "reset_cached_settings", "health_check", "shutdown_csv_utils"
]

if __name__ == "__main__":
    _cli()