# backend/services/aws_utils.py
"""
Shared S3 utilities (Diamond-grade R-01).

This module centralizes S3 interaction for the codebase. It is designed to be:
 - robust (retries + transient detection),
 - thread-safe (cached fsspec FS and a semaphore),
 - configurable (S3 options via JSON string),
 - minimally surprising (returns booleans for recoverable flows; raises for fatal misconfig).

Primary callers:
 - backend.services.csv_processor
 - backend.services.image_processor
 - other modules that require simple S3 list/download/upload primitives

Important notes:
 - fsspec is the primary interface (s3fs under the hood). If fsspec is not installed,
   functions will either raise (get_s3_fs) or return False for download/upload functions.
 - You can pass S3 options as a JSON string (matching csv_processor's S3_STORAGE_OPTIONS),
   e.g. '{"key": "...", "secret": "...", "endpoint_url": "...", "anon": false}'
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("backend.services.aws_utils")
logger.addHandler(logging.NullHandler())

# Optional imports (graceful degradation)
try:
    import fsspec
except Exception:
    fsspec = None  # type: ignore

# boto3 optional for a convenience helper; not required for main flows
try:
    import boto3
    from botocore.exceptions import ClientError
except Exception:
    boto3 = None
    ClientError = None  # type: ignore

# --- module-level caches and locks for thread-safety ---
_S3_FS_LOCK = threading.Lock()
_FSSPEC_FS: Optional[Any] = None
_S3_SEMAPHORE: Optional[threading.Semaphore] = None
_S3_SEMAPHORE_LOCK = threading.Lock()

# Retry/backoff defaults (tunable by callers by re-calling wrapper)
_DEFAULT_RETRIES = 2
_DEFAULT_BACKOFF = 1.0  # seconds

# Allowed option keys when parsing S3 options JSON
_ALLOWED_S3_OPTION_KEYS = {"key", "secret", "endpoint_url", "anon", "client_kwargs"}


# -------------------------
# Helper utilities
# -------------------------
def _sanitize_for_logging(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Redact sensitive-looking keys from a dict for logging.
    """
    out = dict(d)
    for k in list(out.keys()):
        if any(tok in k.lower() for tok in ("key", "secret", "token", "password", "credential")):
            out[k] = "***REDACTED***"
    return out


def _parse_s3_options_from_json(s3_options_json: Optional[str]) -> Dict[str, Any]:
    """
    Parse a JSON string of S3 options into a filtered dict accepted by fsspec.
    This is intentionally conservative — we only keep keys in the allowed set.
    """
    if not s3_options_json:
        return {}
    try:
        parsed = json.loads(s3_options_json)
        if not isinstance(parsed, dict):
            logger.warning("S3 options JSON did not parse to a dict; ignoring")
            return {}
        filtered = {k: v for k, v in parsed.items() if k in _ALLOWED_S3_OPTION_KEYS}
        # Ensure anon boolean is boolean, accept string forms
        if "anon" in filtered and isinstance(filtered["anon"], str):
            filtered["anon"] = filtered["anon"].lower() in ("1", "true", "yes", "on", "t")
        logger.debug("Parsed S3 options (redacted) %s", _sanitize_for_logging(filtered))
        return filtered
    except Exception:
        logger.exception("Failed to parse S3 options JSON; ignoring options")
        return {}


def _is_transient_error(exc: Exception) -> bool:
    """
    Heuristic to detect transient S3/network errors for retry decisions.
    Covers common keywords and some exception types.
    """
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    tokens = ("timeout", "timedout", "connect", "connection", "throttl", "503", "serviceunavailable", "transient", "busy", "temporar")
    if any(t in name for t in tokens) or any(t in msg for t in tokens):
        return True
    # boto3/ClientError heuristics
    if ClientError is not None and isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("Throttling", "ThrottlingException", "RequestTimeout", "InternalError", "ServiceUnavailable"):
            return True
    return False


def _retry_op(fn, retries: int = _DEFAULT_RETRIES, backoff: float = _DEFAULT_BACKOFF):
    """
    Execute `fn()` with retries and exponential backoff for detected transient errors.
    Re-raises the final exception if all retries exhausted.
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if not _is_transient_error(e):
                # non-transient — fail fast
                logger.debug("Non-transient error encountered during S3 operation: %s", e)
                raise
            # transient — backoff and retry if possible
            logger.warning("Transient S3 error attempt %d/%d: %s", attempt + 1, retries + 1, e)
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            # exhausted
            logger.error("Retries exhausted for S3 operation: %s", last_exc)
            raise
    # Shouldn't reach here, but raise last exc if it exists
    if last_exc:
        raise last_exc
    raise RuntimeError("Retry wrapper failed unexpectedly")


# -------------------------
# Public API
# -------------------------
def get_s3_fs(s3_options_json: Optional[str] = None, concurrency: int = 4) -> Tuple[Any, threading.Semaphore]:
    """
    Return a cached fsspec filesystem for S3 and a semaphore for concurrency control.

    Args:
      s3_options_json: optional JSON string of allowed S3 options (key, secret, endpoint_url, anon, client_kwargs)
      concurrency: integer size for the returned semaphore (only used on first creation)

    Returns:
      (fs, semaphore) where `fs` is a fsspec filesystem (S3) instance.

    Raises:
      RuntimeError if fsspec is not installed or filesystem creation fails.
    """
    global _FSSPEC_FS, _S3_SEMAPHORE
    if fsspec is None:
        raise RuntimeError("fsspec is required for S3 operations but is not installed")

    with _S3_FS_LOCK:
        if _FSSPEC_FS is None:
            opts = _parse_s3_options_from_json(s3_options_json)
            try:
                # create and cache fsspec filesystem
                _FSSPEC_FS = fsspec.filesystem("s3", **opts)
                logger.debug("Created fsspec s3 filesystem (opts redacted) %s", _sanitize_for_logging(opts))
            except TypeError as te:
                logger.exception("Failed to create S3 filesystem with provided options: %s", _sanitize_for_logging(opts))
                raise
            except Exception:
                logger.exception("Failed to create S3 filesystem")
                raise
        # create semaphore lazily
        if _S3_SEMAPHORE is None:
            with _S3_SEMAPHORE_LOCK:
                if _S3_SEMAPHORE is None:
                    _S3_SEMAPHORE = threading.Semaphore(max(1, int(concurrency)))
    return _FSSPEC_FS, _S3_SEMAPHORE  # type: ignore[return-value]


def list_s3_prefix(s3_prefix: str, s3_options_json: Optional[str] = None, concurrency: int = 4) -> List[str]:
    """
    Return a sorted list of S3 URIs under the given prefix.

    - Accepts either a prefix that ends with the folder (e.g., 's3://bucket/path') or
      a deeper path. It will attempt to list recursively.
    - Returns an empty list on empty / not-found.
    - Raises RuntimeError if fsspec not available or filesystem creation fails.

    Note: This function uses the cached fsspec filesystem and performs retries for transient errors.
    """
    fs, sem = get_s3_fs(s3_options_json, concurrency)
    def _op():
        # fsspec.glob returns path-like objects or strings depending on implementation
        objs = fs.glob(f"{s3_prefix.rstrip('/')}/**/*")
        out: List[str] = []
        for o in objs:
            s = str(o)
            if s.endswith("/"):
                continue
            if not s.startswith("s3://"):
                s = "s3://" + s
            out.append(s)
        return sorted(out)
    return _retry_op(_op)


def download_prefix_to_local(
    s3_prefix: str,
    local_dir: Path,
    s3_options_json: Optional[str] = None,
    concurrency: int = 4
) -> bool:
    """
    Download all objects under s3_prefix into local_dir.

    Returns:
      True if at least one file was downloaded successfully,
      False if prefix empty or fsspec unavailable or unrecoverable.
    The function logs exceptions and uses retries for transient errors.
    """
    try:
        fs, sem = get_s3_fs(s3_options_json, concurrency)
    except RuntimeError as e:
        logger.debug("S3 download not available: %s", e)
        return False

    def _op() -> bool:
        entries = fs.glob(f"{s3_prefix.rstrip('/')}/**/*")
        # Filter out "directory" markers and keep file-like entries
        files = [e for e in entries if not str(e).endswith("/")]
        if not files:
            return False
        local_dir.mkdir(parents=True, exist_ok=True)
        for key in files:
            key_s = str(key)
            name = key_s.split("/")[-1]
            if not name:
                continue
            dest = local_dir / name
            # Acquire semaphore for concurrency control if provided
            # Note: fsspec methods are not necessarily thread-safe; semaphore protects application-level concurrency
            if sem:
                acquired = sem.acquire(timeout=60)
            else:
                acquired = True
            try:
                with fs.open(key_s, "rb") as r, open(dest, "wb") as w:
                    while True:
                        chunk = r.read(16 * 1024)
                        if not chunk:
                            break
                        w.write(chunk)
            finally:
                if sem and acquired:
                    try:
                        sem.release()
                    except Exception:
                        pass
        return True

    try:
        return _retry_op(_op)
    except Exception:
        logger.exception("S3 download failed for %s", s3_prefix)
        return False


def upload_dir_to_s3(
    local_dir: Path,
    s3_prefix: str,
    s3_options_json: Optional[str] = None,
    concurrency: int = 4
) -> bool:
    """
    Upload all files (non-recursive) from local_dir into s3_prefix.

    Returns True on success (or if local_dir is empty -> False).
    Logs exceptions and returns False on failure.
    """
    try:
        fs, sem = get_s3_fs(s3_options_json, concurrency)
    except RuntimeError as e:
        logger.debug("S3 upload not available: %s", e)
        return False

    def _op() -> bool:
        if not local_dir.exists() or not local_dir.is_dir():
            return False
        prefix = s3_prefix.rstrip("/")
        for p in sorted(local_dir.glob("*")):
            if not p.is_file():
                continue
            key = f"{prefix}/{p.name}"
            if sem:
                acquired = sem.acquire(timeout=60)
            else:
                acquired = True
            try:
                with open(p, "rb") as r, fs.open(key, "wb") as w:
                    while True:
                        chunk = r.read(16 * 1024)
                        if not chunk:
                            break
                        w.write(chunk)
            finally:
                if sem and acquired:
                    try:
                        sem.release()
                    except Exception:
                        pass
        return True

    try:
        return _retry_op(_op)
    except Exception:
        logger.exception("S3 upload failed for %s", s3_prefix)
        return False


def put_bytes_to_s3(bucket: str, key: str, data: bytes, s3_options_json: Optional[str] = None) -> bool:
    """
    Convenience helper to put raw bytes into an S3 key. Uses boto3 if available; otherwise
    tries fsspec as a fallback. Returns True on success, False on failure.

    This helper is useful for small objects such as generated CSVs, JSON manifests, or small images.
    """
    # First try boto3 (if available) for a straightforward put_object
    if boto3 is not None:
        try:
            opts = _parse_s3_options_from_json(s3_options_json)
            # Create a boto3 client; allow endpoint_url via opts
            client_kwargs = {}
            if "endpoint_url" in opts:
                client_kwargs["endpoint_url"] = opts["endpoint_url"]
            # If key/secret provided we could pass them via aws_access_key_id etc. but skip for brevity
            s3_client = boto3.client("s3", **client_kwargs)
            s3_client.put_object(Bucket=bucket, Key=key, Body=data)
            return True
        except Exception as e:
            logger.exception("boto3 put_object failed for s3://%s/%s: %s", bucket, key, e)
            # fall through to fsspec fallback

    # Fallback to fsspec if boto3 unavailable or failed
    if fsspec is None:
        logger.debug("Neither boto3 nor fsspec available for put_bytes_to_s3")
        return False

    try:
        fs, sem = get_s3_fs(s3_options_json)
    except RuntimeError as e:
        logger.debug("S3 filesystem unavailable: %s", e)
        return False

    def _op():
        key_uri = f"s3://{bucket.rstrip('/')}/{key.lstrip('/')}"
        if _S3_SEMAPHORE:
            acquired = _S3_SEMAPHORE.acquire(timeout=60)
        else:
            acquired = True
        try:
            with fs.open(key_uri, "wb") as w:
                w.write(data)
            return True
        finally:
            if _S3_SEMAPHORE and acquired:
                try:
                    _S3_SEMAPHORE.release()
                except Exception:
                    pass

    try:
        return _retry_op(_op)
    except Exception:
        logger.exception("Failed to write bytes to s3://%s/%s", bucket, key)
        return False