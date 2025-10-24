#!/usr/bin/env python3
"""
image_utils.py - Helpers to serve processed logos and banners for frontend

Behaviour:
 - Prefer S3 (if configured and reachable) then local filesystem.
 - Returns a stable ImageCandidate dataclass:
     { filename, s3_uri?, local_path?, public_url? }
 - Returned public_url will use CDN_BASE_URL when candidate is on S3 (URL-encoded),
   otherwise the local static URL (/static/images/...) for local files.

This updated version:
 - does NOT configure global logging (library should not call basicConfig)
 - uses deterministic logging (logger)
 - S3 exists retry/backoff + boto3 fallback
 - normalization of S3 prefix and URL-encoding of object keys
 - TTL cache for hot lookups (configurable via env)
 - async-compatible wrappers (safe for FastAPI via ThreadPoolExecutor)
 - unified metrics wrapper (Prometheus optional; fallback in-memory)
 - .env auto-load only in non-production (but callers may choose to move .env loading to app startup)
"""

from __future__ import annotations

import os
import re
import time
import asyncio
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Any, Dict
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor

# Optionally load .env only in non-production (call at import-time for convenience).
# For maximum safety you may remove this and load .env in app startup instead.
try:
    if (os.getenv("ENVIRONMENT", "dev") or "").lower() != "prod":
        from dotenv import load_dotenv  # type: ignore
        _env_path = Path(__file__).resolve().parents[2] / ".env"
        if _env_path.exists():
            load_dotenv(dotenv_path=_env_path)
except Exception:
    # ignore; production environment shouldn't rely on .env
    pass

# Optional dependencies
try:
    import fsspec  # type: ignore
except Exception:
    fsspec = None

try:
    import boto3  # type: ignore
    from botocore.exceptions import ClientError  # type: ignore
except Exception:
    boto3 = None
    ClientError = Exception  # type: ignore

# Logging: library should not call basicConfig; apps will configure logging centrally.
import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("image_utils")
# Respect user/app configured level if present, otherwise set to env-provided level.
if not logger.handlers:
    # if no handlers are attached, avoid setting basicConfig; set logger level only.
    logger.setLevel(LOG_LEVEL)

# -------------------------
# Config
# -------------------------
BASE = Path(__file__).resolve().parents[1]
PROCESSED_LOGOS = BASE / "input_data" / "images" / "processed_images" / "processed_logos"
PROCESSED_BANNERS = BASE / "input_data" / "images" / "processed_images" / "processed_banners"

URL_BASE = "/static/images"
PLACEHOLDER_LOGO = "default_logo.png"
PLACEHOLDER_BANNER = "default_banner.webp"

# read environment at import time (can be overridden by callers setting env before import)
S3_PROCESSED_IMAGES_PATH = (os.getenv("S3_PROCESSED_IMAGES_PATH") or "").strip()
CDN_BASE_URL = (os.getenv("CDN_BASE_URL") or "").strip()

# Cache TTL (seconds)
IMAGE_CACHE_TTL_SECONDS = int(os.getenv("IMAGE_CACHE_TTL_SECONDS", "3600"))

# Thread pool for blocking I/O when used from async frameworks
_thread_pool: Optional[ThreadPoolExecutor] = None


def _ensure_thread_pool(max_workers: int = 6) -> ThreadPoolExecutor:
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    return _thread_pool


def shutdown_thread_pool(wait: bool = False) -> None:
    """Shutdown the module-level thread pool. Call from app shutdown if used."""
    global _thread_pool
    if _thread_pool is not None:
        try:
            _thread_pool.shutdown(wait=wait)
        except Exception as e:
            logger.warning("Error shutting down thread pool: %s", e)
        _thread_pool = None


# -------------------------
# Small helpers
# -------------------------


def normalize_symbol(sym: str) -> str:
    """Normalize trading symbol to uppercase with underscores."""
    if not sym:
        return ""
    return re.sub(r"\s+", "_", sym.strip().upper())


def slugify_company(name: str) -> str:
    """Make a lowercase slug from company name (safe for filenames)."""
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


# -------------------------
# Candidate filename generators
# -------------------------


def candidate_logo_filenames(symbol: str, company_name: str) -> List[str]:
    sym = normalize_symbol(symbol)
    slug = slugify_company(company_name)
    cands: List[str] = []
    if sym:
        cands.append(f"{sym}_logo.png")
        cands.append(f"{sym.lower()}_logo.png")
    if slug:
        cands.append(f"{slug}_logo.png")
    return cands


def candidate_banner_filenames(symbol: str, company_name: str) -> List[str]:
    sym = normalize_symbol(symbol)
    slug = slugify_company(company_name)
    cands: List[str] = []
    if sym:
        cands.append(f"{sym}_banner.webp")
        cands.append(f"{sym.lower()}_banner.webp")
        cands.append(f"{sym}_banner.png")
    if slug:
        cands.append(f"{slug}_banner.webp")
        cands.append(f"{slug}_banner.png")
    return cands


# -------------------------
# Stable return type
# -------------------------


@dataclass
class ImageCandidate:
    """Stable return type for image lookups."""
    filename: str
    s3_uri: Optional[str] = None
    local_path: Optional[str] = None
    public_url: Optional[str] = None


# -------------------------
# S3 helpers (normalization & encoding)
# -------------------------


def _ensure_s3_uri(val: str) -> str:
    """Normalize a value into a proper s3://bucket/key URI string, trimmed."""
    if not isinstance(val, str):
        return val
    v = val.strip()
    if not v:
        return v
    if v.lower().startswith("s3://"):
        return v.rstrip("/")
    # if looks like bucket/key, prepend scheme
    if "/" in v:
        return "s3://" + v.lstrip("/").rstrip("/")
    return v


def _s3_object_key_from_s3_uri(s3_uri: str) -> str:
    """Return object key (path after bucket) for s3://bucket/key or bucket/key."""
    if not s3_uri:
        return ""
    u = s3_uri[len("s3://"):] if s3_uri.startswith("s3://") else s3_uri
    parts = u.split("/", 1)
    return parts[1] if len(parts) > 1 else ""


def _encode_key_for_cdn(key: str) -> str:
    """URL-encode the object key portion for safe CDN usage (preserve slashes)."""
    if not key:
        return ""
    parts = key.split("/")
    return "/".join(quote(p, safe="") for p in parts)


def _fsspec_exists_key(s3_uri: str) -> str:
    """
    Return the key to pass to fsspec.filesystem("s3").exists().
    Some fsspec implementations accept full s3://bucket/key; others expect bucket/key or just key.
    We'll return bucket/key (no leading scheme) which is broadly accepted.
    """
    u = s3_uri[len("s3://"):] if s3_uri.startswith("s3://") else s3_uri
    return u.lstrip("/")


# -------------------------
# S3 exists with retry/backoff (fsspec or boto3)
# -------------------------


def _s3_exists_with_retry(fs, s3_uri: str, attempts: int = 3, base_delay: float = 0.2) -> bool:
    """
    Check existence with retries. Supports fsspec filesystem object (preferred),
    otherwise uses boto3 head_object as fallback.
    """
    last_err = None
    # prefer fsspec if provided
    if fsspec is not None and fs is not None:
        key_for_exists = _fsspec_exists_key(s3_uri)
        for attempt in range(1, attempts + 1):
            try:
                exists = fs.exists(key_for_exists)
                logger.debug("fsspec.exists(%s) -> %s (attempt %d/%d)", key_for_exists, exists, attempt, attempts)
                return bool(exists)
            except Exception as e:
                last_err = e
                logger.warning("fsspec.exists failed for %s (attempt %d/%d): %s", key_for_exists, attempt, attempts, e)
                if attempt < attempts:
                    time.sleep(base_delay * (2 ** (attempt - 1)))
        logger.error("fsspec.exists ultimately failed for %s after %d attempts: %s", key_for_exists, attempts, last_err)
        return False

    # else try boto3 head_object
    if boto3 is not None:
        # parse bucket/key
        try:
            s = s3_uri[len("s3://"):] if s3_uri.startswith("s3://") else s3_uri
            bucket, key = s.split("/", 1)
        except Exception as e:
            logger.warning("Invalid s3_uri for boto3 head_object: %s (%s)", s3_uri, e)
            return False
        s3c = boto3.client("s3")
        for attempt in range(1, attempts + 1):
            try:
                s3c.head_object(Bucket=bucket, Key=key)
                logger.debug("boto3.head_object succeeded for %s/%s", bucket, key)
                return True
            except ClientError as ce:
                code = getattr(ce, "response", {}).get("Error", {}).get("Code", "")
                if code == "404" or code == "NotFound":
                    logger.debug("boto3 head_object not found for %s/%s", bucket, key)
                    return False
                last_err = ce
                logger.warning("boto3 head_object error for %s/%s (attempt %d/%d): %s", bucket, key, attempt, attempts, ce)
                if attempt < attempts:
                    time.sleep(base_delay * (2 ** (attempt - 1)))
            except Exception as e:
                last_err = e
                logger.warning("boto3 head_object exception for %s/%s: %s", bucket, key, e)
                if attempt < attempts:
                    time.sleep(base_delay * (2 ** (attempt - 1)))
        logger.error("boto3 head_object ultimately failed for %s after %d attempts: %s", s3_uri, attempts, last_err)
        return False

    # No method available
    logger.debug("No S3 check method available (fsspec and boto3 missing)")
    return False


# -------------------------
# TTL cache decorator (simple)
# -------------------------


def ttl_cache(ttl_seconds: int, maxsize: int = None):
    """Thread-safe LRU+TTL cache decorator.

    - ttl_seconds: time-to-live in seconds for each entry
    - maxsize: maximum number of entries to keep (None -> unbounded). If provided,
      the cache evicts least-recently-used entries when full.

    Provides a `cache_clear()` method on the wrapped function to clear the cache.
    """
    from collections import OrderedDict
    import threading

    # read default maxsize from env if not passed
    if maxsize is None:
        try:
            maxsize = int(os.getenv("IMAGE_CACHE_MAXSIZE", "1024"))
        except Exception:
            maxsize = 1024

    def decorator(func: Callable):
        cache: OrderedDict = OrderedDict()
        lock = threading.RLock()

        @wraps(func)
        def wrapper(*args, **kwargs):
            # build a cache key that is reasonably safe for common args
            try:
                key = (args, tuple(sorted(kwargs.items()))) if kwargs else args
            except Exception:
                # if key is not hashable, skip caching
                return func(*args, **kwargs)

            now = time.time()

            with lock:
                # hit -> refresh order and return if TTL not expired
                if key in cache:
                    ts, val = cache.pop(key)
                    if now - ts < ttl_seconds:
                        # reinsert as most-recently-used
                        cache[key] = (ts, val)
                        logger.debug("Cache HIT for %s args", func.__name__)
                        return val
                    else:
                        # expired
                        logger.debug("Cache EXPIRED for %s args", func.__name__)
                # miss or expired -> compute
            val = func(*args, **kwargs)

            with lock:
                # evict if needed
                try:
                    # insert as most-recently-used
                    cache[key] = (now, val)
                    # enforce maxsize
                    if maxsize is not None:
                        while len(cache) > maxsize:
                            try:
                                evicted_key, _ = cache.popitem(last=False)
                                logger.debug("Cache EVICT key=%s for %s", evicted_key, func.__name__)
                            except Exception:
                                break
                except Exception:
                    # if insertion fails (unhashable key), ignore caching
                    pass
            return val

        def cache_clear():
            with lock:
                cache.clear()

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        return wrapper
    return decorator


# -------------------------
# Metrics: unified _inc(name) wrapper (Prometheus optional)
# -------------------------


try:
    from prometheus_client import Counter  # type: ignore
    S3_HITS = Counter("image_utils_s3_hits", "S3 hits for image lookups")
    S3_MISSES = Counter("image_utils_s3_misses", "S3 misses for image lookups")
    LOCAL_HITS = Counter("image_utils_local_hits", "Local filesystem hits for image lookups")
    PLACEHOLDERS = Counter("image_utils_placeholders", "Placeholder served for image lookups")

    def _inc(name: str) -> None:
        try:
            if name == "s3_hits":
                S3_HITS.inc()
            elif name == "s3_misses":
                S3_MISSES.inc()
            elif name == "local_hits":
                LOCAL_HITS.inc()
            elif name == "placeholders":
                PLACEHOLDERS.inc()
        except Exception:
            # fallback noop if prometheus client misbehaves
            pass

except Exception:
    _COUNTERS: Dict[str, int] = {"s3_hits": 0, "s3_misses": 0, "local_hits": 0, "placeholders": 0}

    def _inc(name: str) -> None:
        try:
            _COUNTERS[name] = _COUNTERS.get(name, 0) + 1
        except Exception:
            pass


# -------------------------
# Lookup helper (S3-first with fallback) with TTL cache
# -------------------------


def _normalize_s3_prefix(prefix: str) -> str:
    """Return normalized s3://bucket/path without trailing slash (if looks like s3)."""
    if not prefix:
        return ""
    p = prefix.strip()
    if p.lower().startswith("s3://"):
        return p.rstrip("/")
    return p.rstrip("/")


@ttl_cache(IMAGE_CACHE_TTL_SECONDS)
def _find_first_existing_uncached(base_dir: Path, candidates: Tuple[str, ...]) -> ImageCandidate:
    """
    Uncached implementation: checks S3 (with retry) then local filesystem.
    Note: cached wrapper uses TTL to avoid stale forever caches.
    """
    # convert tuple to list for iteration
    candidates_list = list(candidates)
    s3_prefix = _normalize_s3_prefix(S3_PROCESSED_IMAGES_PATH)

    # S3-first
    if s3_prefix and (fsspec is not None or boto3 is not None):
        logger.debug("Checking S3 prefix %s for %d candidates", s3_prefix, len(candidates_list))
        try:
            fs = None
            if fsspec is not None:
                try:
                    fs = fsspec.filesystem("s3")
                except Exception as e:
                    logger.warning("Failed to initialize fsspec S3 filesystem: %s", e)
                    fs = None

            subdir = base_dir.name if isinstance(base_dir, Path) else str(base_dir).rstrip("/").split("/")[-1]
            for fn in candidates_list:
                # Build two candidate keys: prefix/subdir/filename and prefix/filename
                candidate_keys = [
                    f"{s3_prefix}/{subdir}/{fn}",
                    f"{s3_prefix}/{fn}"
                ]
                for raw_key in candidate_keys:
                    s3_uri = _ensure_s3_uri(raw_key)
                    try:
                        if _s3_exists_with_retry(fs, s3_uri):
                            # build public URL using CDN_BASE_URL if present; encode key path carefully
                            key_only = _s3_object_key_from_s3_uri(s3_uri)
                            if CDN_BASE_URL:
                                encoded = _encode_key_for_cdn(key_only)
                                public_url = f"{CDN_BASE_URL.rstrip('/')}/{encoded.lstrip('/')}"
                            else:
                                public_url = s3_uri
                            logger.info("Found candidate on S3: %s (filename=%s)", s3_uri, fn)
                            _inc("s3_hits")
                            return ImageCandidate(filename=fn, s3_uri=s3_uri, local_path=None, public_url=public_url)
                        else:
                            logger.debug("S3 key not found: %s", s3_uri)
                    except Exception as e:
                        logger.warning("Error checking S3 key %s: %s", s3_uri, e)
            logger.debug("No S3 candidate found for %s under %s", candidates_list, s3_prefix)
            _inc("s3_misses")
        except Exception as e:
            logger.error("S3 listing/lookup failed entirely for prefix %s: %s. Falling back to local.", s3_prefix, e)
            _inc("s3_misses")

    # Local fallback
    logger.debug("Falling back to local directory %s", base_dir)
    for fn in candidates_list:
        p = base_dir / fn
        try:
            if p.exists():
                public_url = f"{URL_BASE}/{base_dir.name}/{fn}"
                logger.info("Found local file: %s", p)
                _inc("local_hits")
                return ImageCandidate(filename=fn, s3_uri=None, local_path=str(p), public_url=public_url)
        except Exception as e:
            logger.warning("Local filesystem check failed for %s: %s", p, e)

    # If nothing found, return placeholder candidate
    logger.warning("No match found for candidates %s under %s; returning placeholder", candidates_list, base_dir)
    placeholder = PLACEHOLDER_LOGO if "logo" in str(base_dir).lower() else PLACEHOLDER_BANNER
    local_placeholder_path = base_dir / placeholder
    placeholder_url = f"{URL_BASE}/{base_dir.name}/{placeholder}"
    _inc("placeholders")
    return ImageCandidate(filename=placeholder, s3_uri=None, local_path=str(local_placeholder_path), public_url=placeholder_url)


def find_first_existing(base_dir: Path, candidates: List[str]) -> ImageCandidate:
    """Public: return ImageCandidate. Cached with TTL controlled by IMAGE_CACHE_TTL_SECONDS."""
    # Use path.as_posix() for stable cache key semantics
    key_base = Path(base_dir).as_posix()
    key_candidates = tuple(candidates)
    return _find_first_existing_uncached(Path(key_base), key_candidates)


# -------------------------
# URL builder (compat)
# -------------------------


def _url_for_candidate(candidate: Optional[ImageCandidate], url_subpath: str) -> str:
    """Build public URL for candidate.

    Prioritizes candidate.public_url (constructed during lookup), then CDN if s3_uri,
    otherwise local static path.
    """
    if candidate is None:
        return f"{URL_BASE}/{url_subpath}/{PLACEHOLDER_LOGO}"

    if candidate.public_url:
        return candidate.public_url

    if candidate.s3_uri and CDN_BASE_URL:
        key = _s3_object_key_from_s3_uri(candidate.s3_uri)
        encoded = _encode_key_for_cdn(key)
        return f"{CDN_BASE_URL.rstrip('/')}/{encoded.lstrip('/')}"

    filename = candidate.filename or PLACEHOLDER_LOGO
    return f"{URL_BASE}/{url_subpath}/{filename}"


# -------------------------
# Public API (sync + async wrappers)
# -------------------------


def get_logo_path(symbol: str, company_name: str) -> Tuple[ImageCandidate, str]:
    """Return (ImageCandidate, url) for logo. Prefer S3 then local."""
    candidates = candidate_logo_filenames(symbol, company_name)
    res = find_first_existing(PROCESSED_LOGOS, candidates)
    url = _url_for_candidate(res, "processed_logos")
    return res, url


async def async_get_logo_path(symbol: str, company_name: str) -> Tuple[ImageCandidate, str]:
    """Async wrapper safe for use in FastAPI endpoints."""
    pool = _ensure_thread_pool()
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(pool, lambda: get_logo_path(symbol, company_name))
    return res


def get_banner_path(symbol: str, company_name: str) -> Tuple[ImageCandidate, str]:
    candidates = candidate_banner_filenames(symbol, company_name)
    res = find_first_existing(PROCESSED_BANNERS, candidates)
    url = _url_for_candidate(res, "processed_banners")
    return res, url


async def async_get_banner_path(symbol: str, company_name: str) -> Tuple[ImageCandidate, str]:
    pool = _ensure_thread_pool()
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(pool, lambda: get_banner_path(symbol, company_name))
    return res


# -------------------------
# CLI quick test
# -------------------------
if __name__ == "__main__":
    sym = "RELIANCE"
    name = "Reliance Industries Limited"

    lp, lu = get_logo_path(sym, name)
    bp, bu = get_banner_path(sym, name)

    print("Logo candidate:", lp)
    print("Logo public_url:", lu)
    print("Banner candidate:", bp)
    print("Banner public_url:", bu)