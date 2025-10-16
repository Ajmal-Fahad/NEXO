# backend/routers/announcements.py
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Any
from pathlib import Path
import json
import re
from datetime import datetime
from schemas.schema import AnnouncementListItem, AnnouncementDetail, MarketSnapshot

import logging
from pathlib import Path as _Path
LOGS_DIR = _Path(__file__).resolve().parent.parent / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
if not logger.handlers:
    from logging.handlers import RotatingFileHandler
    fh = RotatingFileHandler(LOGS_DIR / 'announcements_warnings.log', maxBytes=10_000_000, backupCount=5)
    fh.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
logger.setLevel(logging.WARNING)

router = APIRouter(prefix="/announcements")

# Cache for list announcements (TTL 30s, check dir mtime)
_LIST_CACHE = {"ts": 0.0, "data": [], "dir_mtime": 0.0}
_CACHE_TTL = 30.0  # seconds

# ────────────────────────────────────────────────────────────────
# Paths
ANN_DIR = Path(__file__).resolve().parent.parent / "data" / "announcements"

# ────────────────────────────────────────────────────────────────
# Helpers
def _safe_load_json(p: Path) -> Optional[dict]:
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.warning("Skipping bad JSON file %s: %s", p, e)
        return None

def _normalize_logo_path(logo: Optional[str]) -> Optional[str]:
    if not isinstance(logo, str):
        return None
    logo = logo.strip()
    if not logo:
        return None
    # Accept and pass through http(s) or /static URLs unchanged
    if logo.startswith(("http://", "https://")) or logo.startswith("/static"):
        return logo
    # convert absolute input_data path -> /static/... so frontend can fetch
    if logo.startswith("/Users") and "/input_data/" in logo:
        idx = logo.find("/input_data/")
        rel = logo[idx + len("/input_data") :]
        return "/static" + rel
    # If absolute FS path but not under input_data, return None to avoid exposure
    if logo.startswith("/"):
        return None
    # already relative/static or other
    return logo

def _parse_dt(x: Any) -> float:
    if not x:
        return 0.0
    try:
        # try ISO first
        return datetime.fromisoformat(str(x)).timestamp()
    except Exception:
        pass
    for fmt in ("%d-%b-%Y, %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%b-%Y"):
        try:
            return datetime.strptime(str(x), fmt).timestamp()
        except Exception:
            continue
    # last resort: try to extract digits and parse
    try:
        # handle "04-Oct-2025, 23:56:23"
        s = str(x)
        # replace comma with space
        s = s.replace(",", "")
        return datetime.strptime(s, "%d %b %Y %H:%M:%S").timestamp()
    except Exception:
        return 0.0

# ────────────────────────────────────────────────────────────────
# Load announcements fresh from disk (no global caching)
def _load_all_announcements() -> list[dict]:
    import time
    now = time.time()
    try:
        dir_mtime = ANN_DIR.stat().st_mtime
    except OSError:
        dir_mtime = 0.0
    if _LIST_CACHE["data"] and (now - _LIST_CACHE["ts"]) < _CACHE_TTL and _LIST_CACHE["dir_mtime"] == dir_mtime:
        return _LIST_CACHE["data"]
    
    results = []
    if not ANN_DIR.exists():
        logger.warning(f"Announcement directory missing: {ANN_DIR}")
        return results

    # iterate files in deterministic order (older -> newer), but we will sort later by announcement datetime
    for p in sorted(ANN_DIR.rglob("*.json")):
        j = _safe_load_json(p)
        if not j:
            continue
        # ensure id exists
        if "id" not in j or not j.get("id"):
            j["id"] = p.stem
        results.append(j)
    
    _LIST_CACHE["data"] = results
    _LIST_CACHE["ts"] = now
    _LIST_CACHE["dir_mtime"] = dir_mtime
    return results

# ────────────────────────────────────────────────────────────────
# Build lightweight summary object for CardsList
def _build_list_item(j: dict) -> dict:
    # Prefer canonical fields set by PDF processor
    company_name = j.get("canonical_company_name") or j.get("company_name") or "Unknown Company"
    logo = _normalize_logo_path(j.get("company_logo") or "")
    # fallback to market_snapshot logos if present
    if not logo and isinstance(j.get("market_snapshot"), dict):
        ms = j["market_snapshot"]
        for k in ("logo_url", "banner_url"):
            v = ms.get(k)
            if isinstance(v, list) and v:
                logo = _normalize_logo_path(v[0])
                break
            elif isinstance(v, str) and v:
                logo = _normalize_logo_path(v)
                break

    # choose headline in order of preference
    headline = j.get("headline_final") or j.get("headline_ai") or j.get("headline_raw") or j.get("headline") or ""
    dt_val = j.get("announcement_datetime_human") or j.get("announcement_datetime_iso") or j.get("announcement_datetime") or ""

    sentiment = None
    emoji = None
    if isinstance(j.get("sentiment_badge"), dict):
        sentiment = j["sentiment_badge"].get("label")
        emoji = j["sentiment_badge"].get("emoji")

    return {
        "id": j.get("id") or j.get("filename") or j.get("source_file"),
        "company_name": company_name,
        "company_logo": logo,
        "headline": (headline or "")[:240],
        "announcement_datetime": dt_val,
        "sentiment": sentiment,
        "sentiment_emoji": emoji,
        "symbol": j.get("symbol") or j.get("canonical_symbol"),
    }

# ────────────────────────────────────────────────────────────────
@router.get("", response_model=List[AnnouncementListItem], response_model_exclude_none=True)
def list_announcements(limit: int = 50, offset: int = 0, q: Optional[str] = None):
    """
    Returns latest announcements; reads fresh from disk each request (no in-memory cache)
    - limit: number of items to return (default 50, max 200)
    - offset: pagination offset
    - q: optional search string (filters company_name, headline, symbol)
    """
    import time
    start_time = time.time()
    raw = _load_all_announcements()
    lite = [_build_list_item(r) for r in raw]

    if q:
        qlow = q.strip().lower()
        lite = [
            it
            for it in lite
            if qlow in (it.get("company_name") or "").lower()
            or qlow in (it.get("headline") or "").lower()
            or qlow in str(it.get("symbol") or "").lower()
        ]

    # dedupe by id, latest datetime wins
    seen: dict[str, dict] = {}
    for it in lite:
        key = str(it.get("id") or "")
        if not key:
            continue
        if key not in seen or _parse_dt(it.get("announcement_datetime")) > _parse_dt(seen[key].get("announcement_datetime")):
            seen[key] = it

    out = list(seen.values())
    # sort by announcement datetime descending
    out.sort(key=lambda x: _parse_dt(x.get("announcement_datetime")), reverse=True)

    start = max(int(offset or 0), 0)
    end = start + max(min(int(limit or 50), 200), 1)
    result = out[start:end]
    
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info("list_announcements returned %d items (limit=%d offset=%d q='%s') in %.2fms", len(result), limit, offset, q or "", elapsed_ms)
    return result

# ────────────────────────────────────────────────────────────────
@router.get("/{announcement_id}", response_model=AnnouncementDetail, response_model_exclude_none=True)
def get_announcement(announcement_id: str):
    import time
    start_time = time.time()
    # Read fresh from disk
    raw = _load_all_announcements()
    for j in raw:
        if str(j.get("id")) == str(announcement_id) or str(j.get("filename", "")).endswith(f"{announcement_id}.json"):
            # prefer canonical/AI headline for detail view if explicit 'headline' missing
            if not j.get("headline"):
                j["headline"] = j.get("headline_final") or j.get("headline_ai") or j.get("headline_raw") or ""

            # If top-level sentiment not set, derive from sentiment_badge for detail view
            sb = j.get("sentiment_badge")
            if not j.get("sentiment") and isinstance(sb, dict):
                j["sentiment"] = sb.get("label") or ""
                if "sentiment_emoji" not in j or j.get("sentiment_emoji") is None:
                    j["sentiment_emoji"] = sb.get("emoji") or ""

            # Defensive: sanitize None -> ""
            for key in ("announcement_datetime", "company_name", "headline", "sentiment", "company_logo"):
                if j.get(key) is None:
                    j[key] = ""
            if not j.get("announcement_datetime"):
                j["announcement_datetime"] = (
                    j.get("announcement_datetime_human") 
                    or j.get("announcement_datetime_iso")
                    or ""
                )
            # Ensure market_snapshot validity
            if isinstance(j.get("market_snapshot"), dict):
                try:
                    j["market_snapshot"] = MarketSnapshot(**j["market_snapshot"])
                except Exception:
                    j["market_snapshot"] = None
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info("get_announcement found id='%s' in %.2fms", announcement_id, elapsed_ms)
            return AnnouncementDetail(**j)

    # direct file fallback
    candidate = ANN_DIR / f"{announcement_id}.json"
    if candidate.exists():
        j = _safe_load_json(candidate)
        if not j:
            logger.warning("Announcement file corrupt for id='%s'", announcement_id)
            raise HTTPException(status_code=500, detail="Announcement file corrupt")
        # prefer canonical/AI headline for detail view if explicit 'headline' missing
        if not j.get("headline"):
            j["headline"] = j.get("headline_final") or j.get("headline_ai") or j.get("headline_raw") or ""
        # If top-level sentiment not set, derive from sentiment_badge for detail view
        sb = j.get("sentiment_badge")
        if not j.get("sentiment") and isinstance(sb, dict):
            j["sentiment"] = sb.get("label") or ""
            if "sentiment_emoji" not in j or j.get("sentiment_emoji") is None:
                j["sentiment_emoji"] = sb.get("emoji") or ""
        for key in ("announcement_datetime", "company_name", "headline", "sentiment", "company_logo"):
            if j.get(key) is None:
                j[key] = ""
        if not j.get("announcement_datetime"):
            j["announcement_datetime"] = (
                j.get("announcement_datetime_human")
                or j.get("announcement_datetime_iso")
                or ""
            )
        if isinstance(j.get("market_snapshot"), dict):
            try:
                j["market_snapshot"] = MarketSnapshot(**j["market_snapshot"])
            except Exception:
                j["market_snapshot"] = None
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info("get_announcement found id='%s' via file fallback in %.2fms", announcement_id, elapsed_ms)
        return AnnouncementDetail(**j)

    logger.warning("Announcement not found for id='%s'", announcement_id)
    raise HTTPException(status_code=404, detail="Announcement not found")
