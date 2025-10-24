#!/usr/bin/env python3
"""
services/pdf_processor.py

Refactored, robust PDF -> announcement master JSON processor (v2.0).

Responsibilities:
 - Use filename_utils.filename_to_symbol() to resolve canonical symbol/company_name.
 - Extract announcement datetime from filename (via filename_utils).
 - Extract text from PDF using pypdf (if available).
 - Enrich using csv_utils.get_market_snapshot and csv_utils.get_indices_for_symbol.
 - Get images via image_utils.get_logo_path/get_banner_path.
 - Call llm_utils for headline and summary and sentiment_utils for blended sentiment.
 - Write master JSON atomically and move processed PDF into date-based folder.
 - Maintain processing_events for debugging/audit.

Behavior:
 - S3-first then local fallback, consistent with image_utils/csv_utils.
 - Warm caches for csv_utils/index_builder at import time (non-forced) to avoid duplicate loads.
 - process_pdf(pdf_path: Path) operates on local Path objects and performs text extraction,
   enrichment, JSON write and moving to processed folder (local).
 - process_single_pdf(s3_or_local: str, ...) is S3-aware wrapper:
     * Accepts either local path or s3://bucket/key.pdf
     * If S3: downloads to a temp local file, calls process_pdf, then uploads:
         - processed PDF -> s3://<bucket>/input_data/pdf/processed/<YYYY-MM-DD>/<filename>
         - JSON -> s3://<bucket>/data/announcements/<YYYY-MM-DD>/<file_id>.json
       (respects `overwrite` flag for uploads)
     * Returns master dict and upload metadata.
"""

from __future__ import annotations
import os
import sys
import json
import hashlib
import shutil
import argparse
import traceback
import tempfile
import uuid
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple

# Attempt to import local services (fall back to top-level imports if run differently)
try:
    # Prefer intra-package relative imports when module is executed as a package
    from . import csv_utils, filename_utils, image_utils, llm_utils, sentiment_utils, index_builder  # type: ignore
except Exception:
    try:
        from backend.services import csv_utils, filename_utils, image_utils, llm_utils, sentiment_utils, index_builder  # type: ignore
    except Exception as e:
        print("Failed to import service modules. Ensure you run from project root with proper PYTHONPATH.", file=sys.stderr)
        raise

# Optional PDF reader
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# Optional fsspec for S3 file-like operations (preferred)
try:
    import fsspec  # type: ignore
except Exception:
    fsspec = None  # pragma: no cover - optional

# Optional boto3 fallback for upload/download
try:
    import boto3  # type: ignore
except Exception:
    boto3 = None  # pragma: no cover - optional

# small helpers & timezone
IST = timezone(timedelta(hours=5, minutes=30))
VERSION = "pdf_processor_v2.0"

# Local directories (relative to repository)
BASE = Path(__file__).resolve().parents[1]
INCOMING_PDF_DIR = BASE / "input_data" / "pdf"
PROCESSED_JSON_BASE = BASE / "data" / "announcements"
PROCESSED_PDF_BASE = BASE / "input_data" / "pdf" / "processed"
ERROR_DIR = BASE / "error_reports"

# Ensure local directories exist (safe no-op for S3 case)
for d in (PROCESSED_JSON_BASE, PROCESSED_PDF_BASE, ERROR_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# ---------------------------
# Warm caches at import time
# ---------------------------
# Important: call non-forced loading so we don't reload CSV every time.
try:
    # load processed df if csv_utils provides it (non-forced)
    if getattr(csv_utils, "load_processed_df", None):
        try:
            csv_utils.load_processed_df(force_reload=False)
        except TypeError:
            # some implementations may not accept force_reload param
            try:
                csv_utils.load_processed_df()
            except Exception:
                pass
        except Exception:
            pass
    # warm index_builder if available
    if getattr(index_builder, "refresh_index", None):
        try:
            index_builder.refresh_index()
        except TypeError:
            try:
                index_builder.build_index()
            except Exception:
                pass
        except Exception:
            pass
except Exception:
    # never fail import because of warm-up
    pass

# ---------------------------
# Helper functions
# ---------------------------


def now_iso() -> str:
    """Return current IST time as ISO string with tzinfo."""
    return datetime.now(IST).isoformat()


def compute_sha1(path: Path) -> str:
    """Compute SHA1 of a local Path file."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def make_json_safe(obj):
    """
    Convert non-JSON-friendly objects into JSON safe types.
    Preserves behavior from previous implementation.
    """
    try:
        import numpy as _np  # type: ignore
        import pandas as _pd  # type: ignore
    except Exception:
        _np = None
        _pd = None

    from pathlib import Path as _Path

    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if _np is not None:
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
    if _pd is not None:
        try:
            if isinstance(obj, _pd.Timestamp):
                return obj.isoformat()
            if _pd.isna(obj):
                return None
        except Exception:
            pass
    if isinstance(obj, _Path):
        return str(obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                out[str(k)] = make_json_safe(v)
            except Exception:
                out[str(k)] = str(v)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return str(obj)


def safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


# ---------------------------
# PDF text extraction
# ---------------------------


def extract_text_from_pdf(pdf_path: Path, max_pages: Optional[int] = 200) -> str:
    """
    Extract textual content from a PDF file using pypdf.
    Returns empty string if pypdf not installed or extraction fails.
    """
    if PdfReader is None:
        return ""
    try:
        parts: List[str] = []
        with pdf_path.open("rb") as fh:
            reader = PdfReader(fh)
            for i, p in enumerate(reader.pages):
                if max_pages and i >= max_pages:
                    break
                try:
                    txt = p.extract_text() or ""
                except Exception:
                    txt = ""
                if txt:
                    parts.append(txt)
        return "\n".join(parts)
    except Exception:
        return ""


# ---------------------------
# Announcement datetime extraction
# ---------------------------


def _extract_datetime_from_filename_local(filename: str) -> Optional[Dict[str, str]]:
    """
    Local fallback extractor for datetime from filename. Returns {"iso":..., "human":...}
    or None if no parseable date found.
    Accepts patterns: DD-MM-YYYY HH_mm_ss, YYYY-MM-DD, YYYYMMDD, DD-MM-YYYY
    """
    if not filename:
        return None
    name = Path(filename).name
    # DD-MM-YYYY HH_mm_ss or with colons
    m = re.search(r'(?P<d>\d{2}-\d{2}-\d{4})[ _-]+(?P<h>\d{2})[:_](?P<m>\d{2})[:_](?P<s>\d{2})', name)
    if m:
        try:
            dt = datetime.strptime(f"{m.group('d')} {m.group('h')}:{m.group('m')}:{m.group('s')}", "%d-%m-%Y %H:%M:%S")
            dt = dt.replace(tzinfo=IST)
            return {"iso": dt.isoformat(), "human": dt.strftime("%d-%b-%Y, %H:%M:%S")}
        except Exception:
            pass
    # YYYY-MM-DD or YYYYMMDD
    m2 = re.search(r'(?P<Y>\d{4})[-_]?((?P<M>\d{2})[-_]?((?P<D>\d{2})))', name)
    if m2:
        try:
            Y = int(m2.group("Y")); M = int(m2.group("M")); D = int(m2.group("D"))
            dt = datetime(Y, M, D, 0, 0, 0, tzinfo=IST)
            return {"iso": dt.isoformat(), "human": dt.strftime("%d-%b-%Y, %H:%M:%S")}
        except Exception:
            pass
    # DD-MM-YYYY only
    m3 = re.search(r'(?P<d2>\d{2}-\d{2}-\d{4})', name)
    if m3:
        try:
            dt = datetime.strptime(m3.group('d2'), "%d-%m-%Y").replace(tzinfo=IST)
            return {"iso": dt.isoformat(), "human": dt.strftime("%d-%b-%Y, %H:%M:%S")}
        except Exception:
            pass
    return None


def extract_datetime_from_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Wrapper that uses filename_utils.extract_datetime_from_filename if available,
    otherwise falls back to local parser above.
    """
    try:
        if hasattr(filename_utils, "extract_datetime_from_filename"):
            dt = filename_utils.extract_datetime_from_filename(filename)
            if isinstance(dt, tuple) and len(dt) == 2:
                iso, human = dt
                if iso or human:
                    return {"iso": iso, "human": human}
            if isinstance(dt, dict):
                return dt
        return _extract_datetime_from_filename_local(filename)
    except Exception:
        return _extract_datetime_from_filename_local(filename)


# ---------------------------
# Market snapshot & images enrichment
# ---------------------------


def enrich_with_market_and_indices(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Return a normalized market snapshot (JSON-ready) or None.
    Uses csv_utils.get_market_snapshot + get_indices_for_symbol and image_utils for images.
    """
    if not symbol:
        return None
    snap = csv_utils.get_market_snapshot(symbol) or {}
    keys = [
        "symbol", "company_name", "rank", "price", "change_1d_pct", "change_1w_pct", "vwap", "mcap_rs_cr",
        "volume_24h_rs_cr", "all_time_high", "atr_pct", "relative_vol", "vol_change_pct", "volatility",
        "market_snapshot_date", "logo_url", "banner_url"
    ]
    normalized = {k: snap.get(k) if snap.get(k) is not None else None for k in keys}

    # numeric conversions (best-effort)
    for k in ("price", "vwap", "mcap_rs_cr", "volume_24h_rs_cr", "all_time_high", "atr_pct", "relative_vol", "vol_change_pct", "volatility", "rank"):
        v = normalized.get(k)
        if v is not None:
            try:
                if k == "rank":
                    normalized[k] = int(v)
                else:
                    normalized[k] = float(v)
            except Exception:
                normalized[k] = v

    # attach indices
    try:
        broad, sector = csv_utils.get_indices_for_symbol(symbol)
    except Exception:
        broad, sector = ("Uncategorised Index", "Uncategorised Sector")
    normalized["broad_index"] = broad
    normalized["sector_index"] = sector

    # attach images (best-effort)
    try:
        lp, lu = image_utils.get_logo_path(symbol, normalized.get("company_name") or "")
        bp, bu = image_utils.get_banner_path(symbol, normalized.get("company_name") or "")
        normalized["logo_url"] = [str(lp), lu] if lp is not None else None
        normalized["banner_url"] = [str(bp), bu] if bp is not None else None
    except Exception:
        pass

    return make_json_safe(normalized)


# ---------------------------
# Filename matching (delegates to filename_utils when available)
# ---------------------------


def match_filename(filename: str) -> Dict[str, Any]:
    """
    Use filename_utils.filename_to_symbol() to determine canonical symbol/company.
    Returns an event-like dict with keys: found, symbol, company_name, score, match_type, candidates.
    Falls back to index_builder direct APIs if filename_utils not available.
    """
    ev = {"found": False, "symbol": None, "company_name": None, "score": 0.0, "match_type": "no_match", "candidates": []}
    try:
        if hasattr(filename_utils, "filename_to_symbol"):
            csv_path = None
            try:
                csv_path = str(csv_utils._EOD_PATH) if getattr(csv_utils, "_EOD_PATH", None) else None
            except Exception:
                csv_path = None
            res = filename_utils.filename_to_symbol(filename, csv_path=csv_path)
            if isinstance(res, dict) and res.get("found"):
                ev.update({
                    "found": True,
                    "symbol": res.get("symbol"),
                    "company_name": res.get("company_name"),
                    "score": res.get("score", 0.0),
                    "match_type": res.get("match_type"),
                    "candidates": res.get("candidates", []) or []
                })
                return ev
    except Exception:
        # We'll fall through to index_builder fallbacks.
        pass

    # index_builder direct fallbacks (best-effort)
    try:
        base_no_ext = re.sub(r'[\-_.]+', ' ', Path(filename).name.rsplit(".", 1)[0]).strip()
        tokens = [t for t in re.split(r'\s+', base_no_ext) if t and not t.isdigit() and len(t) >= 2]

        # exact token match
        for t in tokens:
            sym = index_builder.get_symbol_by_exact(t)
            if sym:
                row = index_builder.get_company_by_symbol(sym)
                ev.update({"found": True, "symbol": sym, "company_name": row.get("company_name") if row else None, "score": 1.0, "match_type": "exact"})
                return ev

        # token -> single symbol
        for t in tokens:
            sym = index_builder.get_symbol_by_token(t)
            if sym:
                row = index_builder.get_company_by_symbol(sym)
                ev.update({"found": True, "symbol": sym, "company_name": row.get("company_name") if row else None, "score": 0.85, "match_type": "token"})
                return ev

        # token overlap
        overlap = index_builder.token_overlap_search(tokens, min_score=0.55)
        if overlap:
            sym, score = overlap
            row = index_builder.get_company_by_symbol(sym)
            ev.update({"found": True, "symbol": sym, "company_name": row.get("company_name") if row else None, "score": float(score), "match_type": "token_overlap"})
            return ev

        # fuzzy
        fuzzy = index_builder.fuzzy_lookup_symbol(base_no_ext)
        if fuzzy:
            row = index_builder.get_company_by_symbol(fuzzy)
            ev.update({"found": True, "symbol": fuzzy, "company_name": row.get("company_name") if row else None, "score": 0.7, "match_type": "fuzzy"})
            return ev
    except Exception:
        pass

    # final fallback: call filename_utils.filename_to_symbol without csv_path if present
    try:
        if hasattr(filename_utils, "filename_to_symbol"):
            res = filename_utils.filename_to_symbol(filename)
            if isinstance(res, dict) and res.get("found"):
                ev.update({"found": True, "symbol": res.get("symbol"), "company_name": res.get("company_name"), "score": res.get("score", 0.0), "match_type": res.get("match_type"), "candidates": res.get("candidates", [])})
                return ev
    except Exception:
        pass

    return ev


# ---------------------------
# JSON atomic writer & helpers
# ---------------------------


def write_master_json(out_path: Path, data: Dict[str, Any]):
    """
    Write JSON atomically using tmp file and os.replace to ensure atomicity.
    """
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(make_json_safe(data), fh, ensure_ascii=False, indent=2)
    os.replace(str(tmp), str(out_path))


def _validate_headline(h: Optional[str]) -> bool:
    """Return True when headline appears plausible."""
    if not h or not isinstance(h, str):
        return False
    h = h.strip()
    parts = h.split()
    if len(parts) < 3 or len(parts) > 40:
        return False
    if not h.endswith((".", "!", "?")):
        return False
    verbs = {"appoint", "approve", "announce", "sign", "acquire", "launch", "enter", "issue", "complete", "report", "declare", "receive", "award", "submit", "file", "receives", "confirms"}
    return any(v in h.lower() for v in verbs)


def deterministic_headline_from_summary(summary: Optional[str]) -> Optional[str]:
    """Fallback headline derived deterministically from summary text (first ~14 words)."""
    if not summary:
        return None
    s = summary.strip().replace("\n", " ")
    parts = s.split()
    head = " ".join(parts[:14])
    if head and head[-1] not in ".!?":
        head = head.strip() + "."
    return head


# ---------------------------
# Core local PDF processing
# ---------------------------


def process_pdf(pdf_path: Path, dry_run: bool = False, force: bool = False) -> Dict[str, Any]:
    """
    Process a single local PDF Path into the master JSON dict and move the PDF (unless dry_run).
    Returns the master dict.

    Note: This function expects a local Path. For S3 inputs use process_single_pdf which
    downloads the object and calls process_pdf() internally.
    """
    evts: List[Dict[str, Any]] = []
    evts.append({"event": "start", "ts": now_iso(), "file": str(pdf_path)})

    try:
        # 0) compute sha1
        sha1 = compute_sha1(pdf_path)
        evts.append({"event": "sha1_computed", "ts": now_iso(), "sha1": sha1})

        # 1) filename -> symbol
        match = match_filename(pdf_path.name)
        evts.append({"event": "filename_matched", "ts": now_iso(), "match": match})
        canonical_symbol = match.get("symbol")
        canonical_company_name = match.get("company_name")

        # 2) announcement datetime (filename_utils preferred)
        dt_info = None
        try:
            if hasattr(filename_utils, "extract_datetime_from_filename"):
                dt_res = filename_utils.extract_datetime_from_filename(pdf_path.name)
                if isinstance(dt_res, tuple) and len(dt_res) == 2:
                    iso, human = dt_res
                    dt_info = {"iso": iso, "human": human} if (iso or human) else None
                elif isinstance(dt_res, dict):
                    dt_info = dt_res
        except Exception:
            dt_info = None
        if not dt_info:
            dt_info = extract_datetime_from_filename(pdf_path.name) or {}

        announcement_iso = dt_info.get("iso")
        announcement_human = dt_info.get("human")

        # 3) extract text (pypdf) - may be empty if binary scan or pypdf unavailable
        body_text = extract_text_from_pdf(pdf_path) if PdfReader else ""
        evts.append({"event": "text_extracted", "ts": now_iso(), "chars": len(body_text)})

        # 4) enrichment
        market_snapshot = enrich_with_market_and_indices(canonical_symbol) if canonical_symbol else None
        evts.append({"event": "enrichment_done", "ts": now_iso(), "market_snapshot_found": bool(market_snapshot)})

        # 5) images
        company_logo = None
        banner_image = None
        if canonical_symbol:
            try:
                logo_path, logo_pub = image_utils.get_logo_path(canonical_symbol, canonical_company_name or "")
                banner_path, banner_pub = image_utils.get_banner_path(canonical_symbol, canonical_company_name or "")
                company_logo = str(logo_path) if logo_path is not None else None
                banner_image = str(banner_path) if banner_path is not None else None
            except Exception:
                company_logo = None
                banner_image = None

        # 6) LLM: headline + summary
        llm_meta = {"used": False}
        headline_ai = None
        headline_final = None
        summary_60 = None
        headline_raw = None

        try:
            hs = llm_utils.classify_headline_and_summary(body_text)
            llm_meta = hs.get("llm_meta", {"ok": False})
            headline_ai = hs.get("headline_ai") or hs.get("headline_final") or None
            summary_60 = hs.get("summary_60") or None
            headline_raw = hs.get("headline_raw") or None
            evts.append({"event": "llm_done", "ts": now_iso(), "meta": {"ok": llm_meta.get("ok"), "model": llm_meta.get("model")}})
        except Exception as e:
            evts.append({"event": "llm_error", "ts": now_iso(), "error": str(e)})

        # choose final headline
        if headline_ai and isinstance(headline_ai, str):
            headline_final = headline_ai.strip()
            if re.search(r'\b(of|for|including|regarding|with|to|on|in)\.$', headline_final.strip().lower()):
                parts = headline_final.rstrip('.').split()
                if len(parts) > 3:
                    headline_final = " ".join(parts[:-1]).rstrip() + "."
        else:
            # retry once if nothing produced
            if not summary_60 and headline_ai is None:
                try:
                    hs2 = llm_utils.classify_headline_and_summary(body_text)
                    headline_ai = headline_ai or hs2.get("headline_ai") or hs2.get("headline_final")
                    summary_60 = summary_60 or hs2.get("summary_60")
                    evts.append({"event": "llm_retry", "ts": now_iso()})
                except Exception:
                    pass
            if headline_ai:
                headline_final = str(headline_ai).strip()
            elif summary_60:
                headline_final = deterministic_headline_from_summary(summary_60)
            else:
                headline_final = None

        evts.append({"event": "headline_finalized", "ts": now_iso(), "path": "llm_valid" if headline_final else "fallback", "headline_final": headline_final})

        # 7) sentiment
        llm_sent_raw = None
        try:
            sent_raw = llm_utils.classify_sentiment(body_text)
            llm_sent_raw = {"label": sent_raw.get("label"), "score": safe_float(sent_raw.get("score"), None)}
        except Exception:
            llm_sent_raw = None

        sent = sentiment_utils.compute_sentiment(body_text or "", llm_raw=llm_sent_raw)
        evts.append({"event": "sentiment_computed", "ts": now_iso(), "badge": {"label": sent.get("label"), "score": sent.get("score")}})

        # 8) build master dict (canonical)
        created_at = now_iso()
        iso_for_id = announcement_iso if announcement_iso else datetime.now(IST).isoformat()
        safe_iso = iso_for_id.replace(":", "").replace("-", "").split("+")[0]
        file_id = f"ann_{safe_iso}_{sha1[:16]}"

        master = {
            "id": file_id,
            "sha1": sha1,
            "canonical_symbol": canonical_symbol,
            "canonical_company_name": canonical_company_name,
            "symbol": canonical_symbol or (market_snapshot.get("symbol") if market_snapshot else None),
            "company_name": canonical_company_name or (market_snapshot.get("company_name") if market_snapshot else None),
            "headline_final": headline_final,
            "headline_raw": headline_raw,
            "headline_ai": headline_ai,
            "summary_60": summary_60,
            "summary_raw": (body_text[:4000] + "...") if body_text and len(body_text) > 4000 else body_text or None,
            "announcement_datetime_iso": announcement_iso,
            "announcement_datetime_human": announcement_human,
            "source_file_original": pdf_path.name,
            "source_file_normalized": pdf_path.name,
            "market_snapshot": market_snapshot,
            "indices": [market_snapshot.get("broad_index") if market_snapshot else "Uncategorised Index",
                        market_snapshot.get("sector_index") if market_snapshot else "Uncategorised Sector"],
            "company_logo": company_logo,
            "banner_image": banner_image,
            "tradingview_url": f"https://www.tradingview.com/symbols/NSE-{canonical_symbol}/" if canonical_symbol else None,
            "sentiment_badge": sent,
            "llm_metadata": {"used": bool(llm_meta.get("ok")), "model": llm_meta.get("model") if isinstance(llm_meta, dict) else None},
            "keywords": sent.get("raw_responses", {}).get("keyword", {}).get("matches", {}).get("positive", []),
            "processing_events": evts,
            "final_status": "processed",
            "created_at": created_at,
            "version": VERSION
        }

        # 9) Write JSON locally and move PDF to local processed folder (date-based)
        if announcement_iso:
            try:
                dt = datetime.fromisoformat(announcement_iso)
                json_dir = PROCESSED_JSON_BASE / dt.strftime("%Y-%m-%d")
            except Exception:
                json_dir = PROCESSED_JSON_BASE / datetime.now(IST).strftime("%Y-%m-%d")
        else:
            json_dir = PROCESSED_JSON_BASE / datetime.now(IST).strftime("%Y-%m-%d")

        out_file = json_dir / f"{file_id}.json"
        if not dry_run:
            json_dir.mkdir(parents=True, exist_ok=True)
            write_master_json(out_file, master)

            # move PDF into processed folder
            if announcement_iso:
                try:
                    dt = datetime.fromisoformat(announcement_iso)
                    dest_dir = PROCESSED_PDF_BASE / dt.strftime("%Y-%m-%d")
                except Exception:
                    dest_dir = PROCESSED_PDF_BASE / datetime.now(IST).strftime("%Y-%m-%d")
            else:
                dest_dir = PROCESSED_PDF_BASE / datetime.now(IST).strftime("%Y-%m-%d")

            dest_dir.mkdir(parents=True, exist_ok=True)
            new_pdf_path = dest_dir / pdf_path.name
            try:
                shutil.move(str(pdf_path), str(new_pdf_path))
                evts.append({"event": "moved_pdf", "ts": now_iso(), "to": str(new_pdf_path)})
            except Exception as e:
                evts.append({"event": "move_failed", "ts": now_iso(), "error": str(e)})
        else:
            evts.append({"event": "dry_run_skipped_write", "ts": now_iso()})

        return master

    except Exception as e:
        tb = traceback.format_exc()
        evts.append({"event": "error", "ts": now_iso(), "error": str(e), "trace": tb})
        try:
            err_path = ERROR_DIR / f"{Path(pdf_path).name}.error.json"
            with err_path.open("w", encoding="utf-8") as fh:
                json.dump(make_json_safe({"processing_events": evts, "trace": tb}), fh, ensure_ascii=False, indent=2)
        except Exception:
            pass
        # re-raise to let CLI/caller see failure
        raise


# ---------------------------
# S3-aware wrapper (S3-first, local fallback)
# ---------------------------


def _is_s3_uri(u: str) -> bool:
    return isinstance(u, str) and u.lower().startswith("s3://")


def _parse_s3_uri(u: str) -> Tuple[str, str]:
    """
    Return (bucket, key) for a valid s3://bucket/key URI.
    Raises ValueError if invalid.
    """
    m = re.match(r"s3://([^/]+)/(.+)", u)
    if not m:
        raise ValueError(f"Invalid S3 URI: {u}")
    return m.group(1), m.group(2)


def process_single_pdf(
    s3_or_local_path: str,
    dry_run: bool = False,
    force: bool = False,
    overwrite: bool = False,
    delete_after: bool = False,
    download_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    S3-aware wrapper that accepts:
      - local path string OR
      - s3://bucket/key.pdf

    Behavior:
      - If S3 URI: download to temp file (fsspec preferred, boto3 fallback), call process_pdf(local_path),
        then upload:
           * processed PDF -> s3://<bucket>/input_data/pdf/processed/<YYYY-MM-DD>/<filename>
           * JSON -> s3://<bucket>/data/announcements/<YYYY-MM-DD>/<file_id>.json
      - If local path: call process_pdf(path) (local behavior unchanged).
      - Respects overwrite flag for S3 uploads (if overwrite False, will skip existing target).
      - Returns {"master": master_dict, "upload": {...}} where upload contains upload booleans and error list.
    """
    upload_results = {"pdf_uploaded": False, "json_uploaded": False, "errors": []}
    is_s3 = _is_s3_uri(str(s3_or_local_path))
    local_temp: Optional[Path] = None
    cleanup_paths: List[Path] = []

    # Helper to upload file via fsspec or boto3
    def _upload_file_to_s3(local_file: Path, bucket: str, key: str, allow_overwrite: bool) -> bool:
        # prefer fsspec
        if fsspec is not None:
            try:
                fs = fsspec.filesystem("s3")
                # Check existence if not overwrite
                exists = False
                if not allow_overwrite:
                    try:
                        exists = fs.exists(f"s3://{bucket}/{key}")
                    except Exception:
                        # ignore existence check errors and attempt upload
                        exists = False
                if exists and not allow_overwrite:
                    return False
                # Use a binary stream copy
                with fs.open(f"s3://{bucket}/{key}", "wb") as w, local_file.open("rb") as r:
                    while True:
                        chunk = r.read(16 * 1024)
                        if not chunk:
                            break
                        w.write(chunk)
                return True
            except Exception as e:
                # fallback to boto3 if available
                upload_results["errors"].append(f"fsspec upload failed for s3://{bucket}/{key} : {e}")
        if boto3 is not None:
            try:
                s3c = boto3.client("s3")
                if not allow_overwrite:
                    # Check existence by head_object
                    try:
                        s3c.head_object(Bucket=bucket, Key=key)
                        # exists
                        return False
                    except Exception:
                        # not exists proceed
                        pass
                s3c.upload_file(str(local_file), bucket, key)
                return True
            except Exception as e:
                upload_results["errors"].append(f"boto3 upload failed for s3://{bucket}/{key} : {e}")
                return False
        upload_results["errors"].append("No S3 upload method available (install fsspec or boto3).")
        return False

    try:
        # 1) If s3 URI: download to local temp file
        if is_s3:
            s3_uri = str(s3_or_local_path)
            bucket, key = _parse_s3_uri(s3_uri)
            # choose temp dir
            tmp_dir = download_dir or Path(tempfile.gettempdir())
            tmp_dir.mkdir(parents=True, exist_ok=True)
            local_name = Path(key).name
            local_temp = tmp_dir / f"{uuid.uuid4().hex[:8]}_{local_name}"
            # Download via fsspec preferred
            downloaded = False
            if fsspec is not None:
                try:
                    fs = fsspec.filesystem("s3")
                    with fs.open(s3_uri, "rb") as r, local_temp.open("wb") as w:
                        while True:
                            chunk = r.read(16 * 1024)
                            if not chunk:
                                break
                            w.write(chunk)
                    downloaded = True
                except Exception as e:
                    download_err = f"fsspec download failed for {s3_uri}: {e}"
                    upload_results["errors"].append(download_err)
                    downloaded = False
            # boto3 fallback
            if not downloaded:
                if boto3 is None:
                    upload_results["errors"].append("boto3 not installed and fsspec download failed; cannot download S3 object.")
                    raise RuntimeError("S3 download failed: no available client")
                try:
                    s3c = boto3.client("s3")
                    s3c.download_file(bucket, key, str(local_temp))
                except Exception as e:
                    upload_results["errors"].append(f"boto3 download failed for s3://{bucket}/{key}: {e}")
                    raise

            local_pdf_path = local_temp
            cleanup_paths.append(local_temp)
        else:
            local_pdf_path = Path(s3_or_local_path)
            if not local_pdf_path.exists():
                raise FileNotFoundError(f"Local file not found: {local_pdf_path}")

        # 2) Process locally
        master = process_pdf(local_pdf_path, dry_run=dry_run, force=force)

        # 3) If dry_run: return master and no uploads
        if dry_run or master is None:
            return {"master": master, "upload": upload_results}

        # 4) If input was S3 -> upload JSON and processed PDF to bucket with date-based layout
        if is_s3:
            ann_iso = master.get("announcement_datetime_iso")
            try:
                if ann_iso:
                    dt = datetime.fromisoformat(ann_iso)
                    date_folder = dt.strftime("%Y-%m-%d")
                else:
                    date_folder = datetime.now(IST).strftime("%Y-%m-%d")
            except Exception:
                date_folder = datetime.now(IST).strftime("%Y-%m-%d")

            file_id = master.get("id") or ("ann_" + hashlib.sha1(str(master).encode("utf-8")).hexdigest()[:10])
            json_name = f"{file_id}.json"
            pdf_name = Path(master.get("source_file_normalized") or local_pdf_path.name).name

            # S3 key targets (allow optional bucket_root prefix)
            # Root S3 prefix (if provided via s3_prefix or env fallback)
            bucket_root = s3_prefix.rstrip('/') if ('s3_prefix' in locals() and s3_prefix) else (os.getenv('S3_BUCKET_ROOT','').rstrip('/') if os.getenv('S3_BUCKET_ROOT') else '')
            # PDF processed S3 key: <bucket_root>/input_data/pdf/processed/<date_folder>/<filename>
            pdf_key = f"{bucket_root}/input_data/pdf/processed/{date_folder}/{pdf_name}".lstrip('/')
            # JSON S3 key: <bucket_root>/data/announcements/<date_folder>/<file_id>.json
            json_key = f"{bucket_root}/data/announcements/{date_folder}/{json_name}".lstrip('/')

            # --- NEW: populate expected S3 URIs on the master regardless of dry_run ---
            try:
                master['s3_output_json'] = f"s3://{bucket}/{json_key}"
                master['s3_output_pdf'] = f"s3://{bucket}/{pdf_key}"
            except Exception:
                # defensive: if master isn't a dict for some reason, ignore
                pass
            # --- end NEW ---

            # upload JSON (write to tmp json and upload)
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as jf:
                json.dump(make_json_safe(master), jf, ensure_ascii=False, indent=2)
                jf_path = Path(jf.name)
            cleanup_paths.append(jf_path)

            json_ok = _upload_file_to_s3(jf_path, bucket, json_key, allow_overwrite=overwrite)
            upload_results["json_uploaded"] = bool(json_ok)
            if json_ok:
                master["s3_output_json"] = f"s3://{bucket}/{json_key}"

            # upload processed PDF (local file processed could have been moved by process_pdf to local processed folder).
            # prefer the moved location recorded in master.processing_events if present
            processed_local_candidate: Optional[Path] = None
            for ev in reversed(master.get("processing_events", []) or []):
                if ev.get("event") == "moved_pdf" and ev.get("to"):
                    cand = Path(ev.get("to"))
                    if cand.exists():
                        processed_local_candidate = cand
                        break
            # fallback to the local_temp or local_pdf_path used for processing
            if processed_local_candidate is None:
                processed_local_candidate = local_pdf_path

            pdf_ok = _upload_file_to_s3(processed_local_candidate, bucket, pdf_key, allow_overwrite=overwrite)
            upload_results["pdf_uploaded"] = bool(pdf_ok)
            if pdf_ok:
                master["s3_output_pdf"] = f"s3://{bucket}/{pdf_key}"

            # Optionally delete original S3 object
            if delete_after:
                # use boto3 if available to delete
                if boto3 is not None:
                    s3c = boto3.client("s3")
                    try:
                        s3c.delete_object(Bucket=bucket, Key=key)
                        master["s3_pdf_deleted"] = True
                    except Exception as e:
                        upload_results["errors"].append(f"Failed to delete original s3://{bucket}/{key}: {e}")
                else:
                    upload_results["errors"].append("delete_after requested but boto3 not installed; cannot delete original S3 object.")
        else:
            # local-only: nothing further to upload. All writes already happened in process_pdf
            pass

        return {"master": master, "upload": upload_results}

    finally:
        # cleanup any temporary files we created
        for p in cleanup_paths:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


# ---------------------------
# CLI
# ---------------------------


def _cli():
    parser = argparse.ArgumentParser(description="pdf_processor - process pdf into master JSON (S3-aware)")
    parser.add_argument("--src", help="Source PDF path or s3 URI (e.g. s3://bucket/input_data/pdf/foo.pdf or input_data/pdf/foo.pdf)", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Do not write JSON or move PDFs locally; when S3 input then no uploads will be performed")
    parser.add_argument("--force", action="store_true", help="Force reprocessing (internal semantics, not strictly implemented)")
    parser.add_argument("--overwrite", action="store_true", help="When uploading to S3, overwrite existing targets")
    parser.add_argument("--delete-after", action="store_true", help="When input is S3, delete original object after successful uploads")
    args = parser.parse_args()

    src = args.src
    # warm caches (best-effort)
    try:
        csv_utils.load_processed_df(force_reload=False)
    except Exception:
        pass
    try:
        index_builder.refresh_index()
    except Exception:
        pass

    try:
        res = process_single_pdf(src, dry_run=args.dry_run, force=args.force, overwrite=args.overwrite, delete_after=args.delete_after)
        master = res.get("master", res)
        print("Processed:", master.get("id"))
        print("Symbol:", master.get("canonical_symbol"))
        print("Upload:", res.get("upload"))
    except Exception as e:
        print("Processing failed:", e, file=sys.stderr)
        raise


if __name__ == "__main__":
    _cli()