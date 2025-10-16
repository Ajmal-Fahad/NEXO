# services/filename_utils.py
"""
Robust filename-based company matching utilities.

Provides:
 - normalize_name(name_or_filename) -> normalized lowercase string
 - strip_timestamp_and_ext(filename) -> string without trailing timestamp and extension
 - extract_datetime_from_filename(filename) -> (iso_str | None, human_str | None)
 - filename_tokens(name_only) -> List[str] tokens suitable for index_builder
 - filename_to_symbol(filename, csv_path=None, ...) -> dict:
       { found, symbol, company_name, score, match_type, candidates }

Behavior:
 - Prefer using services.index_builder if available (fast in-memory index + token/fuzzy APIs).
 - Otherwise fall back to scanning processed CSV files.
 - Disambiguation logic: prefix-match (3-10 chars), token-by-token, token-overlap, fuzzy lookup.
 - Defensive: never raises on errors; returns predictable dict.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import logging
from difflib import SequenceMatcher
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

logger = logging.getLogger(__name__)

# Config
DEFAULT_JACCARD_THRESH = 0.35
DEFAULT_RATIO_THRESH = 0.65
CSV_GLOB = "input_data/csv/processed_csv/*.csv"
TOKEN_MIN_LEN = 2
PREFIX_MIN = 3
PREFIX_MAX = 10

# Abbreviation expansions (used by normalize_name)
_ABBREV_MAP = {
    r"\bltd\b": "limited",
    r"\bco\b": "company",
    r"\bpvt\b": "private",
    r"\b&\b": "and",
}

_nonword_re = re.compile(r"[^a-z0-9\s]+")


# ---- Basic helpers ----
def normalize_name(s: str) -> str:
    """Normalize a filename or company name to simple lowercase tokens.
    Keeps spaces between words; expands common abbreviations and removes punctuation."""
    try:
        if not s:
            return ""
        name = os.path.basename(s)
        name = name.replace("_", " ").replace("-", " ")
        # strip extension
        name = re.sub(r"\.[a-z0-9]+$", "", name, flags=re.IGNORECASE)
        name = name.strip()
        low = name.lower()
        for pat, rep in _ABBREV_MAP.items():
            low = re.sub(pat, rep, low, flags=re.IGNORECASE)
        low = _nonword_re.sub(" ", low)
        low = re.sub(r"\s+", " ", low).strip()
        return low
    except Exception:
        return ""


def _tokens_from_normalized(norm: str) -> List[str]:
    if not norm:
        return []
    toks = [t for t in norm.split() if t and len(t) >= TOKEN_MIN_LEN]
    return toks


def _compact(s: Optional[str]) -> str:
    """Remove non-alnum and uppercase compact form (for compact matching)."""
    if not s:
        return ""
    return re.sub(r"[^A-Za-z0-9]+", "", str(s)).upper()


def _jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    a = set(a_tokens)
    b = set(b_tokens)
    if not a and not b:
        return 0.0
    inter = a & b
    union = a | b
    return float(len(inter)) / float(len(union)) if union else 0.0


def _similarity_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# ---- CSV helpers (fallback) ----
def _load_latest_csv(csv_path: Optional[str] = None) -> Optional[str]:
    try:
        if csv_path:
            return csv_path if os.path.exists(csv_path) else None
        files = sorted(glob.glob(CSV_GLOB))
        return files[-1] if files else None
    except Exception:
        return None


def _iter_csv_rows(csv_file: str):
    """Yield dict rows from csv_file. Defensive; yields nothing on error."""
    try:
        with open(csv_file, encoding="utf-8", errors="replace") as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                yield row
    except Exception:
        return
        yield  # make this a generator even on error (no rows)


# ---- Filename datetime helpers ----
# Support common filename timestamp patterns like:
#   "Company Name_04-10-2025 09_34_18.pdf" or "Company-Name 04-10-2025 09:34:18.pdf"
_dt_re = re.compile(r"(?P<d>\d{2}[-/]\d{2}[-/]\d{4})[ _-]+(?P<t>\d{2}[:_]\d{2}[:_]\d{2})")


def extract_datetime_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract datetime from filename.
    Returns (iso_string or None, human_readable or None).
    ISO uses naive local datetime string (no tz).
    """
    try:
        if not filename:
            return None, None
        base = os.path.basename(filename)
        m = _dt_re.search(base)
        if not m:
            return None, None
        date_str = m.group("d").replace("/", "-")
        time_str = m.group("t").replace("_", ":")
        dt = datetime.strptime(f"{date_str} {time_str}", "%d-%m-%Y %H:%M:%S")
        iso = dt.isoformat()
        human = dt.strftime("%d-%b-%Y, %H:%M:%S")
        return iso, human
    except Exception:
        return None, None


def strip_timestamp_and_ext(filename: str) -> str:
    """Remove trailing timestamp and extension from filename; returns the cleaned 'company-like' name."""
    try:
        if not filename:
            return ""
        name = os.path.basename(filename)
        # remove extension
        if "." in name:
            name = ".".join(name.split(".")[:-1])
        # Remove trailing datetime pattern if present
        name = re.sub(r'[_ \-]*\d{2}[-/]\d{2}[-/]\d{4}[_ \-]+\d{2}[:_]\d{2}[:_]\d{2}$', "", name)
        name = name.strip()
        # collapse repeated punctuation/underscores/hyphens -> space
        name = re.sub(r"[_\-]+", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name
    except Exception:
        return ""


def filename_tokens(name_only: str) -> List[str]:
    """Return uppercase tokens for filename-based matching."""
    try:
        if not name_only:
            return []
        parts = re.split(r"[^A-Za-z0-9]+", name_only)
        toks = [p.upper() for p in parts if p and len(p) >= TOKEN_MIN_LEN]
        return toks
    except Exception:
        return []


# ---- Prefix match helper ----
def _prefix_from_filename(name_only: str) -> str:
    """Compact filename (remove non-alnum) and return PREFIX_MIN..PREFIX_MAX chars uppercased."""
    if not name_only:
        return ""
    compacted = re.sub(r"[^A-Za-z0-9]+", "", name_only).upper()
    return compacted[:PREFIX_MAX]


def _try_prefix_match_with_index(index_builder, name_only: str) -> Optional[Tuple[str, str]]:
    """
    Try to match using the first 3..10 chars of compacted filename against index_builder's _symbol_map keys
    and company_name values. Returns (symbol, company_name) or None.
    """
    try:
        prefix = _prefix_from_filename(name_only)
        if not prefix or len(prefix) < PREFIX_MIN:
            return None
        # access internal symbol map if available (best-effort)
        sym_map = getattr(index_builder, "_symbol_map", None)
        if isinstance(sym_map, dict):
            # check symbols that startwith prefix
            for sym, row in sym_map.items():
                try:
                    if not sym:
                        continue
                    if str(sym).upper().startswith(prefix):
                        return sym, (row.get("company_name") if isinstance(row, dict) else None)
                except Exception:
                    continue
            # check compact company names
            for sym, row in sym_map.items():
                try:
                    comp_name = (row.get("company_name") if isinstance(row, dict) else None) or ""
                    comp_compact = re.sub(r"[^A-Za-z0-9]+", "", str(comp_name)).upper()
                    if comp_compact.startswith(prefix):
                        return sym, (row.get("company_name") if isinstance(row, dict) else None)
                except Exception:
                    continue
    except Exception:
        pass
    return None


def _try_prefix_match_in_csv(csv_file: str, name_only: str) -> Optional[Tuple[str, str]]:
    """Prefix match scanning CSV rows; returns (symbol, company_name) or None."""
    try:
        prefix = _prefix_from_filename(name_only)
        if not prefix or len(prefix) < PREFIX_MIN:
            return None
        for row in _iter_csv_rows(csv_file):
            try:
                raw_symbol = (row.get("symbol") or row.get("SYMBOL") or row.get("Symbol") or row.get("ticker") or "").strip()
                raw_name = (row.get("company_name") or row.get("Company Name") or row.get("Company") or row.get("company") or "").strip()
                if not raw_symbol and not raw_name:
                    continue
                if raw_symbol and re.sub(r"[^A-Za-z0-9]+", "", raw_symbol).upper().startswith(prefix):
                    return raw_symbol.upper(), raw_name or None
                if raw_name and re.sub(r"[^A-Za-z0-9]+", "", raw_name).upper().startswith(prefix):
                    return raw_symbol.upper() if raw_symbol else None, raw_name
            except Exception:
                continue
    except Exception:
        pass
    return None


# ---- Primary matching function (improved) ----
def filename_to_symbol(
    filename: str,
    csv_path: Optional[str] = None,
    jaccard_thresh: float = DEFAULT_JACCARD_THRESH,
    ratio_thresh: float = DEFAULT_RATIO_THRESH,
) -> Dict[str, Any]:
    """
    Map filename -> symbol/company_name using index_builder if available; else falls back to CSV scan.
    Returns:
      {
        found: bool,
        symbol: Optional[str],
        company_name: Optional[str],
        score: float,
        match_type: str,
        candidates: [ {jaccard, ratio, symbol, company_name}, ... ]
      }
    """
    result: Dict[str, Any] = {
        "found": False,
        "symbol": None,
        "company_name": None,
        "score": 0.0,
        "match_type": "no_match",
        "candidates": [],
    }

    try:
        if not filename:
            return result

        base_no_ext = strip_timestamp_and_ext(filename)
        norm_filename = normalize_name(base_no_ext)
        fn_tokens = _tokens_from_normalized(norm_filename)
        raw_tokens = filename_tokens(base_no_ext)

        # try index_builder first
        index_builder = None
        try:
            from services import index_builder as _ib  # type: ignore
            index_builder = _ib
        except Exception:
            try:
                import index_builder as _ib  # fallback
                index_builder = _ib
            except Exception:
                index_builder = None

        # If index_builder exists, use a fast, multi-step strategy
        if index_builder is not None:
            try:
                # 0) prefix-match against index (new heuristic)
                try:
                    pref = _try_prefix_match_with_index(index_builder, base_no_ext)
                    if pref:
                        sym, cname = pref
                        row = index_builder.get_company_by_symbol(sym) if hasattr(index_builder, "get_company_by_symbol") else None
                        company_name = cname or (row.get("company_name") if row else None)
                        result.update({"found": True, "symbol": sym, "company_name": company_name, "score": 0.95, "match_type": "prefix_index"})
                        return result
                except Exception:
                    pass

                # 1) Exact / compact match (company or symbol)
                try:
                    sym = None
                    if hasattr(index_builder, "get_symbol_by_exact"):
                        sym = index_builder.get_symbol_by_exact(base_no_ext) or index_builder.get_symbol_by_exact(norm_filename)
                    if sym:
                        row = index_builder.get_company_by_symbol(sym) if hasattr(index_builder, "get_company_by_symbol") else None
                        result.update({"found": True, "symbol": sym, "company_name": row.get("company_name") if row else None, "score": 1.0, "match_type": "exact"})
                        return result
                except Exception:
                    pass

                # 2) Token-by-token: scan tokens in order, pick first unique mapping
                try:
                    for t in raw_tokens:
                        if not t or len(t) < TOKEN_MIN_LEN:
                            continue
                        if hasattr(index_builder, "get_symbol_by_token"):
                            try:
                                sym = index_builder.get_symbol_by_token(t)
                            except Exception:
                                sym = None
                            if sym:
                                row = index_builder.get_company_by_symbol(sym) if hasattr(index_builder, "get_company_by_symbol") else None
                                result.update({"found": True, "symbol": sym, "company_name": row.get("company_name") if row else None, "score": 0.85, "match_type": "token"})
                                return result
                except Exception:
                    pass

                # 3) Token-overlap search across tokens
                try:
                    if hasattr(index_builder, "token_overlap_search"):
                        overlap = index_builder.token_overlap_search(fn_tokens or raw_tokens)
                        if overlap:
                            sym, score = overlap
                            row = index_builder.get_company_by_symbol(sym) if hasattr(index_builder, "get_company_by_symbol") else None
                            result.update({"found": True, "symbol": sym, "company_name": row.get("company_name") if row else None, "score": float(score), "match_type": "token_overlap"})
                            return result
                except Exception:
                    pass

                # 4) Fuzzy lookup (conservative)
                try:
                    if hasattr(index_builder, "fuzzy_lookup_symbol"):
                        fuzzy = index_builder.fuzzy_lookup_symbol(base_no_ext) or index_builder.fuzzy_lookup_symbol(norm_filename)
                        if fuzzy:
                            row = index_builder.get_company_by_symbol(fuzzy) if hasattr(index_builder, "get_company_by_symbol") else None
                            result.update({"found": True, "symbol": fuzzy, "company_name": row.get("company_name") if row else None, "score": 0.7, "match_type": "fuzzy"})
                            return result
                except Exception:
                    pass

                # no index path match
                result["match_type"] = "no_match"
                return result
            except Exception as e:
                logger.exception("index_builder path failed: %s", e)
                # fall through to CSV scan
                pass

        # ---- Fallback: CSV scan (if index_builder missing or failed) ----
        csv_file = _load_latest_csv(csv_path)
        if not csv_file:
            result["match_type"] = "no_index"
            return result

        # 0) prefix match in CSV (quick)
        try:
            pref_csv = _try_prefix_match_in_csv(csv_file, base_no_ext)
            if pref_csv:
                sym, cname = pref_csv
                result.update({"found": True, "symbol": sym, "company_name": cname, "score": 0.9, "match_type": "prefix_csv"})
                return result
        except Exception:
            pass

        # compute jaccard & ratio for every row and keep top candidates
        top: Tuple[float, float, str, str] = (0.0, 0.0, "", "")  # (jaccard, ratio, symbol, company_name)
        candidates = []
        try:
            for row in _iter_csv_rows(csv_file):
                try:
                    raw_name = (row.get("company_name") or row.get("Company Name") or row.get("Company") or row.get("company") or "").strip()
                    raw_symbol = (row.get("symbol") or row.get("SYMBOL") or row.get("Symbol") or row.get("ticker") or "").strip().upper()
                    if not raw_name:
                        continue
                    norm_name = normalize_name(raw_name)
                    name_tokens = _tokens_from_normalized(norm_name)
                    j = _jaccard(fn_tokens, name_tokens)
                    r = _similarity_ratio(norm_filename, norm_name)
                    candidates.append((j, r, raw_symbol, raw_name, norm_name))
                    if (j, r) > (top[0], top[1]):
                        top = (j, r, raw_symbol, raw_name)
                except Exception:
                    continue
        except Exception as e:
            logger.exception("Error iterating CSV rows in filename_to_symbol: %s", e)

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        result["candidates"] = [{"jaccard": round(c[0], 3), "ratio": round(c[1], 3), "symbol": c[2], "company_name": c[3]} for c in candidates[:10]]

        top_j, top_r, top_sym, top_name = top
        result["score"] = float(max(top_j, top_r))
        if top_j >= jaccard_thresh or top_r >= ratio_thresh:
            result.update({"found": True, "symbol": top_sym or None, "company_name": top_name or None, "score": result["score"], "match_type": "index_builder"})
            return result

        # final fallback heuristic: dominant token among top candidates
        try:
            token_counts: Dict[str, int] = {}
            for c in candidates:
                sym = c[2] or ""
                token_counts[sym] = token_counts.get(sym, 0) + 1
            if token_counts:
                best_sym, cnt = max(token_counts.items(), key=lambda x: x[1])
                if cnt >= 2 or len(fn_tokens) == 1:
                    for c in candidates:
                        if c[2] == best_sym:
                            result.update({"found": True, "symbol": c[2], "company_name": c[3], "score": float(max(top_j, top_r)), "match_type": "token_dominant"})
                            return result
        except Exception:
            pass

        result["match_type"] = "no_match"
        return result

    except Exception as e:
        logger.exception("filename_to_symbol top-level failure: %s", e)
        return result


# CLI helper for quick tests
if __name__ == "__main__":
    import sys
    import json

    tests = sys.argv[1:] or [
        "HDFC Bank Ltd_04-10-2025 09_34_18.pdf",
        "AXIS Bank Ltd_03-10-2025 16_26_34.pdf",
        "Yes Bank_20-09-2025 22_46_38.pdf",
        "Perfect-Octave Media Projects Ltd_27-09-2025 02_01_29.pdf",
    ]
    out = []
    for t in tests:
        try:
            iso, human = extract_datetime_from_filename(t)
            stripped = strip_timestamp_and_ext(t)
            toks = filename_tokens(stripped)
            match = filename_to_symbol(t)
            j = {
                "file": t,
                "iso": iso,
                "human": human,
                "stripped": stripped,
                "tokens": toks,
                "match": match,
            }
            out.append(j)
        except Exception as e:
            out.append({"file": t, "error": str(e)})
    print(json.dumps(out, ensure_ascii=False, indent=2))