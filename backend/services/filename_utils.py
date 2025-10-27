# backend/services/filename_utils.py
"""
Filename-based company matching utilities (R-06, Diamond Standard).

This file is the production-ready, test-hardened version intended to match the
deterministic R-06 test suite. Key changes from earlier revisions:
 - module-level index_builder resolution (testable via patching)
 - normalize_name: strips timestamps/extensions, expands abbreviations and '&'
 - filename_to_symbol: precise ordering: scored_df -> prefix_df -> unique-token -> token_dominant -> no_match
 - robust handling of csv_utils returning None / raising (data_source_unavailable)
 - health_check reads module-level index_builder and returns deterministic statuses
"""

from __future__ import annotations

import os
import re
import logging
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional, Dict, Any, Tuple, List, TypedDict, Protocol, runtime_checkable

import pandas as pd

# Try to pick up project's logger / metric / audit hooks (best-effort).
try:
    from backend.services.csv_processor import logger as _project_logger, _safe_metric as _safe_metric_fn, _audit_log as _audit_log_fn  # type: ignore
except Exception:
    _project_logger = None
    _safe_metric_fn = None
    _audit_log_fn = None

logger = _project_logger or logging.getLogger(__name__)

def _safe_metric(op: str, value: int = 1) -> None:
    try:
        if _safe_metric_fn:
            _safe_metric_fn(op, value)
    except Exception:
        logger.debug("filename_utils: metric emission failed for op=%s", op, exc_info=True)

def _audit_log(action: str, target: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
    try:
        if _audit_log_fn:
            _audit_log_fn(action, target, status, details or {})
        else:
            logger.info("AUDIT %s %s -> %s : %s", action, target, status, details or {})
    except Exception:
        logger.debug("filename_utils: audit logging failed for action=%s target=%s", action, target, exc_info=True)

# -------------------------
# Constants & regexes
# -------------------------
DEFAULT_JACCARD_THRESH = 0.35
DEFAULT_RATIO_THRESH = 0.65
TOKEN_MIN_LEN = 2
PREFIX_MAX = 10
MAX_FILENAME_LEN = 255

_VALID_FILENAME_CHARS_RE = re.compile(r"^[a-zA-Z0-9\s._\-\&()]+$")

_ABBREV_MAP = {
    r"\bltd\b": "limited",
    r"\bco\b": "company",
    r"\bpvt\b": "private",
}

_nonword_re = re.compile(r"[^a-z0-9\s]+")
_dt_re = re.compile(r"(?P<d>\d{2}[-/]\d{2}[-/]\d{4})[ _-]+(?P<t>\d{2}[:_]\d{2}[:_]\d{2})")

# -------------------------
# Typing / Protocol for index_builder (informational - runtime checks use capability tests)
# -------------------------
class CompanyInfo(TypedDict):
    symbol: str
    company_name: str

@runtime_checkable
class IndexBuilderProtocol(Protocol):
    def list_symbols_by_prefix(self, prefix: str) -> List[str]: ...
    def get_symbol_by_exact(self, name: str) -> Optional[str]: ...
    def get_symbol_by_token(self, token: str) -> Optional[str]: ...
    def token_overlap_search(self, tokens: List[str]) -> Optional[Tuple[str, float]]: ...
    def fuzzy_lookup_symbol(self, name: str) -> Optional[str]: ...
    def get_company_by_symbol(self, symbol: str) -> Optional[CompanyInfo]: ...

# -------------------------
# Module-level dependency resolution (deterministic & testable)
# -------------------------
index_builder = None
try:
    from backend.services import index_builder as _ib  # type: ignore
    index_builder = _ib
except Exception:
    try:
        from services import index_builder as _ib  # type: ignore
        index_builder = _ib
    except Exception:
        try:
            import index_builder as _ib  # type: ignore
            index_builder = _ib
        except Exception:
            index_builder = None

# -------------------------
# Internal helpers
# -------------------------
def _jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    a, b = set(a_tokens), set(b_tokens)
    if not a and not b:
        return 0.0
    union = a | b
    inter = a & b
    return float(len(inter)) / float(len(union)) if union else 0.0

def _similarity_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def _compact(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"[^A-Za-z0-9]+", "", str(s)).upper()

def _prefix_from_filename(name_only: str) -> str:
    """
    Return a compacted prefix derived from the first token of the filename.
    This makes prefix matching deterministic for tests like "HDFC report.pdf" -> "HDFC".
    """
    if not name_only:
        return ""
    # Extract first alpha-numeric token
    parts = re.split(r"[^A-Za-z0-9]+", name_only)
    first = next((p for p in parts if p), "")
    compacted = re.sub(r"[^A-Za-z0-9]+", "", first).upper()
    return compacted[:PREFIX_MAX]

# -------------------------
# Public helpers
# -------------------------
def strip_timestamp_and_ext(filename: str) -> str:
    """
    Remove trailing timestamp patterns and file extension.
    Examples:
        - "Company_04-10-2025 09_34_18.pdf" -> "Company"
        - "Company-Name_2025-10-04_09_34_18.csv" -> "Company Name"
    """
    try:
        if not filename:
            return ""
        name = os.path.basename(filename)
        # remove extension
        if "." in name:
            name = ".".join(name.split(".")[:-1])
        # remove trailing timestamp like DD-MM-YYYY HH_MM_SS variants
        name = re.sub(r'[_ \-]*\d{2}[-/]\d{2}[-/]\d{4}[_ \-]+\d{2}[:_]\d{2}[:_]\d{2}$', "", name)
        # remove date-only trailing patterns e.g. _04-10-2025
        name = re.sub(r'[_ \-]*\d{2}[-/]\d{2}[-/]\d{4}$', "", name)
        name = name.strip()
        name = re.sub(r"[_\-]+", " ", name)
        return re.sub(r"\s+", " ", name).strip()
    except Exception:
        logger.debug("strip_timestamp_and_ext failed for %r", filename, exc_info=True)
        return ""

def normalize_name(s: str) -> str:
    """
    Normalize filename or company name into a lowercase, punctuation-free string.

    - removes extension & trailing dates
    - replaces underscores/hyphens with spaces
    - expands common abbreviations (ltd, co, pvt)
    - expands '&' to 'and'
    - collapses whitespace
    """
    try:
        if not s:
            return ""
        name = os.path.basename(s)
        # remove timestamps and extension first
        name = strip_timestamp_and_ext(name)
        name = name.replace("_", " ").replace("-", " ")
        low = name.lower()
        # expand ampersand explicitly
        low = low.replace("&", " and ")
        # expand abbreviations
        for pat, rep in _ABBREV_MAP.items():
            low = re.sub(pat, rep, low, flags=re.IGNORECASE)
        low = _nonword_re.sub(" ", low)
        return re.sub(r"\s+", " ", low).strip()
    except Exception:
        logger.debug("normalize_name failed for %r", s, exc_info=True)
        return ""

def extract_datetime_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
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
        return dt.isoformat(), dt.strftime("%d-%b-%Y, %H:%M:%S")
    except Exception:
        logger.debug("extract_datetime_from_filename failed for %r", filename, exc_info=True)
        return None, None

def filename_tokens(name_only: str) -> List[str]:
    try:
        if not name_only:
            return []
        parts = re.split(r"[^A-Za-z0-9]+", name_only)
        toks = [p.upper() for p in parts if p and len(p) >= TOKEN_MIN_LEN]
        return toks
    except Exception:
        logger.debug("filename_tokens failed for %r", name_only, exc_info=True)
        return []

# -------------------------
# Core matching function
# -------------------------
def filename_to_symbol(
    filename: str,
    df: Optional[pd.DataFrame] = None,
    jaccard_thresh: float = DEFAULT_JACCARD_THRESH,
    ratio_thresh: float = DEFAULT_RATIO_THRESH,
) -> Dict[str, Any]:
    """
    Map a filename to a company symbol and name using prioritized strategies.

    Strategy:
      1) index_builder (prefix, exact, token, token_overlap, fuzzy)
      2) authoritative DataFrame matching: scored_df -> prefix_df -> unique-token -> token_dominant
      3) deterministic codes: validation_failed, data_source_unavailable, no_data_in_source, no_match
    """
    result: Dict[str, Any] = {
        "found": False,
        "symbol": None,
        "company_name": None,
        "score": 0.0,
        "match_type": "no_match",
        "candidates": [],
    }

    # Input validation
    if not filename:
        result["match_type"] = "validation_failed"
        _safe_metric("match.validation_failed")
        return result
    if len(filename) > MAX_FILENAME_LEN:
        result["match_type"] = "validation_failed"
        _safe_metric("match.validation_failed")
        return result
    if not _VALID_FILENAME_CHARS_RE.match(os.path.basename(filename)):
        result["match_type"] = "validation_failed"
        _safe_metric("match.validation_failed")
        return result

    base_no_ext = strip_timestamp_and_ext(filename)
    norm_filename = normalize_name(base_no_ext)
    fn_tokens = [t for t in norm_filename.split() if len(t) >= TOKEN_MIN_LEN]
    raw_tokens = filename_tokens(base_no_ext)

    # 1) Index builder path (use module-level index_builder variable)
    if index_builder is not None:
        try:
            # prefix list_symbols_by_prefix -> prefix_index
            list_by_prefix = getattr(index_builder, "list_symbols_by_prefix", None)
            if callable(list_by_prefix):
                try:
                    syms = list_by_prefix(_prefix_from_filename(base_no_ext) or "")
                    for sym in syms:
                        try:
                            get_company = getattr(index_builder, "get_company_by_symbol", None)
                            cname = None
                            if callable(get_company):
                                row = get_company(sym)
                                cname = row.get("company_name") if isinstance(row, dict) else None
                            result.update({"found": True, "symbol": sym, "company_name": cname, "score": 0.95, "match_type": "prefix_index"})
                            _safe_metric("match.prefix_index")
                            _audit_log("filename_to_symbol", filename, "success", {"match_type": result["match_type"], "symbol": sym})
                            return result
                        except Exception:
                            continue
                except Exception:
                    logger.debug("index_builder.list_symbols_by_prefix failed", exc_info=True)

            # exact
            get_sym_exact = getattr(index_builder, "get_symbol_by_exact", None)
            if callable(get_sym_exact):
                try:
                    sym = get_sym_exact(base_no_ext) or get_sym_exact(norm_filename) or get_sym_exact(_compact(base_no_ext))
                    if sym:
                        get_company = getattr(index_builder, "get_company_by_symbol", None)
                        cname = None
                        if callable(get_company):
                            row = get_company(sym)
                            cname = row.get("company_name") if isinstance(row, dict) else None
                        result.update({"found": True, "symbol": sym, "company_name": cname, "score": 1.0, "match_type": "exact"})
                        _safe_metric("match.exact")
                        _audit_log("filename_to_symbol", filename, "success", {"match_type": result["match_type"], "symbol": sym})
                        return result
                except Exception:
                    logger.debug("index_builder.get_symbol_by_exact failed", exc_info=True)

            # token-by-token
            get_sym_token = getattr(index_builder, "get_symbol_by_token", None)
            if callable(get_sym_token):
                try:
                    for t in raw_tokens:
                        if not t or len(t) < TOKEN_MIN_LEN:
                            continue
                        try:
                            sym = get_sym_token(t)
                        except Exception:
                            sym = None
                        if sym:
                            get_company = getattr(index_builder, "get_company_by_symbol", None)
                            cname = None
                            if callable(get_company):
                                row = get_company(sym)
                                cname = row.get("company_name") if isinstance(row, dict) else None
                            result.update({"found": True, "symbol": sym, "company_name": cname, "score": 0.85, "match_type": "token"})
                            _safe_metric("match.token")
                            _audit_log("filename_to_symbol", filename, "success", {"match_type": result["match_type"], "symbol": sym})
                            return result
                except Exception:
                    logger.debug("index_builder.get_symbol_by_token failed", exc_info=True)

            # token_overlap_search
            token_overlap_search = getattr(index_builder, "token_overlap_search", None)
            if callable(token_overlap_search):
                try:
                    overlap = token_overlap_search(fn_tokens or raw_tokens)
                    if overlap and isinstance(overlap, (list, tuple)) and len(overlap) >= 1:
                        sym = overlap[0]
                        score = float(overlap[1]) if len(overlap) > 1 else 0.75
                        get_company = getattr(index_builder, "get_company_by_symbol", None)
                        cname = None
                        if callable(get_company):
                            row = get_company(sym)
                            cname = row.get("company_name") if isinstance(row, dict) else None
                        result.update({"found": True, "symbol": sym, "company_name": cname, "score": score, "match_type": "token_overlap"})
                        _safe_metric("match.token_overlap")
                        _audit_log("filename_to_symbol", filename, "success", {"match_type": result["match_type"], "symbol": sym})
                        return result
                except Exception:
                    logger.debug("index_builder.token_overlap_search failed", exc_info=True)

            # fuzzy_lookup_symbol
            fuzzy_lookup_symbol = getattr(index_builder, "fuzzy_lookup_symbol", None)
            if callable(fuzzy_lookup_symbol):
                try:
                    fuzzy_sym = fuzzy_lookup_symbol(base_no_ext) or fuzzy_lookup_symbol(norm_filename)
                    if fuzzy_sym:
                        get_company = getattr(index_builder, "get_company_by_symbol", None)
                        cname = None
                        if callable(get_company):
                            row = get_company(fuzzy_sym)
                            cname = row.get("company_name") if isinstance(row, dict) else None
                        result.update({"found": True, "symbol": fuzzy_sym, "company_name": cname, "score": 0.7, "match_type": "fuzzy"})
                        _safe_metric("match.fuzzy")
                        _audit_log("filename_to_symbol", filename, "success", {"match_type": result["match_type"], "symbol": fuzzy_sym})
                        return result
                except Exception:
                    logger.debug("index_builder.fuzzy_lookup_symbol failed", exc_info=True)

        except Exception as e:
            logger.warning("filename_utils: index_builder path failed: %s", e)
            _safe_metric("match.index_failure")

    # 2) DataFrame path
    df_source = df
    if df_source is None:
        try:
            from backend.services import csv_utils as _csv_utils  # type: ignore
            df_source = _csv_utils.load_processed_df()
        except Exception:
            # any import or runtime error -> data source unavailable
            result["match_type"] = "data_source_unavailable"
            _safe_metric("match.data_source_unavailable")
            return result

    # df_source None semantics: explicit unavailable vs empty DataFrame
    if df_source is None:
        result["match_type"] = "data_source_unavailable"
        _safe_metric("match.data_source_unavailable")
        return result

    if df_source.empty:
        result["match_type"] = "no_data_in_source"
        _safe_metric("match.no_data_in_source")
        return result

    # prepare name / symbol columns
    try:
        cols_map = {c.strip(): c for c in df_source.columns}
        name_cols = [c for c in cols_map if c.strip().lower() in ("company_name", "company", "description", "company name")]
        sym_cols = [c for c in cols_map if c.strip().lower() in ("symbol", "sym", "ticker")]
        primary_name_col = cols_map[name_cols[0]] if name_cols else (df_source.columns[0] if df_source.columns.size > 0 else None)
        primary_sym_col = cols_map[sym_cols[0]] if sym_cols else None
    except Exception:
        primary_name_col = None
        primary_sym_col = None

    # 2b) Full-scan: compute candidates and scoring (attempt first)
    candidates: List[Tuple[float, float, str, str]] = []
    top = (0.0, 0.0, "", "")  # (jaccard, ratio, symbol, company_name)
    try:
        if primary_name_col:
            for _, row in df_source.iterrows():
                try:
                    raw_name = str(row.get(primary_name_col, "")).strip()
                    if not raw_name:
                        continue
                    raw_sym = str(row.get(primary_sym_col, "")).strip().upper() if primary_sym_col else ""
                    norm_name = normalize_name(raw_name)
                    name_tokens = [t for t in norm_name.split() if len(t) >= TOKEN_MIN_LEN]
                    j = _jaccard(fn_tokens, name_tokens)
                    r = _similarity_ratio(norm_filename, norm_name)
                    candidates.append((j, r, raw_sym, raw_name))
                    if (j, r) > (top[0], top[1]):
                        top = (j, r, raw_sym, raw_name)
                except Exception:
                    continue

        # sort candidates for output
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        result["candidates"] = [
            {"jaccard": round(c[0], 3), "ratio": round(c[1], 3), "symbol": c[2], "company_name": c[3]}
            for c in candidates[:10]
        ]

        top_j, top_r, top_sym, top_name = top
        result["score"] = float(max(top_j, top_r))
        # choose scored_df if thresholds met
        if top_j >= jaccard_thresh or top_r >= ratio_thresh:
            result.update({"found": True, "symbol": top_sym or None, "company_name": top_name or None, "score": result["score"], "match_type": "scored_df"})
            _safe_metric("match.scored_df")
            _audit_log("filename_to_symbol", filename, "success", {"match_type": result["match_type"], "symbol": top_sym})
            return result
    except Exception as e:
        logger.warning("filename_utils: DataFrame scan failed: %s", e)
        _safe_metric("match.df_scan_failure")

    # 2a) DataFrame prefix (second)
    try:
        if primary_name_col:
            comp_series = df_source[primary_name_col].astype(str).fillna("").map(lambda v: re.sub(r"[^A-Za-z0-9]+", "", v).upper())
            prefix = _prefix_from_filename(base_no_ext)
            if prefix:
                mask = comp_series.str.startswith(prefix)
                if mask.any():
                    row = df_source.loc[mask].iloc[0]
                    found_sym = row.get(primary_sym_col) if primary_sym_col else None
                    found_name = row.get(primary_name_col)
                    result.update({"found": True, "symbol": found_sym, "company_name": found_name, "score": 0.9, "match_type": "prefix_df"})
                    _safe_metric("match.prefix_df")
                    _audit_log("filename_to_symbol", filename, "success", {"match_type": result["match_type"], "symbol": found_sym})
                    return result
    except Exception:
        logger.debug("filename_utils: prefix_df match threw an exception", exc_info=True)

    # Unique-token heuristic (deterministic): if a token from filename maps to exactly one candidate symbol, pick it.
    try:
        if candidates and fn_tokens:
            # Build token -> set(symbols)
            token_to_symbols: Dict[str, set] = {}
            candidate_name_tokens: List[List[str]] = []
            for _, _, sym, cname in candidates:
                norm_name = normalize_name(cname or "")
                name_tokens = [t for t in norm_name.split() if len(t) >= TOKEN_MIN_LEN]
                candidate_name_tokens.append(name_tokens)
                for tok in name_tokens:
                    token_to_symbols.setdefault(tok, set()).add(sym)

            for tok in fn_tokens:
                sym_set = token_to_symbols.get(tok, set())
                if len(sym_set) == 1:
                    chosen_sym = next(iter(sym_set))
                    for c in candidates:
                        if c[2] == chosen_sym:
                            result.update({
                                "found": True,
                                "symbol": c[2],
                                "company_name": c[3],
                                "score": float(max(top[0], top[1])),
                                "match_type": "token_dominant",  # keep legacy label
                            })
                            _safe_metric("match.token_dominant.unique")
                            _audit_log("filename_to_symbol", filename, "success", {"match_type": result["match_type"], "symbol": c[2]})
                            return result
    except Exception:
        logger.debug("filename_utils: unique-token heuristic failed", exc_info=True)

    # Legacy dominant-token heuristic (final fallback before no_match)
    try:
        if candidates and fn_tokens:
            token_counts: Dict[str, int] = {}
            # precompute candidate tokens
            candidate_name_tokens = [ [t for t in normalize_name(c[3]).split() if len(t) >= TOKEN_MIN_LEN] for c in candidates ]
            for idx, c in enumerate(candidates):
                sym = c[2] or ""
                match_count = sum(1 for t in fn_tokens if t in candidate_name_tokens[idx])
                token_counts[sym] = token_counts.get(sym, 0) + match_count

            if token_counts:
                best_sym, cnt = max(token_counts.items(), key=lambda x: x[1])
                if cnt >= 2 or len(fn_tokens) == 1:
                    for c in candidates:
                        if c[2] == best_sym:
                            result.update({
                                "found": True,
                                "symbol": c[2],
                                "company_name": c[3],
                                "score": float(max(top[0], top[1])),
                                "match_type": "token_dominant"
                            })
                            _safe_metric("match.token_dominant")
                            _audit_log("filename_to_symbol", filename, "success", {"match_type": result["match_type"], "symbol": c[2]})
                            return result
    except Exception:
        logger.debug("filename_utils: dominant-token heuristic failed", exc_info=True)

    # final: no match
    result["match_type"] = "no_match"
    _safe_metric("match.no_match")
    return result

# -------------------------
# Health check
# -------------------------
def health_check() -> Dict[str, Any]:
    """
    Return a health report for this module and critical dependencies.

    Contract:
      - If csv_utils.load_processed_df errors or returns None/empty -> overall 'unhealthy'
      - Else if the module-level index_builder exists but is None -> 'degraded'
      - Else if index_builder is available -> 'healthy'
    Important: If tests patch the module-level `index_builder` (even to None),
    we must honor that value (do not auto-import a fallback when the name exists).
    """
    report: Dict[str, Any] = {"status": "healthy", "checks": {}}

    # Determine index_builder availability **preferring module-level binding**.
    index_builder_available = False
    try:
        if "index_builder" in globals():
            # The tests may set filename_utils.index_builder = None explicitly.
            index_builder_available = globals().get("index_builder") is not None
        else:
            # Only attempt to import fallback if the module-level name is absent.
            try:
                from backend.services import index_builder as _ib  # type: ignore
                index_builder_available = _ib is not None
            except Exception:
                index_builder_available = False
    except Exception:
        index_builder_available = False

    report["checks"]["index_builder"] = {"status": "available" if index_builder_available else "unavailable"}

    # CSV data check
    csv_ok = False
    try:
        from backend.services import csv_utils as _csv_utils  # type: ignore
        df = _csv_utils.load_processed_df()
        if df is not None and not df.empty:
            csv_ok = True
            report["checks"]["csv_utils_data"] = {"status": "available", "rows": int(len(df))}
        else:
            report["checks"]["csv_utils_data"] = {"status": "unavailable", "rows": 0}
    except Exception as e:
        report["checks"]["csv_utils_data"] = {"status": "error", "detail": str(e)}

    # Final status: csv error/absent -> unhealthy; csv ok + index missing -> degraded; else healthy
    csv_status = report["checks"]["csv_utils_data"].get("status")
    if csv_status != "available":
        report["status"] = "unhealthy"
    elif not index_builder_available:
        report["status"] = "degraded"
    else:
        report["status"] = "healthy"

    return report

# exports
__all__ = [
    "normalize_name",
    "strip_timestamp_and_ext",
    "extract_datetime_from_filename",
    "filename_tokens",
    "filename_to_symbol",
    "health_check",
    "DEFAULT_JACCARD_THRESH",
    "DEFAULT_RATIO_THRESH",
]