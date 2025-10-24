#!/usr/bin/env python3
"""
services/index_builder.py

In-memory index builder over processed EOD CSV (via csv_utils.load_processed_df()).

Responsibilities:
 - Build indexes from the processed DataFrame (symbol -> row, compact company -> symbol,
   token -> set(symbols), normalized company -> symbol).
 - Provide safe, thread-safe lookup functions:
     * build_index(force=False)
     * refresh_index()
     * index_status()
     * get_symbol_by_exact(sym)
     * get_symbol_by_token(token)
     * fuzzy_lookup_symbol(query, min_ratio=0.70)
     * token_overlap_search(filename_tokens, min_score=0.60)
     * get_company_by_symbol(sym)
 - Defensive: tolerates missing csv_utils or missing processed CSVs.
"""

from __future__ import annotations
import threading
import time
import re
from typing import Optional, Dict, Any, Set, List, Tuple
from collections import defaultdict
from difflib import SequenceMatcher

# Defensive import of csv_utils (caller must ensure backend/ on PYTHONPATH)
try:
    from services import csv_utils  # type: ignore
except Exception:
    try:
        import csv_utils  # fallback local import, if module path differs
    except Exception:
        csv_utils = None  # type: ignore

_lock = threading.RLock()
_built_at: Optional[float] = None

# Primary indexes (initialized empty)
_symbol_map: Dict[str, Dict[str, Any]] = {}           # SYMBOL -> row dict (original row dict)
_compact_map: Dict[str, str] = {}                    # compact(company_name) -> SYMBOL
_token_index: Dict[str, Set[str]] = defaultdict(set) # token -> set(SYMBOL)
_company_norm_map: Dict[str, str] = {}               # normalized company name -> SYMBOL

# Tunables
DEFAULT_MIN_FUZZY_RATIO = 0.70
TOKEN_MIN_LENGTH = 2

# Helpers ---------------------------------------------------------------------
_non_alnum_re = re.compile(r"[^A-Za-z0-9]+")
_spaces_re = re.compile(r"\s+")

def _compact(s: Optional[str]) -> str:
    """Compact string by stripping non-alnum and uppercasing (no spaces)."""
    if not s:
        return ""
    return re.sub(r"[^A-Za-z0-9]+", "", str(s)).upper()

def _tokens(s: Optional[str]) -> List[str]:
    """Return list of tokens (uppercase), drop short tokens and numeric-only tokens."""
    if not s:
        return []
    toks = [t.strip().upper() for t in re.split(r"[^A-Za-z0-9]+", str(s)) if t.strip()]
    out = []
    for t in toks:
        if len(t) < TOKEN_MIN_LENGTH:
            continue
        if t.isdigit():
            continue
        out.append(t)
    return out

def _normalize_company_name(s: Optional[str]) -> str:
    """Lowercased, collapsed-space company name suitable as normalization key."""
    if not s:
        return ""
    s2 = _spaces_re.sub(" ", str(s)).strip()
    return s2.lower()

# Prefix-match tuning (used for filename-like prefix prioritization)
PREFIX_MIN = 3
PREFIX_MAX = 10

def _prefix_from_filename_fn(name: Optional[str]) -> str:
    """Compact string by removing non-alnum characters and returning the first PREFIX_MAX chars (uppercased)."""
    if not name:
        return ""
    compacted = re.sub(r"[^A-Za-z0-9]+", "", str(name)).upper()
    return compacted[:PREFIX_MAX]

def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

# Core builder ---------------------------------------------------------------
def build_index(force: bool = False) -> None:
    """
    Build the in-memory index from csv_utils.load_processed_df().
    Safe to call multiple times; uses a lock. If csv_utils is missing or processed DF
    is unavailable, builds empty indexes.
    """
    global _symbol_map, _compact_map, _token_index, _company_norm_map, _built_at

    with _lock:
        # If csv_utils not available, clear indexes and return quickly
        if csv_utils is None or not hasattr(csv_utils, "load_processed_df"):
            _symbol_map = {}
            _compact_map = {}
            _token_index = defaultdict(set)
            _company_norm_map = {}
            _built_at = time.time()
            return

        # Avoid rebuilding repeatedly unless forced
        if _built_at and not force:
            return

        try:
            df = csv_utils.load_processed_df()
            if df is None:
                _symbol_map = {}
                _compact_map = {}
                _token_index = defaultdict(set)
                _company_norm_map = {}
                _built_at = time.time()
                return

            # initialize empty structures
            _symbol_map = {}
            _compact_map = {}
            _token_index = defaultdict(set)
            _company_norm_map = {}

            # detect likely symbol/company columns (case-insensitive heuristics)
            cols = [c for c in getattr(df, "columns", [])] if hasattr(df, "columns") else []
            symbol_cols = [c for c in cols if c.strip().lower() in ("symbol", "sym", "ticker")]
            comp_cols = [c for c in cols if c.strip().lower() in ("company_name", "company", "description", "company name")]
            if not comp_cols and cols:
                comp_cols = [cols[0]]

            # iterate rows safely (supports pandas DataFrame or list-of-dicts)
            rows_iter = []
            if hasattr(df, "iterrows"):
                for _, r in df.iterrows():
                    try:
                        rows_iter.append(r.to_dict() if hasattr(r, "to_dict") else dict(r))
                    except Exception:
                        # fallback: attempt dict conversion
                        rows_iter.append(dict(r))
            elif isinstance(df, (list, tuple)):
                rows_iter = list(df)
            else:
                try:
                    rows_iter = list(df)
                except Exception:
                    rows_iter = []

            for row in rows_iter:
                # obtain symbol
                sym = None
                for sc in symbol_cols:
                    if sc in row and row.get(sc):
                        sym = str(row.get(sc)).strip()
                        break
                # fallback: find fields that look like symbol
                if not sym:
                    for k in getattr(row, "keys", lambda: [])():
                        if k and str(k).strip().lower() in ("symbol", "sym", "ticker"):
                            sym = str(row.get(k)).strip()
                            break
                if not sym:
                    continue
                sym_up = str(sym).upper()

                # store canonical row (keep raw values)
                try:
                    row_dict = dict(row)
                except Exception:
                    row_dict = {str(k): row.get(k) for k in getattr(row, "keys", lambda: [])()}

                _symbol_map[sym_up] = row_dict

                # canonical company name
                comp_val = None
                for cc in comp_cols:
                    if cc in row and row.get(cc):
                        comp_val = str(row.get(cc)).strip()
                        break
                if not comp_val:
                    # fallback: search common keys
                    for k in getattr(row, "keys", lambda: [])():
                        if k and str(k).strip().lower() in ("company_name", "company", "description", "company name"):
                            comp_val = str(row.get(k)).strip()
                            break

                comp_compact = _compact(comp_val)
                if comp_compact:
                    _compact_map[comp_compact] = sym_up

                comp_norm = _normalize_company_name(comp_val)
                if comp_norm:
                    _company_norm_map[comp_norm] = sym_up

                # token index (company tokens + symbol tokens)
                for tok in set(_tokens(comp_val) + _tokens(sym_up)):
                    _token_index[tok].add(sym_up)

            _built_at = time.time()

        except Exception:
            # On failure, make sure structures exist and set timestamp
            _symbol_map = {}
            _compact_map = {}
            _token_index = defaultdict(set)
            _company_norm_map = {}
            _built_at = time.time()

# Query APIs -----------------------------------------------------------------
def get_symbol_by_exact(sym: str) -> Optional[str]:
    """Return canonical symbol (UPPER) for exact match or compact company match, else None."""
    if not sym:
        return None
    s = str(sym).strip().upper()
    with _lock:
        if s in _symbol_map:
            return s
        # accept compact company name
        c = _compact(s)
        if c and c in _compact_map:
            return _compact_map[c]
    return None

def get_symbol_by_token(token: str) -> Optional[str]:
    """
    If a token maps to one or more symbols, return a best candidate:
     - prefer exact token == symbol
     - else pick shortest symbol (heuristic)
     - else prefer symbols with first 6-7 characters matching token prefix
    """
    if not token:
        return None
    t = token.strip().upper()
    with _lock:
        candidates = list(_token_index.get(t, set()))
        if not candidates:
            return None
        if t in candidates:
            return t
        # prefix-based prioritization: filter candidates whose first 6-7 chars match token prefix
        prefix_len = min(7, len(t))
        prefix = t[:prefix_len]
        prefix_matches = [c for c in candidates if c[:prefix_len].upper() == prefix]
        if prefix_matches:
            # among prefix matches, pick shortest symbol then lexicographic
            prefix_matches_sorted = sorted(prefix_matches, key=lambda x: (len(x), x))
            return prefix_matches_sorted[0]
        # heuristic: prefer shortest symbol name then lexicographic
        candidates_sorted = sorted(candidates, key=lambda x: (len(x), x))
        return candidates_sorted[0]

def fuzzy_lookup_symbol(query: str, min_ratio: float = DEFAULT_MIN_FUZZY_RATIO) -> Optional[str]:
    """
    Conservative fuzzy lookup:
      - check compact-company direct match
      - gather token candidate symbols and score by token overlap + similarity
      - final scan over compacts/symbols using SequenceMatcher
      - prefer candidates with first 6-7 chars matching query prefix (case-insensitive)
    Returns canonical SYMBOL (UPPER) or None.
    """
    if not query:
        return None
    q = str(query).strip()
    q_compact = _compact(q)
    q_norm = _normalize_company_name(q)

    with _lock:
        # direct compact match
        if q_compact and q_compact in _compact_map:
            return _compact_map[q_compact]

        # prefix-based prioritization: try to apply filename_utils logic
        try:
            q_prefix = _prefix_from_filename_fn(q)
        except Exception:
            q_prefix = ""
        prefix_len = len(q_prefix)
        q_toks = _tokens(q)
        if q_toks:
            cand_symbols: Set[str] = set()
            for tok in q_toks:
                cand_symbols.update(_token_index.get(tok, set()))
            # if only one candidate, return quickly
            if len(cand_symbols) == 1:
                return next(iter(cand_symbols))

            # If we have a meaningful compact prefix, prefer candidates matching that prefix
            if q_prefix and len(q_prefix) >= PREFIX_MIN:
                prefix_matches = [sym for sym in cand_symbols if str(sym).upper().startswith(q_prefix)]
                candidates_to_score = prefix_matches if prefix_matches else list(cand_symbols)
            else:
                candidates_to_score = list(cand_symbols)

            # score candidates
            best_sym = None
            best_score = 0.0
            for sym in candidates_to_score:
                # compute tokens for sym + its company name
                sym_tokens = set(_tokens(sym) + _tokens(_symbol_map.get(sym, {}).get("company_name", "") if sym in _symbol_map else []))
                common = len(set(q_toks).intersection(sym_tokens))
                denom = max(1, (len(sym_tokens) + len(q_toks)) / 2)
                token_coverage = common / denom
                sim = _similarity(q_compact, _compact(sym))
                # weighted scoring: token coverage heavier
                score = token_coverage * 0.7 + sim * 0.3
                if score > best_score:
                    best_score = score
                    best_sym = sym
            if best_score >= min_ratio:
                return best_sym

        # final: scan all compacts and symbols
        best_sym = None
        best_score = 0.0
        for sym, row in _symbol_map.items():
            sc = _similarity(q_compact, _compact(sym))
            if sc > best_score:
                best_score = sc
                best_sym = sym
        for comp_compact, sym in _compact_map.items():
            sc = _similarity(q_compact, comp_compact)
            if sc > best_score:
                best_score = sc
                best_sym = sym
        if best_score >= min_ratio:
            return best_sym

    return None

def token_overlap_search(filename_tokens: List[str], min_score: float = 0.60) -> Optional[Tuple[str, float]]:
    """
    Score token overlap between filename tokens and indexed company tokens.
    Returns (symbol, score) if any candidate >= min_score else None.
    """
    if not filename_tokens:
        return None
    toks = [t.upper() for t in filename_tokens if t and len(t) >= TOKEN_MIN_LENGTH and not t.isdigit()]
    if not toks:
        return None
    with _lock:
        cand: Set[str] = set()
        for t in toks:
            cand.update(_token_index.get(t, set()))
        if not cand:
            return None
        file_set = set(toks)
        best = (None, 0.0)
        for sym in cand:
            row = _symbol_map.get(sym, {})
            comp_name = row.get("company_name") if isinstance(row, dict) else None
            comp_tokens = set(_tokens(comp_name) + _tokens(sym))
            inter = file_set.intersection(comp_tokens)
            denom = max(1, (len(file_set) + len(comp_tokens)) / 2)
            score = len(inter) / denom
            if score > best[1]:
                best = (sym, score)
        if best[0] and best[1] >= min_score:
            return best[0], best[1]
    return None

def get_company_by_symbol(sym: str) -> Optional[Dict[str, Any]]:
    """Return the raw row/dict for canonical symbol (or None)."""
    if not sym:
        return None
    s = str(sym).strip().upper()
    with _lock:
        return _symbol_map.get(s)

def refresh_index() -> None:
    """Public: force rebuild of the in-memory index."""
    build_index(force=True)

def index_status() -> Dict[str, Any]:
    """Return diagnostics about the index (counts and timestamp)."""
    with _lock:
        return {
            "built_at": _built_at,
            "symbols": len(_symbol_map),
            "compact_mappings": len(_compact_map),
            "token_index_size": len(_token_index),
            "company_norms": len(_company_norm_map),
        }

# Eager build on import (best-effort)
try:
    build_index()
except Exception:
    # be resilient: callers can explicitly call refresh_index()
    pass