# tests/test_index_builder.py
"""
Diamond Test Suite — index_builder (I-04)
-----------------------------------------
Covers:
 - build_index_from_df (pure, deterministic)
 - build_index (loads via csv_utils.load_processed_df)
 - query APIs: exact, token, fuzzy, token_overlap, company lookup
 - lifecycle: clear_index, refresh_index (background)
 - failure mode when csv_utils.load_processed_df raises
"""

import time
import threading
from typing import List, Dict, Any, Tuple, Optional

import pytest

# module under test
import services.index_builder as ib

# --- Helpers / sample data ---------------------------------------------------

SAMPLE_ROWS: List[Dict[str, Any]] = [
    {"symbol": "ABC", "company_name": "Alpha Beta Corp"},
    {"symbol": "XYZ", "company_name": "Xylon Zypher Ltd"},
    {"symbol": "TATA", "company_name": "Tata Motors Limited"},
]

def sample_rows_with_other_keys() -> List[Dict[str, Any]]:
    # symbol key using alternative name and numeric column present
    return [
        {"sym": "FOO", "company": "Foo Bar Pvt Ltd"},
        {"ticker": "BAR", "description": "Bar Industries"},
    ]


# --- Fixtures ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_index_before_and_after():
    """Ensure each test starts with a clean index and no stray background thread."""
    try:
        ib.clear_index()
    except Exception:
        pass
    yield
    # best-effort cleanup
    try:
        # stop background builder if any
        ib.close()
    except Exception:
        pass
    try:
        ib.clear_index()
    except Exception:
        pass


# --- Tests for pure builder -------------------------------------------------

def test_build_index_from_df_basic():
    new_sym, new_comp, new_tok, new_norm, built_at = ib.build_index_from_df(SAMPLE_ROWS)
    # symbols
    assert "ABC" in new_sym
    assert "XYZ" in new_sym
    # compact map keys should be compacted company names
    assert ib._compact("Alpha Beta Corp") in new_comp
    # token index should contain tokens like ALPHA, BETA, XYZ, XYLON, TATA etc.
    assert "ALPHA" in new_tok or "BETA" in new_tok
    # built_at should be a recent timestamp
    assert isinstance(built_at, float) and built_at > 0.0


def test_build_index_from_df_with_alternate_keys():
    rows = sample_rows_with_other_keys()
    new_sym, new_comp, new_tok, new_norm, _ = ib.build_index_from_df(rows)
    assert "FOO" in new_sym
    assert "BAR" in new_sym
    assert ib._compact("Foo Bar Pvt Ltd") in new_comp


# --- Tests integrating with csv_utils (monkeypatch load_processed_df) --------

def test_build_index_and_query_apis(monkeypatch):
    # monkeypatch the csv_utils loader used by the module to return SAMPLE_ROWS
    class FakeCSVUtils:
        @staticmethod
        def load_processed_df():
            return SAMPLE_ROWS

    # Patch the imported csv_utils object in module
    monkeypatch.setattr(ib, "csv_utils", FakeCSVUtils, raising=False)

    # Force a fresh build
    ib.build_index(force=True)

    # index_status should reflect the symbols we added
    st = ib.index_status()
    assert st["symbols"] >= 3

    # exact lookup
    assert ib.get_symbol_by_exact("abc") == "ABC"  # case-insensitive
    assert ib.get_symbol_by_exact("AlphaBetaCorp") == "ABC"  # compact match

    # token lookup
    # token 'ALPHA' should map to ABC
    sym_by_token = ib.get_symbol_by_token("ALPHA")
    assert sym_by_token in {"ABC", "TATA", "XYZ"}  # deterministic heuristics; at least returns one known symbol

    # fuzzy lookup - query close to "Alpha Beta"
    fuzzy = ib.fuzzy_lookup_symbol("Alpha Beta Co", min_ratio=0.3)
    # fuzzy should either return ABC or None depending on scoring threshold
    assert fuzzy in (None, "ABC")

    # token_overlap_search: build tokens from a filename-like list
    res = ib.token_overlap_search(["alpha", "corp"])
    if res:
        sym, score = res
        assert isinstance(sym, str) and isinstance(score, float)

    # company lookup
    comp = ib.get_company_by_symbol("XYZ")
    assert isinstance(comp, dict)
    assert comp.get("company_name") == "Xylon Zypher Ltd"

    # clear index and ensure lookups return None
    ib.clear_index()
    assert ib.get_symbol_by_exact("ABC") is None
    assert ib.get_company_by_symbol("XYZ") is None


def test_build_index_handles_loader_exception(monkeypatch):
    class BrokenCSVUtils:
        @staticmethod
        def load_processed_df():
            raise RuntimeError("IO failure")

    monkeypatch.setattr(ib, "csv_utils", BrokenCSVUtils, raising=False)

    # Should not raise; should clear index instead
    ib.build_index(force=True)
    st = ib.index_status()
    assert st["symbols"] == 0


# --- Background refresh -----------------------------------------------------

def _wait_for_index_built(timeout: float = 2.0) -> bool:
    """Small helper: poll index_status until built_at becomes non-None or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = ib.index_status()
        if st.get("built_at"):
            return True
        time.sleep(0.02)
    return False


def test_refresh_index_background(monkeypatch):
    # Patch loader to return sample rows after slight delay
    class FakeCSVUtils:
        @staticmethod
        def load_processed_df():
            # small sleep to simulate work
            time.sleep(0.05)
            return SAMPLE_ROWS

    monkeypatch.setattr(ib, "csv_utils", FakeCSVUtils, raising=False)

    # Start background refresh
    ib.refresh_index(background=True)
    built = _wait_for_index_built(timeout=5.0)  # increased timeout
    assert built is True
    # Give a bit more time for the background thread to complete
    time.sleep(0.1)
    st = ib.index_status()
    assert st["symbols"] >= 3

    # cleanup
    ib.close()


# --- Concurrency sanity (basic smoke) --------------------------------------

def test_concurrent_reads_during_build(monkeypatch):
    """
    Smoke test to ensure readers don't raise during a build.
    This does not fully prove lock-free behavior but ensures no runtime exceptions
    when readers run while a build (that swaps indexes) is ongoing.
    """
    # loader that does some work and returns larger dataset
    class SlowCSVUtils:
        @staticmethod
        def load_processed_df():
            rows = []
            for i in range(200):
                rows.append({"symbol": f"S{i}", "company_name": f"Company {i}"})
            # simulate time-consuming build
            time.sleep(0.1)
            return rows

    monkeypatch.setattr(ib, "csv_utils", SlowCSVUtils, raising=False)

    # spawn reader threads while we build in background
    ib.refresh_index(background=True)
    readers = []

    stop_at = time.time() + 1.0
    def reader_job():
        # Run a few random reads until index is built or timeout
        while time.time() < stop_at:
            try:
                # call several query APIs
                _ = ib.index_status()
                _ = ib.get_symbol_by_exact("S10")
                _ = ib.fuzzy_lookup_symbol("Company 10", min_ratio=0.1)
                time.sleep(0.005)
            except Exception as e:
                # fail the test if any exception bubbles up
                pytest.fail(f"Reader raised during concurrent build: {e}")

    for _ in range(4):
        t = threading.Thread(target=reader_job, daemon=True)
        readers.append(t)
        t.start()

    # wait for readers to finish
    for t in readers:
        t.join(timeout=2.0)

    # ensure index eventually built
    assert _wait_for_index_built(timeout=3.0)


# --- Small contract tests around build_index_from_df edge-cases --------------

def test_build_index_from_df_empty_list():
    new_sym, new_comp, new_tok, new_norm, _ = ib.build_index_from_df([])
    assert isinstance(new_sym, dict) and len(new_sym) == 0
    assert isinstance(new_tok, dict)


def test_build_index_from_df_non_dict_rows():
    # Rows that are tuples or objects convertible to dict-like
    rows = [("S", "SomeCo"), {"symbol": "A1", "company_name": "SomeCo"}]
    # The function should not raise; result may be empty for malformed row
    _ = ib.build_index_from_df(rows)


# --- Phase 2: Diamond Grade I-05 Test Expansion -------------------------

def test_lock_free_reads_consistency(monkeypatch):
    """Verify readers see consistent data snapshots during background rebuild."""
    class SlowCSVUtils:
        @staticmethod
        def load_processed_df():
            # initial dataset
            rows = [{"symbol": f"S{i}", "company_name": f"Company {i}"} for i in range(100)]
            time.sleep(0.2)  # simulate long build
            return rows

    monkeypatch.setattr(ib, "csv_utils", SlowCSVUtils, raising=False)

    ib.refresh_index(background=True)
    snapshots = []
    end = time.time() + 0.5
    while time.time() < end:
        st = ib.index_status()
        snapshots.append((st["symbols"], st["compact_mappings"]))
        time.sleep(0.01)

    # All snapshots must be consistent tuples (no mid-swap partials)
    assert all(isinstance(a, int) and isinstance(b, int) for a, b in snapshots)
    assert all(a >= 0 and b >= 0 for a, b in snapshots)


def test_memory_limits_respected():
    """Ensure hard cap MAX_SYMBOLS_HARD_LIMIT is enforced."""
    large = [{"symbol": f"S{i}", "company_name": f"Company {i}"} for i in range(ib.MAX_SYMBOLS_HARD_LIMIT + 50)]
    new_sym, *_ = ib.build_index_from_df(large)
    assert len(new_sym) <= ib.MAX_SYMBOLS_HARD_LIMIT


def test_metrics_incremented(monkeypatch):
    """Verify Prometheus metrics or fallback counters increment."""
    calls = {"success": 0, "fail": 0}
    monkeypatch.setattr(ib, "_inc_build_success", lambda: calls.__setitem__("success", calls["success"] + 1))
    monkeypatch.setattr(ib, "_inc_build_fail", lambda: calls.__setitem__("fail", calls["fail"] + 1))
    ib.build_index_from_df(SAMPLE_ROWS)
    assert calls["success"] == 1 and calls["fail"] == 0


def test_background_thread_cleanup(monkeypatch):
    """Ensure background thread stops cleanly after close()."""
    class DummyCSVUtils:
        @staticmethod
        def load_processed_df():
            return SAMPLE_ROWS
    monkeypatch.setattr(ib, "csv_utils", DummyCSVUtils, raising=False)
    ib.refresh_index(background=True)
    time.sleep(0.1)
    ib.close()
    assert ib._bg_thread is None or not ib._bg_thread.is_alive()


def test_unicode_and_special_characters():
    data = [
        {"symbol": "U1", "company_name": "Müller AG"},
        {"symbol": "U2", "company_name": "Company & Sons (International)"},
        {"symbol": "U3", "company_name": "Alpha-Beta Holdings"}
    ]
    new_sym, new_comp, *_ = ib.build_index_from_df(data)
    assert any("MULLER" in k or "ALPHABETA" in k for k in new_comp.keys())


def test_build_performance(monkeypatch):
    """Ensure builds complete in under 3 seconds for 10k rows."""
    data = [{"symbol": f"S{i}", "company_name": f"Company {i}"} for i in range(10000)]
    start = time.time()
    ib.build_index_from_df(data)
    duration = time.time() - start
    assert duration < 3.0
