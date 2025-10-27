# backend/tests/test_filename_utils.py
"""
Diamond Standard: The definitive, deterministic, and robust unit test suite for
backend.services.filename_utils.

This suite provides a verifiable guarantee of the module's correctness by covering:
- All public helper functions with parametrized edge cases.
- All matching strategies in filename_to_symbol with precise, deterministic assertions.
- All critical failure and fallback modes, including dependency exceptions.
- The health_check function's contractual output for all possible states.
"""
from __future__ import annotations

import pytest
import pandas as pd
from types import SimpleNamespace

# Import the implementation under test using the canonical project path
from backend.services import filename_utils as fn_utils

# --- Fixtures ---

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Provides a consistent, sample DataFrame for deterministic matching tests."""
    data = {
        "symbol": ["HDFCBANK", "AXISBANK", "RELIANCE"],
        "company_name": ["HDFC Bank Limited", "Axis Bank Co", "Reliance Industries Pvt Ltd"],
    }
    return pd.DataFrame(data)

# --- Parametrized Tests for Helper Functions ---

@pytest.mark.parametrize("input_str, expected", [
    ("HDFC Bank Ltd_04-10-2025.pdf", "hdfc bank limited"),
    ("Axis Bank & Co.", "axis bank and company"),
    ("Reliance_Industries-Pvt-Ltd", "reliance industries private limited"),
    ("", ""),
])
def test_normalize_name(input_str: str, expected: str):
    assert fn_utils.normalize_name(input_str) == expected

@pytest.mark.parametrize("fname, expected", [
    ("HDFC Bank Ltd_04-10-2025 09_34_18.pdf", "HDFC Bank Ltd"),
    ("NoTimestampHere.txt", "NoTimestampHere"),
])
def test_strip_timestamp_and_ext(fname: str, expected: str):
    assert fn_utils.strip_timestamp_and_ext(fname) == expected

# --- filename_to_symbol: Deterministic Matching and Validation Tests ---

def test_filename_to_symbol_validation_failed():
    """Tests that input validation returns 'validation_failed' deterministically."""
    assert fn_utils.filename_to_symbol("a" * 256)["match_type"] == "validation_failed"
    assert fn_utils.filename_to_symbol("file_with_@#$.pdf")["match_type"] == "validation_failed"
    assert fn_utils.filename_to_symbol("")["match_type"] == "validation_failed"

def test_filename_to_symbol_index_exact_match(mocker):
    """Tests a successful match via the index_builder's 'exact' strategy."""
    mock_index = SimpleNamespace(
        get_symbol_by_exact=lambda n: "HDFCBANK",
        get_company_by_symbol=lambda s: {"symbol": "HDFCBANK", "company_name": "HDFC Bank Limited"}
    )
    mocker.patch('backend.services.filename_utils.index_builder', mock_index, create=True)
    
    result = fn_utils.filename_to_symbol("HDFC Bank_clean.pdf")
    assert result["match_type"] == "exact"
    assert result["symbol"] == "HDFCBANK"
    assert result["found"] is True

def test_filename_to_symbol_df_prefix_match(sample_df, mocker):
    """Tests a successful match via the DataFrame 'prefix_df' strategy."""
    mocker.patch('backend.services.filename_utils.index_builder', None, create=True)
    
    result = fn_utils.filename_to_symbol("HDFC report.pdf", df=sample_df)
    assert result["match_type"] == "prefix_df"
    assert result["symbol"] == "HDFCBANK"
    assert result["found"] is True

def test_filename_to_symbol_df_scored_match(sample_df, mocker):
    """Tests a successful match via the DataFrame 'scored_df' strategy."""
    mocker.patch('backend.services.filename_utils.index_builder', None, create=True)

    result = fn_utils.filename_to_symbol("Reliance Industries Pvt Ltd_report.pdf", df=sample_df)
    assert result["match_type"] == "scored_df"
    assert result["symbol"] == "RELIANCE"
    assert result["found"] is True
    assert result["score"] >= fn_utils.DEFAULT_RATIO_THRESH

def test_token_dominant_is_deterministic(mocker):
    """Tests the token_dominant heuristic with a crafted DataFrame to force the outcome."""
    df = pd.DataFrame([
        {"symbol": "FOO", "company_name": "Foo Bar Company"},
        {"symbol": "BAR", "company_name": "Bar Baz Corp"}
    ])
    mocker.patch('backend.services.filename_utils.index_builder', None, create=True)
    
    result = fn_utils.filename_to_symbol("Some Baz Announcement.pdf", df=df)
    
    assert result["found"] is True
    assert result["symbol"] == "BAR"
    assert result["match_type"] == "token_dominant"

# --- Failure and Fallback Mode Tests ---

def test_index_raises_exception_falls_back_to_df_deterministically(mocker, sample_df):
    """Tests that if an index method raises an exception, the logic gracefully falls back to a specific DataFrame match."""
    class BrokenIndex:
        def get_symbol_by_exact(self, name): raise ValueError("Index connection failed")

    mocker.patch('backend.services.filename_utils.index_builder', BrokenIndex(), create=True)
    mocker.patch('backend.services.csv_utils.load_processed_df', return_value=sample_df)
    
    result = fn_utils.filename_to_symbol("HDFC Bank Limited_report.pdf")
    
    assert result["found"] is True
    assert result["symbol"] == "HDFCBANK"
    assert result["match_type"] == "scored_df"

def test_csv_utils_returns_none_handled_gracefully(mocker):
    """Tests the specific 'data_source_unavailable' outcome when csv_utils returns None."""
    mocker.patch('backend.services.filename_utils.index_builder', None, create=True)
    mocker.patch('backend.services.csv_utils.load_processed_df', return_value=None)
    
    result = fn_utils.filename_to_symbol("Some Unknown Company.pdf")
    assert result["found"] is False
    assert result["match_type"] == "data_source_unavailable"

def test_csv_utils_raises_exception_handled_gracefully(mocker):
    """Tests that an exception from csv_utils is handled and returns the correct status."""
    mocker.patch('backend.services.filename_utils.index_builder', None, create=True)
    mocker.patch('backend.services.csv_utils.load_processed_df', side_effect=IOError("S3 connection failed"))
    
    result = fn_utils.filename_to_symbol("Any Co.pdf")
    assert result["found"] is False
    assert result["match_type"] == "data_source_unavailable"

# --- Health Check Contract Tests (Robust & Deterministic) ---

def test_health_check_all_healthy(mocker):
    """Tests the health check for the exact contract of a healthy state, while tolerating extensions."""
    mocker.patch('backend.services.filename_utils.index_builder', object(), create=True)
    mocker.patch('backend.services.csv_utils.load_processed_df', return_value=pd.DataFrame({"symbol": ["X"]}))
    
    health = fn_utils.health_check()
    
    assert health["status"] == "healthy"
    checks = health["checks"]
    assert checks["index_builder"]["status"] == "available"
    assert checks["csv_utils_data"]["status"] == "available"
    assert "rows" in checks["csv_utils_data"]

def test_health_check_degraded_when_index_is_missing(mocker):
    """Tests the health check for the exact contract of a degraded state."""
    mocker.patch('backend.services.filename_utils.index_builder', None, create=True)
    mocker.patch('backend.services.csv_utils.load_processed_df', return_value=pd.DataFrame({"symbol": ["X"]}))

    health = fn_utils.health_check()
    
    assert health["status"] == "degraded"
    checks = health["checks"]
    assert checks["index_builder"]["status"] == "unavailable"
    assert checks["csv_utils_data"]["status"] == "available"

def test_health_check_unhealthy_when_data_source_fails(mocker):
    """Tests the health check for the exact contract of an unhealthy state."""
    mocker.patch('backend.services.filename_utils.index_builder', object(), create=True)
    mocker.patch('backend.services.csv_utils.load_processed_df', side_effect=RuntimeError("S3 is down"))

    health = fn_utils.health_check()

    assert health["status"] == "unhealthy"
    checks = health["checks"]
    assert checks["index_builder"]["status"] == "available"
    assert checks["csv_utils_data"]["status"] == "error"
    assert "S3 is down" in checks["csv_utils_data"]["detail"]