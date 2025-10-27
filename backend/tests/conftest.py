"""
backend/tests/conftest.py
Test configuration and fixtures for pytest.
"""

import logging
import pytest
from unittest.mock import patch

# Ensure repo root is on sys.path
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services import csv_utils

@pytest.fixture(scope="session", autouse=True)
def configure_test_logging():
    """Configure logging for tests to avoid correlation_id errors."""
    # Set a test correlation_id to prevent KeyError in log formatter
    if hasattr(csv_utils, "correlation_id_var"):
        csv_utils.correlation_id_var.set("test-session-id")

    # Configure logging with a safe formatter
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Patch the logger formatter if it uses correlation_id
    try:
        from backend.services.csv_processor import logger
        for handler in logger.handlers:
            if hasattr(handler, 'formatter') and handler.formatter:
                # Replace any formatter that might use correlation_id
                handler.setFormatter(logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                ))
    except Exception:
        pass

@pytest.fixture(scope="function", autouse=True)
def set_test_correlation_id():
    """Set correlation_id for each test."""
    if hasattr(csv_utils, "correlation_id_var"):
        csv_utils.correlation_id_var.set("test-function-id")