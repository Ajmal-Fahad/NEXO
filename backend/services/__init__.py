"""
Services package for NEXIS-APP backend.
Expose common helpers so callers can do: `from services import csv_utils, image_utils`
"""

from pathlib import Path
from dotenv import load_dotenv

# Auto-load project-level .env (NEXO/.env)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

__all__ = ["csv_utils", "image_utils", "pdf_processor"]

try:
    from . import csv_utils  # type: ignore
except Exception:
    csv_utils = None

try:
    from . import image_utils  # type: ignore
except Exception:
    image_utils = None
