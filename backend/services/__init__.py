"""
Lightweight services package initializer.

Dual-Mode Support:
- Core Mode (default): Only core API modules are lazy-accessible.
- Full Mode: Set SERVICES_FULL_MODE=1 to enable lazy access to all submodules (for dev/debug).
"""

from __future__ import annotations
import importlib
import os
import types
from pathlib import Path
from typing import Optional, Dict

__all__ = ["load_env", "csv_utils", "image_utils", "pdf_processor", "index_builder", "llm_utils", "sentiment_utils"]

# --- Configuration ---
FULL_MODE = os.getenv("SERVICES_FULL_MODE", "0").strip().lower() in ("1", "true", "yes")
_core_allowed = ("csv_utils", "image_utils", "pdf_processor", "index_builder", "llm_utils", "sentiment_utils")
_dev_only = ("image_processor", "csv_processor", "filename_utils", "pdf_watcher", "update_logo_and_patch_json")

# --- Type hints for IDEs / linters ---
# Note: Actual values are set lazily via __getattr__
csv_utils: Optional[types.ModuleType]
image_utils: Optional[types.ModuleType]
pdf_processor: Optional[types.ModuleType]
index_builder: Optional[types.ModuleType]
llm_utils: Optional[types.ModuleType]
sentiment_utils: Optional[types.ModuleType]

_import_cache: Dict[str, Optional[types.ModuleType]] = {}

def load_env(dotenv_path: Optional[Path] = None) -> None:
    """Explicit .env loader; call manually from app startup (e.g. backend/main.py)."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    dotenv_path = dotenv_path or Path(__file__).resolve().parents[2] / ".env"
    try:
        if dotenv_path.exists():
            load_dotenv(dotenv_path=str(dotenv_path))
    except Exception:
        pass  # silent; production should set env externally


def _lazy_import(name: str) -> Optional[types.ModuleType]:
    """Attempt lazy import via several fallback paths."""
    if name in _import_cache:
        return _import_cache[name]

    candidates = [
        f"{__name__}.{name}",
        f"services.{name}",
        name,
    ]

    mod = None
    for candidate in candidates:
        try:
            mod = importlib.import_module(candidate)
            break
        except Exception:
            continue

    _import_cache[name] = mod
    globals()[name] = mod
    return mod


def __getattr__(attr: str) -> Optional[types.ModuleType]:
    """
    Lazy attribute resolver.

    - In core mode, only core modules are permitted.
    - In full mode, all modules are accessible.
    - Raises AttributeError if a *core* module fails to import.
    """
    allowed = _core_allowed + _dev_only if FULL_MODE else _core_allowed

    if attr not in allowed:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{attr}' "
            "(core mode; set SERVICES_FULL_MODE=1 for full access)"
        )

    mod = _lazy_import(attr)
    if attr in _core_allowed and mod is None:
        raise AttributeError(
            f"core module '{attr}' could not be imported (check for missing deps or syntax errors)"
        )
    return mod


def __dir__():
    """Dynamic autocomplete for IDEs (core + dev depending on mode)."""
    base = set(globals().keys())
    allowed = _core_allowed + _dev_only if FULL_MODE else _core_allowed
    base.update(allowed)
    return sorted(base)