#!/usr/bin/env python3
"""
main.py

Clean, single-app FastAPI bootstrap with:
 - Pydantic v2 settings (pydantic_settings.BaseSettings)
 - Optional .env loading (AUTO_LOAD_ENV)
 - Optional modern lifespan startup (USE_MODERN_STARTUP) that runs warm-up tasks before app accepts requests
 - Centralized logging
 - Package-qualified router imports (container friendly)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic_settings import BaseSettings

# ---------- Minimal early logging so _maybe_load_dotenv can log safely ----------
import logging as _logging
_logging.basicConfig(level=_logging.INFO)
logger = _logging.getLogger("backend.main")

# ---------- Settings ----------
class Settings(BaseSettings):
    ENVIRONMENT: str = "dev"
    ALLOWED_ORIGINS: Optional[str] = None
    ENABLE_RAW_ANNOUNCEMENTS: bool = False
    WARM_INDEX_ON_STARTUP: bool = False
    AUTO_LOAD_ENV: bool = False
    USE_MODERN_STARTUP: bool = False
    BACKEND_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    class Config:
        # Do NOT auto-load .env here; we control loading explicitly with AUTO_LOAD_ENV
        env_file = None
        case_sensitive = False


def _maybe_load_dotenv(settings: Settings) -> None:
    """Load .env if requested and python-dotenv is available."""
    if not settings.AUTO_LOAD_ENV:
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        repo_root = Path(__file__).resolve().parents[1]
        env_path = repo_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=str(env_path))
            logger.info("Loaded .env from %s", env_path)
        else:
            logger.debug(".env requested but not found at %s", env_path)
    except Exception as exc:
        logger.warning("Failed to load .env (ignored): %s", exc)


# Preliminary settings so AUTO_LOAD_ENV can cause a second pass
_pre_settings = Settings()
_maybe_load_dotenv(_pre_settings)

# Final settings (picks up .env if loaded)
settings = Settings()
# reconfigure logging level to the value in settings
logger.setLevel(getattr(_logging, settings.LOG_LEVEL.upper(), _logging.INFO))
logger.info("Settings loaded: ENVIRONMENT=%s, USE_MODERN_STARTUP=%s", settings.ENVIRONMENT, settings.USE_MODERN_STARTUP)

# ---------- Warm index helper ----------
async def _maybe_warm_index() -> None:
    """Run index warm-up in a threadpool if available and requested."""
    if not settings.WARM_INDEX_ON_STARTUP:
        logger.debug("WARM_INDEX_ON_STARTUP is false; skipping warm index")
        return
    try:
        # import lazily so missing optional deps don't break startup
        from backend.services import index_builder  # type: ignore

        loop = asyncio.get_running_loop()
        if hasattr(index_builder, "refresh_index"):
            logger.info("Running index_builder.refresh_index in executor (warm start)")
            await loop.run_in_executor(None, getattr(index_builder, "refresh_index"))
        else:
            logger.info("index_builder has no refresh_index; skipping")
    except Exception as exc:
        # Non-fatal — log and continue
        logger.exception("Index warm-up failed (non-fatal): %s", exc)


# ---------- Metrics collector setup ----------
import threading
_metrics_installed = False
_metrics_lock = threading.Lock()

# Try to import prometheus; be tolerant if it's absent
try:
    from prometheus_client import Counter, make_asgi_app
except Exception:
    Counter = None
    make_asgi_app = None
    logger.warning("prometheus_client not available; metrics features disabled")

# Prometheus metrics (only if available)
if Counter is not None:
    _csv_fallback_counter = Counter(
        "csv_processor_fallback_to_local_total",
        "Count of S3→local CSV fallback events"
    )
else:
    _csv_fallback_counter = None

def prometheus_collector(name: str, value: int):
    """Real Prometheus metrics integration (no-op if prometheus not available)."""
    try:
        if name == "csv_processor.fallback_to_local" and _csv_fallback_counter is not None:
            _csv_fallback_counter.inc(int(value))
    except Exception as exc:
        logger.warning("Metrics collection failed: %s", exc)


def setup_metrics():
    """Thread-safe, idempotent setup for CSV metrics collector."""
    global _metrics_installed
    with _metrics_lock:
        if _metrics_installed:
            logger.debug("Metrics already setup; skipping")
            return
        try:
            # Lazy import to avoid early coupling
            from backend.services.csv_processor import set_metrics_collector
            set_metrics_collector(prometheus_collector)
            _metrics_installed = True
            logger.info("Metrics collector set up for CSV processor")
        except Exception as exc:
            logger.warning("Metrics setup failed (non-fatal): %s", exc)


# ---------- Lifespan (optional modern startup) ----------
lifespan_ctx = None
if settings.USE_MODERN_STARTUP:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await _maybe_warm_index()
        setup_metrics()  # safe to call here during app startup
        yield
    lifespan_ctx = lifespan
    logger.info("Using modern lifespan startup (USE_MODERN_STARTUP=True)")


# ---------- App creation (single app) ----------
show_docs = settings.ENVIRONMENT.lower() != "prod"
app = FastAPI(
    title="NEXO Backend",
    version="0.0.1",
    docs_url="/docs" if show_docs else None,
    redoc_url="/redoc" if show_docs else None,
    openapi_url="/openapi.json" if show_docs else None,
    lifespan=lifespan_ctx,
)

# ---------- CORS ----------
if settings.ALLOWED_ORIGINS:
    origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
else:
    origins = ["*"] if settings.ENVIRONMENT.lower() == "dev" else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
logger.debug("CORS origins: %s", origins)

# ---------- Routers (package-qualified imports) ----------
# We import package-qualified names so `uvicorn --app-dir backend main:app` and container runs both work.
health_router = None
announcements_router = None
raw_announcements_router = None

try:
    from backend.routers import health as health_router  # type: ignore
    logger.debug("Imported backend.routers.health")
except Exception as exc:
    logger.exception("Could not import health router: %s", exc)
    health_router = None

try:
    from backend.routers import announcements as announcements_router  # type: ignore
    logger.debug("Imported backend.routers.announcements")
except Exception as exc:
    logger.exception("Could not import announcements router: %s", exc)
    announcements_router = None

try:
    from backend.routers import raw_announcements as raw_announcements_router  # type: ignore
    logger.debug("Imported backend.routers.raw_announcements")
except Exception:
    # raw_announcements is optional and gated behind setting
    raw_announcements_router = None

# Mount routers that are present
if health_router is not None:
    app.include_router(health_router.router)
if announcements_router is not None:
    app.include_router(announcements_router.router)
if settings.ENABLE_RAW_ANNOUNCEMENTS and raw_announcements_router is not None:
    app.include_router(raw_announcements_router.router)

# ---------- Static files ----------
static_dir = Path("input_data")
if static_dir.exists() and static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info("Mounted /static -> %s", static_dir)
else:
    logger.debug("Static dir %s not found; skipping mount", static_dir)

# ---------- Expose settings on app.state ----------
app.state.settings = settings  # type: ignore[attr-defined]

# ---------- Metrics endpoint ----------
if make_asgi_app is not None:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
else:
    logger.debug("/metrics not mounted because prometheus_client is missing")

# ---------- Classic startup fallback (if not using lifespan) ----------
if not settings.USE_MODERN_STARTUP:
    @app.on_event("startup")
    async def _startup_event():
        await _maybe_warm_index()
        setup_metrics()  # moved here from module-level


# ---------- CLI for local dev (not required by uvicorn) ----------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=settings.BACKEND_PORT, reload=True)