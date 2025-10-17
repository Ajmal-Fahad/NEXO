from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from routers import health
import json
import os

show_docs = os.getenv("ENVIRONMENT", "dev") != "prod"
app = FastAPI(
    title="NEXO Backend",
    version="0.0.1",
    docs_url="/docs" if show_docs else None,
    redoc_url="/redoc" if show_docs else None,
    openapi_url="/openapi.json" if show_docs else None,
)

# Add CORS middleware
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS")
if ALLOWED_ORIGINS:
    origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
else:
    origins = ["*"] if os.getenv("ENVIRONMENT", "dev") == "dev" else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(health.router)
from routers import announcements
app.include_router(announcements.router)

if os.getenv("ENABLE_RAW_ANNOUNCEMENTS", "false").lower() in ("1", "true", "yes"):
    from routers import raw_announcements
    app.include_router(raw_announcements.router)

# Serve everything inside input_data/ under /static/
app.mount("/static", StaticFiles(directory="input_data"), name="static")
