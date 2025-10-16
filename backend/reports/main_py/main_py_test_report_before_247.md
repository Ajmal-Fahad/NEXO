# main.py — Pre-Fix Test Report

**File:** backend/main.py
**Generated:** 2025-10-16 23:21 IST

## Purpose
- Initializes FastAPI app, middleware, routers, and mounts static files.

## What works well
- Docs gating by ENVIRONMENT implemented.
- Raw announcements router gated by env var.
- `ALLOWED_ORIGINS` support present.

## Issues & Risks
1. Static mount exposes entire `input_data` (may leak raw PDFs, CSVs).
2. Duplicate `import os` statements — tidy.
3. CORS: in prod, `origins = []` unless ALLOWED_ORIGINS set (could block legitimate frontends).
4. No centralized config loader — env vars used ad-hoc.
5. No startup logging to report configuration values.
6. StaticFiles uses relative path — may behave inconsistently on different working dirs.
7. No auth/rate-limiting (future concern for public deployment).

## Recommended Minimal Changes
- Serve a restricted, configurable static directory (e.g., `input_data/served`).
- Centralize config values at top of file (ENVIRONMENT, ALLOWED_ORIGINS, STATIC_DIR, ENABLE_RAW_ANNOUNCEMENTS).
- Add startup logging (environment, origins, static_dir).
- Remove duplicate imports.
- Document recommended uvicorn run for production.

## Suggested patch plan (reversible)
1. Add config block (ENVIRONMENT, ALLOWED_ORIGINS, STATIC_DIR).
2. Replace `app.mount(...)` with `app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")`.
3. Add `logger.info` lines printing startup config.
4. Optionally add a requirement check to refuse to start in prod if STATIC_DIR is not configured.

## Safety & Verification
- Changes are non-breaking (only config & logging).
- I will create a `.bak` before applying patches if you ask me to apply them.
- After patch: run `python -c "import main; print('main imported OK')"` and `uvicorn main:app --reload` then `curl` endpoints.

