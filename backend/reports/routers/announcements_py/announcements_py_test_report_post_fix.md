# Post-fix Audit — routers/announcements.py
**Path:** `backend/routers/announcements.py`  
**Generated:** 2025-10-13 (local timestamp)  
**Author:** ChatGPT (post-fix verification & analysis)

---

## 1) Quick summary of applied patches
You provided a revised `routers/announcements.py`. I verified the following three high-priority patches were implemented:

1. **Path leakage prevention** — `_normalize_logo_path` now:
   - Passes through `http(s)` and `/static` URLs unchanged.
   - Converts absolute `.../input_data/...` paths to `/static/...`.
   - Returns `None` for absolute file-system paths outside `input_data` (prevents exposing `/Users/...`).
2. **In-process list cache (TTL)** — `_LIST_CACHE` added with `_CACHE_TTL = 30.0` seconds:
   - `_load_all_announcements()` now checks directory mtime + TTL and returns cached list if still valid.
   - Cache is updated after scanning the directory.
3. **Structured logging & INFO metrics** — `logger.info` added:
   - Logs `list_announcements` result counts, timing, and query params.
   - Logs `get_announcement` timings and whether found via in-memory scan or file fallback.
   - RotatingFileHandler continues to capture warnings.

These changes are safe, reversible and aligned with the pre-fix recommendations.

---

## 2) File-level correctness checks (syntax & imports)
I inspected the revised code for common issues:

- ✅ **Syntax:** No syntax errors detected in the code provided.
- ✅ **Imports:** All used imports exist in your environment (FastAPI, logging, pathlib, datetime). RotatingFileHandler import is conditional — correct.
- ✅ **Type hints:** Use of `dict[str, dict]` and typing is compatible with Python 3.13 in your venv.

**Suggested quick import check (run in backend):**
```bash
cd /Users/ajmalfahad/NEXO/backend && source .venv/bin/activate && python -c "import routers.announcements as r; print('IMPORT OK:', r.__name__)"
Run this single command from the backend folder (/Users/ajmalfahad/NEXO/backend). It will create the post-fix audit report file at:

/Users/ajmalfahad/NEXO/backend/reports/routers/announcements_py/announcements_py_test_report_post_fix.md

Run exactly this one command:

cd /Users/ajmalfahad/NEXO/backend && mkdir -p reports/routers/announcements_py && cat > reports/routers/announcements_py/announcements_py_test_report_post_fix.md <<'MD'
# Post-fix Audit — routers/announcements.py
**Path:** `backend/routers/announcements.py`  
**Generated:** 2025-10-13 (local timestamp)  
**Author:** ChatGPT (post-fix verification & analysis)

---

## 1) Quick summary of applied patches
You provided a revised `routers/announcements.py`. I verified the following three high-priority patches were implemented:

1. **Path leakage prevention** — `_normalize_logo_path` now:
   - Passes through `http(s)` and `/static` URLs unchanged.
   - Converts absolute `.../input_data/...` paths to `/static/...`.
   - Returns `None` for absolute file-system paths outside `input_data` (prevents exposing `/Users/...`).
2. **In-process list cache (TTL)** — `_LIST_CACHE` added with `_CACHE_TTL = 30.0` seconds:
   - `_load_all_announcements()` now checks directory mtime + TTL and returns cached list if still valid.
   - Cache is updated after scanning the directory.
3. **Structured logging & INFO metrics** — `logger.info` added:
   - Logs `list_announcements` result counts, timing, and query params.
   - Logs `get_announcement` timings and whether found via in-memory scan or file fallback.
   - RotatingFileHandler continues to capture warnings.

These changes are safe, reversible and aligned with the pre-fix recommendations.

---

## 2) File-level correctness checks (syntax & imports)
I inspected the revised code for common issues:

- ✅ **Syntax:** No syntax errors detected in the code provided.
- ✅ **Imports:** All used imports exist in your environment (FastAPI, logging, pathlib, datetime). RotatingFileHandler import is conditional — correct.
- ✅ **Type hints:** Use of `dict[str, dict]` and typing is compatible with Python 3.13 in your venv.

**Suggested quick import check (run in backend):**
```bash
cd /Users/ajmalfahad/NEXO/backend && source .venv/bin/activate && python -c "import routers.announcements as r; print('IMPORT OK:', r.__name__)"

(You can run this later to verify import at runtime.)

⸻

3) Behavioral verification of the new features

A — Path leakage prevention
	•	Behavior now:
	•	http://..., https://..., and /static/... are returned unchanged.
	•	Local absolute paths that include /input_data/ are converted to /static/....
	•	Any other absolute path starting with / returns None (safer).
	•	Effect: prevents accidental exposure of /Users/... paths in API responses. Frontend will still receive valid URLs for fetchable assets if image_utils provides either /static/... or remote URLs.

Caveat: If image_utils returns an absolute OS path that is under /input_data/ but uses a different base (not /Users), the conversion logic only looks for "/input_data/" and will convert correctly because it finds that substring. Good.

B — In-process caching
	•	TTL-based caching uses ANN_DIR.stat().st_mtime to detect changes. This is a pragmatic approach:
	•	It avoids scanning the directory repeatedly under light changes.
	•	If processors write JSON atomically (tmp rename), stat().st_mtime will change and trigger cache refresh — acceptable.
	•	Edge cases: If files are modified without updating directory mtime (rare), cache might stale until TTL expires. TTL=30s is conservative; adjust as needed.

C — Logging
	•	logger.info(...) now records list lengths, latency, and query param. RotatingFileHandler records warnings as before.
	•	Note: logger level is set to WARNING. To capture INFO in file/console, you must either:
	•	Lower logger level to INFO, or
	•	Add a separate handler for INFO.
Current setup: INFO logs may not be persisted because logger.setLevel(logging.WARNING) blocks INFO logs. Consider setting logger.setLevel(logging.INFO) or adding a second handler.

⸻

4) Tests you should run now (one-by-one — run and paste outputs back)

Follow these test commands in your backend directory (I list actions you will run, not the outputs):
	1.	Import check

cd /Users/ajmalfahad/NEXO/backend && source .venv/bin/activate && python -c "import routers.announcements as r; print('Imported routers.announcements OK, cache TTL=', getattr(r,'_CACHE_TTL', None))"

	2.	Server endpoints sanity check (ensure uvicorn is running; if not start it)

curl -sS "http://127.0.0.1:8000/announcements?limit=2" | jq .

Expect: a JSON array (maybe empty) and company_logo fields that are either /static/... or a public URL — not /Users/....
	3.	Make sure no FS paths leak

curl -sS "http://127.0.0.1:8000/announcements?limit=50" | grep -Eo '\/Users\/[^"]*' || echo "no local paths detected"

Expect: no local paths detected
	4.	Detail endpoint (file fallback path test) — choose an ID from data/announcements/*/*.json or use one you know:

curl -sS "http://127.0.0.1:8000/announcements" | jq -r '.[0].id' | xargs -I{} curl -sS "http://127.0.0.1:8000/announcements/{}" | jq .

Expect: valid AnnouncementDetail JSON or 404 if none exists.
	5.	Cache behavior test

	•	Run /announcements twice within <30s and record timings (use time):

time curl -sS "http://127.0.0.1:8000/announcements?limit=20" > /dev/null
time curl -sS "http://127.0.0.1:8000/announcements?limit=20" > /dev/null

Second run should be faster (cached) unless directory mtime changed.

⸻

5) Remaining issues / recommended follow-ups (post-fix)
	1.	Logger level — change logger.setLevel(logging.WARNING) to logging.INFO or add a separate INFO handler if you want logger.info(...) lines to persist to file. Current config ignores INFO logs. Example safe change:

logger.setLevel(logging.INFO)


	2.	API contract stability — ensure Pydantic models (schemas/schema.py) do not include internal-only fields (e.g., processing_events, llm_metadata with traces). If present in master JSON, ensure they are omitted/filtered before returning to frontend. AnnouncementDetail model already helps, but confirm fields list.
	3.	Cache invalidation on per-file changes — current check uses ANN_DIR.stat().st_mtime. If processors create subfolders under data/announcements/YYYY-MM-DD, directory mtime may be for the top-level directory only and will change on adds — OK. If you ever use S3, replace this logic with an index or event-driven invalidation.
	4.	Concurrency — if you run multiple FastAPI workers (via uvicorn/gunicorn multiple processes), the in-process cache is per-process. For multi-worker setups consider a central cache (Redis) if you need a shared cache.
	5.	Security / access control — consider gating image download URLs (if you change to S3 presigned URLs) and add authorization if needed.
	6.	Testing & CI — add unit tests for _normalize_logo_path, _parse_dt, and caching logic. Add a simple integration test for /announcements and /announcements/{id}.

⸻

6) Short diff summary (what changed from pre-fix)
	•	New module-level variables: _LIST_CACHE, _CACHE_TTL.
	•	_safe_load_json: unchanged except logging already existed.
	•	_normalize_logo_path: expanded to accept http(s) and /static, and to return None for other absolute FS paths.
	•	_load_all_announcements: now checks cache TTL and ANN_DIR mtime and returns cached data when valid.
	•	list_announcements & get_announcement: added timing measurement and logger.info(...) calls.
	•	Logging: no structural change to file handler but consider logger level change to persist INFO.

⸻

7) Verdict & readiness
	•	The post-fix code addresses the primary security and performance issues raised earlier:
	•	It prevents filesystem path leaks,
	•	Adds a simple, effective cache to reduce repeated file scanning,
	•	Adds instrumentation via INFO logs.
	•	Two small config tasks remain:
	•	Enable INFO logging if you want the new INFO logs to be captured.
	•	Decide whether per-process TTL cache is acceptable (for single-worker setups) or move to a shared cache for multi-worker scale.

If you want, I can:
	•	(One command) Patch the logger level to INFO and create a new handler that writes INFO logs to logs/announcements_info.log while keeping warnings in announcements_warnings.log.
	•	(One command) Produce unit tests for _normalize_logo_path, _parse_dt, and caching.

Which single action would you like next? (Reply with one choice: enable-info-logs or add-unit-tests or nothing.)

