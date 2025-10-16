# Pre-fix Audit — routers/announcements.py
**Path:** `backend/routers/announcements.py`  
**Generated:** 2025-10-13 (local timestamp)  
**Author:** ChatGPT (pre-fix analysis)

---

## 1) Quick summary
`routers/announcements.py` is the main public-facing router that serves announcement summaries and announcement detail objects to clients. It reads announcement master JSON files from `data/announcements`, normalizes and sanitizes fields, provides pagination/search/deduping for the list endpoint, and validates detail responses with Pydantic models.

Overall: **well-structured and fit for purpose**, with proper use of Pydantic response models, defensive JSON loading, and sanitization logic. Several improvements are recommended for robustness, security, observability, and performance at scale.

---

## 2) What the file does (functionally)
- Exposes two main endpoints:
  - `GET /announcements` — lists lightweight `AnnouncementListItem` objects. Supports `limit`, `offset`, `q`.
  - `GET /announcements/{announcement_id}` — returns `AnnouncementDetail` for a specific announcement id.
- Reads JSON files live from `data/announcements` on every request (no in-memory caching).
- Normalizes image paths (converts local absolute `input_data` paths to `/static/...`), parses dates, dedupes announcements by `id`, and sorts by announcement datetime.
- Adds logging (rotating file handler) for warnings (e.g., bad JSON files).

---

## 3) Syntax & import checks
- ✅ No syntax errors; module imports successfully when services and schemas exist.
- Uses type hints (e.g., `List[AnnouncementListItem]`) and modern Python features (dict[str, dict]) — compatible with Python 3.13 in your environment.
- Logging initialization is safe (creates `backend/logs` and RotatingFileHandler).

Verdict: **Syntax / imports OK.**

---

## 4) Runtime / correctness risks
1. **Reading all files on each request**  
   - Risk: As `data/announcements` grows to hundreds/thousands of files, `GET /announcements` will slow and use more memory. The list endpoint builds all items then slices; this is acceptable for small volumes but not scalable.

2. **_parse_dt fragility**  
   - It tries multiple formats and falls back to a final parse; unparseable dates return `0.0` (oldest). This behavior is acceptable but should be logged when many entries have unparseable dates.

3. **Date/time timezone ambiguity**  
   - `datetime.fromisoformat` may accept a variety of ISO formats; ensure consistent writing of `announcement_datetime_iso` in generator (pdf_processor). Consider adding a canonical epoch (integer) `announcement_timestamp` in master JSON for reliable comparisons.

4. **Normalization assumptions for logos**  
   - `_normalize_logo_path` assumes absolute paths start with `/Users` and contain `/input_data/`. On other machines or S3-based deployments these assumptions fail. It will return the original string for anything else (which may be a filesystem path or an already-public URL). This can leak internal paths to clients.

5. **Potential Type/Structure mismatches**  
   - The code expects certain structures (e.g., `market_snapshot` may contain `logo_url` as list or str). There are defensive checks, but inconsistent upstream shapes may still produce incorrect `company_logo` values.

6. **Error handling**  
   - `_safe_load_json` catches exceptions and logs a warning — good. In `get_announcement`, the code falls back to direct file access if not found in in-memory scan — robust.

---

## 5) Performance & scalability concerns
- **Disk I/O**: Repeated file reads are the main bottleneck. Recommended fixes:
  - Introduce caching (in-memory LRU or Redis) for the list output for a short TTL (e.g., 30–60s) or cache per date folder.
  - Maintain an index (e.g., SQLite or small JSON index) updated when processors write JSON to avoid scanning the directory each request.
- **Filtering & dedupe**: Current approach builds full `lite` list then filters/dedupes. For large datasets, prefer streaming processing or precomputed indices to query/filter quickly.
- **Pagination**: The endpoint supports `limit` & `offset`; however offset-based pagination on large lists is expensive when skipping many items — consider cursor-based pagination if needed.

---

## 6) Security / exposure
- **Local path exposure**: `_normalize_logo_path` tries to convert local paths into `/static/...`, but if not matching the pattern it may return absolute local paths or internal file-system strings. Those might be exposed in API responses. **Do not serve absolute internal paths** publicly.
- **Static mount implications**: main.py currently serves `input_data` (discussed earlier). Ensure static serving is limited to a public folder (e.g., `input_data/served`), and keep raw/processed files private or behind presigned URLs when on S3.
- **Raw JSON leakage**: Although this router serves sanitized views, ensure no internal-only fields (LLM traces, file-system traces) are present in `AnnouncementDetail` model or are redacted before returning to clients.

---

## 7) Observability & logging
- A rotating warning logger exists and logs skipped/invalid JSONs. Suggest upgrading:
  - Add `INFO` logs for list requests (count returned, duration) and detail requests (id requested, found).
  - Log slow responses (>500ms) to help find bottlenecks.
  - Consider structured logs (JSON) for easier ingestion to monitoring tools.

---

## 8) Usefulness to frontend
- This file is **directly required by frontend** for announcement listing and detail views.
- Returned schemas are Pydantic models — great for typed clients. Keep stable schema versions or add explicit API versioning (`/v1/announcements`) before breaking changes.
- `company_logo` should be a URL string that frontend can fetch (absolute CDN or `/static/...`). Avoid sending local filesystem paths.

---

## 9) Suggested fixes (prioritized)
**High priority (apply ASAP)**  
1. **Prevent leaking local filesystem paths**: Normalize/transform all `company_logo`, `banner` fields to either (A) `/static/...` or (B) a placeholder or (C) an S3/CDN URL. Never return absolute system paths. (Patch location: `_normalize_logo_path` + callers in `_build_list_item` and `get_announcement`.)  
2. **Cache list endpoint**: Add a short TTL in-memory cache (LRU with TTL or simple module-level timestamped cache) to avoid scanning JSON files for every request. Consider a cache key based on `ANN_DIR` mtime.  
3. **Log unparseable datetimes**: When `_parse_dt` returns 0.0 for many items, log them for processor fixes.

**Medium priority**  
4. **Add API rate-limiting or auth** for public deployment.  
5. **Add metrics (request duration, counts)** to instrument endpoints.  
6. **Convert date parsing to use an epoch field**: Prefer `announcement_timestamp` (seconds int) in master JSON to simplify sorting and avoid repeated parsing overhead.

**Low priority / optional**  
7. Consider cursor-based pagination for large result sets.  
8. Limit `q` search scope or add an indexed search if needed (Elasticsearch/SQLite FTS) for advanced search.

---

## 10) Concrete patch suggestions (safe & reversible)
I provide minimal, reversible patches you can apply one by one. (I will not apply them unless you ask.) Example patches:

**A. Prevent path leakage (recommended minimal change)**  
- Modify `_normalize_logo_path` to:
  - If input is absolute path under `input_data`, convert to `/static/...`
  - Else if input looks like an absolute FS path (starts with `/`) but not under input_data, return `None` (to avoid exposure).
  - Accept and pass through http(s) or `/static` URLs unchanged.

**B. Add in-process caching for list_announcements**  
- Add module-level cache variable:
  ```py
  _LIST_CACHE = {"ts": 0.0, "data": [], "dir_mtime": 0.0}
	•	On call, check ANN_DIR.stat().st_mtime (or compute hash of filenames) and time.time() for TTL (e.g., 30s). Return cached out if valid.

C. Emit structured INFO logs
	•	Add logger.info("list_announcements returned %d items (limit=%d offset=%d) in %.2fms", len(out), limit, offset, elapsed_ms)

I can produce the exact patch diffs and one-line sed or python - <<'PY' commands to apply each (one command per patch) when you say which patch to apply first.

⸻

11) Tests to validate behavior (one-offs)

Run these tests after applying fixes:
	1.	Functional
	•	curl -sS "http://127.0.0.1:8000/announcements?limit=10" | jq . — ensure response is JSON array and company_logo is a web-safe path/URL.
	•	curl -sS "http://127.0.0.1:8000/announcements/<some-id>" | jq . — ensure AnnouncementDetail returns valid fields.
	2.	Performance
	•	Simulate many JSON files using a script (create 1000 small announcement JSONs) and measure response time for /announcements. Expect direct scan to slow; caching should help.
	3.	Security
	•	Check responses do not contain any /Users/ or other absolute paths: curl ... | grep "/Users" || echo "no local paths".

⸻

12) Recommended next step (one command)

If you want a concrete immediate improvement, I recommend preventing local path leakage in _normalize_logo_path first. This is small, high-impact, and low-risk.

If you confirm, I will give a single command that will:
	•	create a safe backup routers/announcements.py.bak,
	•	patch _normalize_logo_path to the safer version,
	•	and leave the file importable.

Say “apply path-fix” if you want that now; otherwise say “next” and tell me which item from section 9 to do first.

⸻

Appendix: key lines referenced
	•	File listing & logging setup: lines 10–22
	•	ANN_DIR path: line 28
	•	_safe_load_json: lines 32–38
	•	_normalize_logo_path: lines 40–52
	•	List endpoint: lines 136–172
	•	Detail endpoint: lines 175–241

