# main_py_test_report_post_fix.md

## Applied Patches Summary
- **P0-1 (CORS Tightening)**: Made CORS env-driven with ALLOWED_ORIGINS and ENVIRONMENT vars. Defaults to ["*"] in dev, [] in prod.
- **P0-2 (Static Mount Restriction)**: Changed to mount only STATIC_DIR (default: input_data/served). Removed processed_images sub-mount.
- **P1 (Docs Disable in Prod)**: Added conditional docs/redoc/openapi URLs based on ENVIRONMENT.
- **P2 (Minor Tidies)**: Removed duplicate processed_images mount line.

## Diffs Applied
### P0-1 CORS
```
- app.add_middleware(
-     CORSMiddleware,
-     allow_origins=["*"],
-     allow_credentials=True,
-     allow_methods=["GET","POST","OPTIONS"],
-     allow_headers=["*"],
- )
+ import os
+ ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS")
+ if ALLOWED_ORIGINS:
+     origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
+ else:
+     origins = ["*"] if os.getenv("ENVIRONMENT","dev") == "dev" else []
+ app.add_middleware(
+     CORSMiddleware,
+     allow_origins=origins,
+     allow_credentials=True,
+     allow_methods=["GET","POST","OPTIONS"],
+     allow_headers=["*"],
+ )
```

### P0-2 Static Mount
```
- # Serve everything inside input_data/ under /static/
- app.mount("/static", StaticFiles(directory="input_data"), name="static")
- app.mount("/static/images/processed_images", StaticFiles(directory="input_data/images/processed_images"), name="processed_images")
+ import os
+ BASE_DIR = Path(__file__).resolve().parents[1]
+ STATIC_DIR = os.getenv("STATIC_DIR", str(BASE_DIR / "input_data" / "served"))
+ # mount only the curated public directory
+ app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
```

### P1 Docs Toggle
```
- app = FastAPI(title="NEXIS Backend", version="0.1.0")
+ import os
+ show_docs = os.getenv("ENVIRONMENT","dev") != "prod"
+ app = FastAPI(
+     title="NEXIS Backend",
+     version="0.1.0",
+     docs_url="/docs" if show_docs else None,
+     redoc_url="/redoc" if show_docs else None,
+     openapi_url="/openapi.json" if show_docs else None,
+ )
```

### P2 Remove Processed Images
Removed line: `app.mount("/static/images/processed_images", StaticFiles(directory="input_data/images/processed_images"), name="processed_images")`

## Command Outputs and Validation Results
- **Precondition Check**: .venv activated, import ok.
- **P0-1 Validation**: Import ok after patch.
- **P0-2 Validation**: Created input_data/served/test.txt. Server started, curl returned 200 OK for /static/test.txt.
- **P1 Validation**: ENVIRONMENT=prod, server started, curl /docs returned 404.
- **P2 Validation**: Import ok after removal.
- **Final Checklist**:
  - Import: ok
  - Server Startup: uvicorn started successfully (logs showed no errors).
  - /health: {"status":"ok"}
  - /openapi.json: 200 OK (in dev mode).

## Follow-up Recommendations
- Set ALLOWED_ORIGINS and ENVIRONMENT in production .env.
- Move public assets to input_data/served or set STATIC_DIR.
- Add unit tests for CORS and static serving.
- Monitor for any frontend issues with new static path.

## Changelog
- P0-1: Tightened CORS for security.
- P0-2: Restricted static serving to prevent data leaks.
- P1: Disabled docs in prod.
- P2: Cleaned up duplicate mounts.
- Backups: main.py.bak created before each patch (kept in repo for rollback).