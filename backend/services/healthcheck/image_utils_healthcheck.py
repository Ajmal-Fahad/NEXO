#!/usr/bin/env python3
"""
Healthcheck for image_utils module.
Ensures S3 connectivity, .env correctness, and processed_images path accessibility.
"""

import os
import sys
import importlib
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).resolve().parents[2] / ".env"
if env_path.exists():
    load_dotenv(env_path)

s3_prefix = os.getenv("S3_PROCESSED_IMAGES_PATH", "").strip()
if not s3_prefix:
    print("⚠️  S3_PROCESSED_IMAGES_PATH not set — assuming local-only mode.")
    sys.exit(0)

# Check for fsspec module
try:
    fsspec = importlib.import_module("fsspec")
except Exception as e:
    print(f"❌ fsspec not installed: {e}")
    sys.exit(2)

try:
    fs = fsspec.filesystem("s3")
    if not fs.exists(s3_prefix):
        files = fs.glob(f"{s3_prefix.rstrip('/')}/*")
        if not files:
            print(f"❌ S3 path not accessible or empty: {s3_prefix}")
            sys.exit(3)
    print(f"✅ S3 connectivity OK — Path accessible: {s3_prefix}")
    sys.exit(0)
except Exception as e:
    print(f"❌ S3 check failed: {e}")
    sys.exit(4)
