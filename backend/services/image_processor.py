"""
image_processor.py - Normalizes raw logos and banners into web-ready assets.

Usage examples:
    # Process all raw logos and banners (batch)
    .venv/bin/python -m services.image_processor --mode batch --verbose

    # Process a single file
    .venv/bin/python -m services.image_processor --mode single --file /path/to/input.jpg --verbose

Notes:
 - No watcher included (you requested watcher be added later).
 - Input folders (you already created):
     backend/input_data/images/raw_images/raw_logos
     backend/input_data/images/raw_images/raw_banners
 - Output folders (script ensures created):
     backend/input_data/images/processed_images/processed_logos  (PNG)
     backend/input_data/images/processed_images/processed_banners (WEBP)
 - Logos: convert to PNG (preserve alpha if present).
 - Banners: convert to WEBP (lossy WebP default; quality configurable).
 - We strip EXIF/metadata and preserve original resolution (frontend should scale).
"""

from pathlib import Path
from PIL import Image, ImageOps
import argparse
import shutil
import sys
import logging

BASE = Path(__file__).resolve().parents[1]  # backend/
RAW_LOGOS = BASE / "input_data" / "images" / "raw_images" / "raw_logos"
RAW_BANNERS = BASE / "input_data" / "images" / "raw_images" / "raw_banners"
PROC_LOGOS = BASE / "input_data" / "images" / "processed_images" / "processed_logos"
PROC_BANNERS = BASE / "input_data" / "images" / "processed_images" / "processed_banners"

# default webp quality for banners (0-100)
DEFAULT_WEBP_QUALITY = 85

# supported input extensions
_LOGO_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
_BANNER_EXTS = _LOGO_EXTS.copy()

# Ensure directories exist
for p in (PROC_LOGOS, PROC_BANNERS):
    p.mkdir(parents=True, exist_ok=True)

# Logging setup
logger = logging.getLogger("image_processor")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _safe_open_image(path: Path) -> Image.Image | None:
    try:
        img = Image.open(path)
        img.load()  # ensure loaded to catch errors early
        return img
    except Exception as e:
        logger.error("Failed to open image %s: %s", path, e)
        return None


def _strip_exif(img: Image.Image) -> Image.Image:
    """
    Return a copy without EXIF/metadata. Keep mode and data.
    """
    data = img.tobytes()
    new = Image.frombytes(img.mode, img.size, data)
    return new


def _to_png_save(img: Image.Image, out_path: Path):
    """
    Save PIL Image to PNG, preserving alpha where possible, stripping metadata.
    """
    try:
        # Ensure RGBA if image has transparency-like data; else convert to RGB
        if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
            out_img = img.convert("RGBA")
        else:
            out_img = img.convert("RGB")
        out_img = _strip_exif(out_img)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out_path, format="PNG", optimize=True)
        logger.info("Saved PNG: %s", out_path)
    except Exception as e:
        logger.error("Failed to save PNG %s: %s", out_path, e)


def _to_webp_save(img: Image.Image, out_path: Path, quality: int = DEFAULT_WEBP_QUALITY):
    """
    Save PIL Image to WEBP (lossy) with given quality, strip metadata.
    """
    try:
        out_img = img.convert("RGB")  # webp will be RGB (no alpha for banners by default)
        out_img = _strip_exif(out_img)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out_path, format="WEBP", quality=quality, method=6)
        logger.info("Saved WEBP: %s", out_path)
    except Exception as e:
        logger.error("Failed to save WEBP %s: %s", out_path, e)


def _filename_stem_for_output(src: Path) -> str:
    """
    Normalize input filename to a safe stem used for processed name.
    Examples:
      'NETWEB_19-09-2025 18_26_33.pdf' -> 'NETWEB_19-09-2025_18_26_33'
      'Mphasis Ltd_22-09-2025 00_58_53'  -> 'Mphasis_Ltd_22-09-2025_00_58_53'
    (Replaces spaces with underscore and removes extension)
    """
    stem = src.stem
    # replace sequences of whitespace with underscore, and spaces in tokens
    safe = "_".join(stem.split())
    # replace problematic chars
    safe = "".join(c if (c.isalnum() or c in "_-") else "_" for c in safe)
    return safe


def process_logo_file(path: Path, overwrite: bool = True):
    """
    Convert input image to PNG and save in processed_logos with canonical name:
      <safe_stem>_logo.png
    If path is already PNG and already in processed folder, still re-save to ensure normalization.
    """
    if not path.exists():
        logger.warning("Logo file not found: %s", path)
        return None
    if path.suffix.lower() not in _LOGO_EXTS:
        logger.warning("Unsupported logo extension for %s", path)
        return None

    img = _safe_open_image(path)
    if not img:
        return None

    out_name = f"{_filename_stem_for_output(path)}_logo.png"
    out_path = PROC_LOGOS / out_name

    if out_path.exists() and not overwrite:
        logger.debug("Skipping existing logo: %s", out_path)
        return out_path

    _to_png_save(img, out_path)
    return out_path


def process_banner_file(path: Path, overwrite: bool = True, quality: int = DEFAULT_WEBP_QUALITY):
    """
    Convert input image to WEBP and save in processed_banners with canonical name:
      <safe_stem>_banner.webp
    """
    if not path.exists():
        logger.warning("Banner file not found: %s", path)
        return None
    if path.suffix.lower() not in _BANNER_EXTS:
        logger.warning("Unsupported banner extension for %s", path)
        return None

    img = _safe_open_image(path)
    if not img:
        return None

    out_name = f"{_filename_stem_for_output(path)}_banner.webp"
    out_path = PROC_BANNERS / out_name

    if out_path.exists() and not overwrite:
        logger.debug("Skipping existing banner: %s", out_path)
        return out_path

    _to_webp_save(img, out_path, quality=quality)
    return out_path


def process_all_batch(overwrite: bool = True, quality: int = DEFAULT_WEBP_QUALITY, verbose: bool = False):
    """
    Process everything found in RAW_LOGOS and RAW_BANNERS directories.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    logger.info("Processing RAW logos in: %s", RAW_LOGOS)
    logo_paths = sorted(RAW_LOGOS.glob("*"))
    processed = 0
    for p in logo_paths:
        if p.is_file():
            res = process_logo_file(p, overwrite=overwrite)
            if res:
                processed += 1

    logger.info("Processing RAW banners in: %s", RAW_BANNERS)
    banner_paths = sorted(RAW_BANNERS.glob("*"))
    processed_b = 0
    for p in banner_paths:
        if p.is_file():
            res = process_banner_file(p, overwrite=overwrite, quality=quality)
            if res:
                processed_b += 1

    logger.info("Batch complete. logos=%d, banners=%d", processed, processed_b)
    return processed, processed_b


def process_single(path_str: str, kind: str = "auto", overwrite: bool = True, quality: int = DEFAULT_WEBP_QUALITY):
    p = Path(path_str)
    if not p.exists():
        logger.error("File not found: %s", p)
        return None
    # decide kind if auto (logo vs banner)
    if kind == "auto":
        # heuristics: if filename contains 'logo' -> treat as logo; filename contains 'banner' -> banner
        low = p.name.lower()
        if "logo" in low or (p.parent == RAW_LOGOS):
            kind = "logo"
        elif "banner" in low or (p.parent == RAW_BANNERS):
            kind = "banner"
        else:
            # fallback by extension size guess: if width < 400 treat as logo else banner (best effort)
            try:
                img = _safe_open_image(p)
                if img and img.width < 400:
                    kind = "logo"
                else:
                    kind = "banner"
            except Exception:
                kind = "logo"

    if kind == "logo":
        return process_logo_file(p, overwrite=overwrite)
    else:
        return process_banner_file(p, overwrite=overwrite, quality=quality)


def cli_entry():
    parser = argparse.ArgumentParser(description="Image processor (logos -> PNG, banners -> WEBP)")
    parser.add_argument("--mode", choices=["batch", "single"], default="batch", help="batch or single")
    parser.add_argument("--file", help="single file path to process (used with --mode single)")
    parser.add_argument("--quality", type=int, default=DEFAULT_WEBP_QUALITY, help="webp quality for banners (0-100)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing processed files")
    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    if args.mode == "batch":
        process_all_batch(overwrite=args.overwrite, quality=args.quality, verbose=args.verbose)
    else:
        if not args.file:
            logger.error("--mode single requires --file")
            return
        res = process_single(args.file, overwrite=args.overwrite, quality=args.quality)
        if res:
            logger.info("Processed single file -> %s", res)


if __name__ == "__main__":
    cli_entry()