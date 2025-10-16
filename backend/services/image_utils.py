"""
image_utils.py - Helpers to serve processed logos and banners for frontend
"""

import re
from pathlib import Path
from typing import Optional, Tuple, List

# -------------------------
# Config
# -------------------------
BASE = Path(__file__).resolve().parents[1]
PROCESSED_LOGOS = BASE / "input_data" / "images" / "processed_images" / "processed_logos"
PROCESSED_BANNERS = BASE / "input_data" / "images" / "processed_images" / "processed_banners"

# URL base (frontend can serve these via FastAPI static route)
URL_BASE = "/static/images"
PLACEHOLDER_LOGO = "default_logo.png"
PLACEHOLDER_BANNER = "default_banner.webp"

# -------------------------
# Normalizers
# -------------------------
def normalize_symbol(sym: str) -> str:
    """Normalize trading symbol to uppercase with underscores."""
    if not sym:
        return ""
    return re.sub(r"\s+", "_", sym.strip().upper())

def slugify_company(name: str) -> str:
    """Make a lowercase slug from company name (safe for filenames)."""
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

# -------------------------
# Candidate generators
# -------------------------
def candidate_logo_filenames(symbol: str, company_name: str) -> List[str]:
    sym = normalize_symbol(symbol)
    slug = slugify_company(company_name)
    cands = []
    if sym:
        cands.append(f"{sym}_logo.png")
        cands.append(f"{sym.lower()}_logo.png")
    if slug:
        cands.append(f"{slug}_logo.png")
    return cands

def candidate_banner_filenames(symbol: str, company_name: str) -> List[str]:
    sym = normalize_symbol(symbol)
    slug = slugify_company(company_name)
    cands = []
    if sym:
        cands.append(f"{sym}_banner.webp")
        cands.append(f"{sym.lower()}_banner.webp")
        cands.append(f"{sym}_banner.png")
    if slug:
        cands.append(f"{slug}_banner.webp")
        cands.append(f"{slug}_banner.png")
    return cands

# -------------------------
# Lookup helper
# -------------------------
def find_first_existing(base_dir: Path, candidates: List[str]) -> Optional[Path]:
    for fn in candidates:
        p = base_dir / fn
        if p.exists():
            return p
    return None

# -------------------------
# Public API
# -------------------------
def get_logo_path(symbol: str, company_name: str) -> Tuple[Path, str]:
    """
    Returns (filesystem_path, url_path) for logo.
    Falls back to default placeholder if not found.
    """
    candidates = candidate_logo_filenames(symbol, company_name)
    path = find_first_existing(PROCESSED_LOGOS, candidates)
    if not path:
        path = PROCESSED_LOGOS / PLACEHOLDER_LOGO
    url = f"{URL_BASE}/processed_logos/{path.name}"
    return path, url

def get_banner_path(symbol: str, company_name: str) -> Tuple[Path, str]:
    """
    Returns (filesystem_path, url_path) for banner.
    Falls back to default placeholder if not found.
    """
    candidates = candidate_banner_filenames(symbol, company_name)
    path = find_first_existing(PROCESSED_BANNERS, candidates)
    if not path:
        path = PROCESSED_BANNERS / PLACEHOLDER_BANNER
    url = f"{URL_BASE}/processed_banners/{path.name}"
    return path, url

# -------------------------
# CLI test
# -------------------------
if __name__ == "__main__":
    sym = "RELIANCE"
    name = "Reliance Industries Limited"

    logo_path, logo_url = get_logo_path(sym, name)
    banner_path, banner_url = get_banner_path(sym, name)

    print("Logo Path:", logo_path)
    print("Logo URL:", logo_url)
    print("Banner Path:", banner_path)
    print("Banner URL:", banner_url)