from pathlib import Path
from typing import List, Dict, Any
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

# base = backend/ (routers/ is backend/routers)
BASE = Path(__file__).resolve().parents[1]
ANN_DIR = BASE / "data" / "announcements"

router = APIRouter(prefix="/raw_announcements", tags=["raw_announcements"])


@router.get("/", response_class=JSONResponse)
def list_raw_announcements() -> List[Dict[str, Any]]:
    """
    Return the full master JSON for every file in data/announcements as a list.
    Useful for debugging. Files are loaded in sorted order.
    """
    out = []
    if not ANN_DIR.exists():
        return out
    for p in sorted(ANN_DIR.glob("*.json")):
        try:
            txt = p.read_text(encoding="utf-8")
            j = json.loads(txt)
            out.append(j)
        except Exception:
            # skip invalid JSON files
            continue
    return JSONResponse(content=out)


@router.get("/{ann_id}", response_class=JSONResponse)
def get_raw_announcement(ann_id: str) -> Dict[str, Any]:
    """
    Return the raw master JSON for a specific announcement id (filename without .json).
    Example id: ann_20250923163704_b0ea86
    """
    p = ANN_DIR / f"{ann_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Announcement not found")
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read announcement JSON")
    return JSONResponse(content=j)
