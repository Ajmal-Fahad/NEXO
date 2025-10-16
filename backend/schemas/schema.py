from pydantic import BaseModel
from typing import Optional, List

class AnnouncementListItem(BaseModel):
    id: str
    company_name: str
    company_logo: Optional[str] = None
    headline: str
    announcement_datetime: Optional[str] = None
    sentiment: Optional[str] = None
    sentiment_emoji: Optional[str] = None
    symbol: Optional[str] = None

class MarketSnapshot(BaseModel):
    # Keep field names aligned with the master JSON so Pydantic can parse without aliases
    symbol: Optional[str] = None
    company_name: Optional[str] = None
    rank: Optional[int] = None
    price: Optional[float] = None
    change_1d_pct: Optional[float] = None
    change_1w_pct: Optional[float] = None
    vwap: Optional[float] = None
    # large numbers in JSON are stored as "mcap_rs_cr" and volumes as "volume_24h_rs_cr"
    mcap_rs_cr: Optional[float] = None
    volume_24h_rs_cr: Optional[float] = None
    all_time_high: Optional[float] = None
    atr_pct: Optional[float] = None
    relative_vol: Optional[float] = None
    vol_change_pct: Optional[float] = None
    volatility: Optional[float] = None
    market_snapshot_date: Optional[str] = None
    # URLs may be provided as lists
    logo_url: Optional[List[str]] = None
    banner_url: Optional[List[str]] = None
    broad_index: Optional[str] = None
    sector_index: Optional[str] = None

class AnnouncementDetail(AnnouncementListItem):
    summary_60: Optional[str] = None
    impact: Optional[str] = None
    indices: Optional[List[str]] = None
    market_snapshot: Optional[MarketSnapshot] = None
    tradingview_url: Optional[str] = None
    source_file: Optional[str] = None
    banner_image: Optional[str] = None
