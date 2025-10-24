"""
services/sentiment_utils.py

Provides compute_sentiment(main_text, llm_raw=None) used by pdf_processor.py.

Author: generated helper for NEXIS-APP
"""

from __future__ import annotations
import re
from typing import Optional, Dict, Any, List

# ---------- Configuration ----------
LLM_WEIGHT = 0.4
KEYWORD_WEIGHT = 0.6

POS_THRESHOLD = 0.7
NEG_THRESHOLD = 0.3

EMOJI = {
    "Positive": "🟢",
    "Negative": "🔴",
    "Neutral": "🟧",  # amber for neutral
    "Ambiguous": "🟦",  # blue for ambiguous
}

# Base keyword sets (expand as needed)
POSITIVE_WORDS = {
    "profit",
    "profits",
    "revenue",
    "growth",
    "beat",
    "dividend",
    "bonus",
    "stock split",
    "buyback",
    "buy-back",
    "upgrade",
    "credit rating upgrade",
    "successful fundraise",
    "listing of subsidiary",
    "order",
    "contract win",
    "profit growth",
    "revenue growth",
    "increase in revenue",
    "launch",
    "unveil",
    "unveils",
    "breakthrough",
    "innovation",
    "innovative",
    "expand",
    "expansion",
}

NEGATIVE_WORDS = {
    "fraud",
    "fraudulent",
    "loss",
    "losses",
    "loss-making",
    "disappoint",
    "warning",
    "downgrade",
    "dividend cut",
    "penalty",
    "regulatory action",
    "sebi",
    "delisting",
    "suspension",
    "default",
    "invocation",
    "pledge invoked",
    "adverse",
}

NEUTRAL_WORDS = {
    "agm",
    "annual general meeting",
    "board meeting",
    "notice",
    "filing",
    "registered office",
    "shareholding pattern",
    "compliance",
    "investor presentation",
    "summary",
    "outcome",
    "minutes",
    "result",
    "scrutinizer",
    "voting",
    "resolution",
    "intimation",
}

AMBIGUOUS_WORDS = {
    "merger",
    "amalgamation",
    "acquisition",
    "demerger",
    "arrangement",
    "rights issue",
    "preferential allotment",
    "private placement",
    "kmp change",
    "ceo change",
    "cfo change",
    "md change",
    "related party transaction",
    "rpt",
    "esop",
    "warrants",
    "conversion",
    "sale of undertaking",
    "asset transfer",
    "business update",
    "material event",
    "m&a",
    "takeover",
}


# ---------- helpers ----------
def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[a-z0-9\-\']+", text.lower())


def _find_phrase_matches(text: str, phrases: List[str]) -> List[str]:
    text_l = text.lower()
    found = []
    for p in phrases:
        if not p:
            continue
        if p in text_l:
            found.append(p)
    return found


def _keyword_counts(text: str) -> Dict[str, Any]:
    if not text:
        return {
            "pos": 0,
            "neg": 0,
            "neu": 0,
            "amb": 0,
            "matches": {"positive": [], "negative": [], "neutral": [], "ambiguous": []},
            "raw_counts": {"pos": 0, "neg": 0, "neu": 0, "amb": 0},
        }

    text_l = text.lower()
    matches = {"positive": [], "negative": [], "neutral": [], "ambiguous": []}

    # phrase-first search (longer phrases first)
    pos_phrases = sorted(POSITIVE_WORDS, key=lambda s: -len(s))
    neg_phrases = sorted(NEGATIVE_WORDS, key=lambda s: -len(s))
    neu_phrases = sorted(NEUTRAL_WORDS, key=lambda s: -len(s))
    amb_phrases = sorted(AMBIGUOUS_WORDS, key=lambda s: -len(s))

    matches["positive"].extend(_find_phrase_matches(text_l, pos_phrases))
    matches["negative"].extend(_find_phrase_matches(text_l, neg_phrases))
    matches["neutral"].extend(_find_phrase_matches(text_l, neu_phrases))
    matches["ambiguous"].extend(_find_phrase_matches(text_l, amb_phrases))

    # token-level matches to catch single words or repetitions
    tokens = _tokenize(text)
    for t in tokens:
        if t in POSITIVE_WORDS and t not in matches["positive"]:
            matches["positive"].append(t)
        if t in NEGATIVE_WORDS and t not in matches["negative"]:
            matches["negative"].append(t)
        if t in NEUTRAL_WORDS and t not in matches["neutral"]:
            matches["neutral"].append(t)
        if t in AMBIGUOUS_WORDS and t not in matches["ambiguous"]:
            matches["ambiguous"].append(t)

    raw_counts = {
        "pos": len(matches["positive"]),
        "neg": len(matches["negative"]),
        "neu": len(matches["neutral"]),
        "amb": len(matches["ambiguous"]),
    }
    return {
        "pos": raw_counts["pos"],
        "neg": raw_counts["neg"],
        "neu": raw_counts["neu"],
        "amb": raw_counts["amb"],
        "matches": matches,
        "raw_counts": raw_counts,
    }


def _keyword_score_from_counts(pos: int, neg: int, neu: int, amb: int) -> float:
    denom = max(1, pos + neg + neu + amb)
    raw = (pos - neg + 0.5 * amb) / denom
    # clamp raw to [-1, 1]
    raw = max(-1.0, min(1.0, raw))
    score = (raw + 1.0) / 2.0
    return float(max(0.0, min(1.0, score)))


def _label_from_score(score: float, ambiguous: bool) -> str:
    if ambiguous:
        return "Ambiguous"
    if score > POS_THRESHOLD:
        return "Positive"
    if score < NEG_THRESHOLD:
        return "Negative"
    return "Neutral"


def _emoji(label: str) -> str:
    return EMOJI.get(label, "🟧")


# ---------- Public API ----------
def compute_sentiment(
    main_text: str, llm_raw: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    main_text: extracted main body text (string)
    llm_raw: optional LLM result dict (may contain 'label' and 'score' keys)
    Returns a dict:
      {
        label, score, emoji, sources: [...], raw_responses: {...}
      }
    """
    try:
        kw = _keyword_counts(main_text)
        keyword_score = _keyword_score_from_counts(
            kw["pos"], kw["neg"], kw["neu"], kw["amb"]
        )
        ambiguous_found = kw["amb"] > 0

        # process llm_raw defensively
        llm_score = None
        llm_label = None
        if llm_raw and isinstance(llm_raw, dict):
            # accept different field names
            if "score" in llm_raw:
                try:
                    llm_score = float(
                        llm_raw.get("score") or llm_raw.get("confidence") or 0.0
                    )
                except Exception:
                    llm_score = None
            if "label" in llm_raw:
                llm_label = str(llm_raw.get("label"))

        # default LLM score neutral if missing (so keywords dominate per weights)
        if llm_score is None:
            llm_score = 0.5

        final_score = float(LLM_WEIGHT * llm_score + KEYWORD_WEIGHT * keyword_score)

        if ambiguous_found:
            label = "Ambiguous"
        else:
            label = _label_from_score(final_score, ambiguous=False)

        emoji = _emoji(label)

        sources = []
        if llm_raw is not None:
            sources.append(
                {
                    "source": "llm",
                    "label": llm_label
                    or (
                        "Positive"
                        if llm_score > 0.6
                        else "Negative" if llm_score < 0.4 else "Neutral"
                    ),
                    "score": round(llm_score, 3),
                }
            )

        sources.append(
            {
                "source": "keyword",
                "label": (
                    "Positive"
                    if keyword_score > 0.6
                    else "Negative" if keyword_score < 0.4 else "Neutral"
                ),
                "score": round(keyword_score, 3),
                "matches": kw.get("matches", {}),
                "raw_counts": kw.get("raw_counts", {}),
            }
        )

        return {
            "label": label,
            "score": round(final_score, 3),
            "emoji": emoji,
            "sources": sources,
            "raw_responses": {
                "llm": llm_raw,
                "keyword": {
                    "label": (
                        "Positive"
                        if keyword_score > 0.6
                        else "Negative" if keyword_score < 0.4 else "Neutral"
                    ),
                    "score": round(keyword_score, 3),
                    "matches": kw.get("matches", {}),
                    "raw_counts": kw.get("raw_counts", {}),
                },
            },
        }
    except Exception as e:
        return {
            "label": "Neutral",
            "score": 0.5,
            "emoji": _emoji("Neutral"),
            "sources": [],
            "raw_responses": {"error": str(e), "llm": llm_raw},
        }


# quick manual test if run directly
if __name__ == "__main__":
    print(
        compute_sentiment(
            "Company reported profit and revenue growth and approved a dividend."
        )
    )
