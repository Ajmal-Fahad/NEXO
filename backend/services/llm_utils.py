#!/usr/bin/env python3
"""
services/llm_utils.py

Robust LLM helpers for:
 - Headline + summary extraction
 - Sentiment classification

Supports both:
 - new OpenAI client (from openai import OpenAI)
 - legacy openai package (openai.ChatCompletion.create)

Output is JSON-serializable (usage objects are converted safely).
"""
from __future__ import annotations
import os
import json
import logging
import traceback
from typing import Optional, Dict, Any

# Try to support both old and new OpenAI python clients.
_OPENAI_CLIENT_AVAILABLE = False
OpenAIClient = None
openai = None
try:
    # new-style client (openai>=1.0)
    from openai import OpenAI as OpenAIClient  # type: ignore
    _OPENAI_CLIENT_AVAILABLE = True
except Exception:
    try:
        import openai  # type: ignore
    except Exception:
        openai = None  # type: ignore

logger = logging.getLogger("llm_utils")

# Model settings (can be overridden by env var)
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")

# ---------- JSON extraction helpers ----------
try:
    # prefer 'regex' package if available for recursive matching
    import regex as re_ex  # type: ignore
    _BARE_JSON_RE = re_ex.compile(r"(\{(?:[^{}]|(?R))*\})", re_ex.S)

    def extract_json_block(text: str) -> Optional[str]:
        if not text:
            return None
        m = _BARE_JSON_RE.search(text)
        return m.group(1) if m else None
except Exception:
    import re

    def extract_json_block(text: str) -> Optional[str]:
        """Naive fallback: find first {...} block."""
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        return None


def _safe_str(o: Any) -> str:
    """Convert arbitrary object to string safely for logging/serialization."""
    try:
        return str(o)
    except Exception:
        return repr(o)


def _serialize_usage_field(usage_obj: Any) -> Optional[Dict[str, Any]]:
    """
    Convert various 'usage' shapes (dict, attr object like CompletionUsage) to a plain dict.
    Returns None when usage is not present.
    """
    if usage_obj is None:
        return None
    # If already a dict-like
    if isinstance(usage_obj, dict):
        # ensure all keys map to JSON-safe simple types
        out = {}
        for k, v in usage_obj.items():
            try:
                json.dumps(v)
                out[k] = v
            except Exception:
                out[k] = _safe_str(v)
        return out
    # Try attribute access (new client usage objects)
    out = {}
    try:
        # common keys to look for
        keys = ["prompt_tokens", "completion_tokens", "total_tokens",
                "prompt_tokens_details", "completion_tokens_details"]
        for k in keys:
            val = getattr(usage_obj, k, None)
            if val is None:
                # try dictionary-like access if supported
                try:
                    val = usage_obj.get(k)  # type: ignore
                except Exception:
                    val = None
            if val is not None:
                try:
                    json.dumps(val)
                    out[k] = val
                except Exception:
                    out[k] = _safe_str(val)
        # Fallback: capture any attributes on the object
        if not out:
            for attr in dir(usage_obj):
                if attr.startswith("_"):
                    continue
                try:
                    v = getattr(usage_obj, attr)
                    # skip methods
                    if callable(v):
                        continue
                    try:
                        json.dumps(v)
                        out[attr] = v
                    except Exception:
                        out[attr] = _safe_str(v)
                except Exception:
                    continue
    except Exception:
        return {"usage_raw": _safe_str(usage_obj)}
    return out or {"usage_raw": _safe_str(usage_obj)}

# ---------- Core OpenAI caller ----------
def call_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 300,
    temperature: float = 0.2,
    retries: int = 2,
) -> Dict[str, Any]:
    """
    Call OpenAI (supports both old and new Python clients).
    Returns dict with { "ok": bool, "resp": <raw response> } on success
    or { "ok": False, "error": "<message>" } on failure.
    """
    last_err = None
    for attempt in range(retries):
        try:
            if _OPENAI_CLIENT_AVAILABLE and OpenAIClient is not None:
                # Explicitly pass API key for new OpenAI SDK (fixes "api_key must be set" error)
                client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
                # new client API (chat completions)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                # fallback to older openai package API
                if openai is None:
                    raise RuntimeError("No OpenAI client available (install openai package).")
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            return {"ok": True, "resp": resp}
        except Exception as e:
            last_err = _safe_str(e)
            logger.warning("call_llm failed (attempt %d/%d): %s", attempt + 1, retries, last_err)
    return {"ok": False, "error": last_err}

# ---------- Headline + Summary ----------
def classify_headline_and_summary(text: str) -> Dict[str, Any]:
    """
    Ask LLM for JSON { "headline": ..., "summary_60": ... }.

    Returns a dict:
      {
        "headline_final": <str or None>,
        "summary_60": <str or None>,
        "headline_ai": <original headline from LLM or None>,
        "headline_raw": <raw content if any>,
        "llm_meta": { "ok": bool, "model": str|None, "usage": {...}, "response_text": str }
      }
    """
    if not text:
        return {
            "headline_final": None,
            "summary_60": None,
            "headline_ai": None,
            "headline_raw": None,
            "llm_meta": {"ok": False, "reason": "empty_text"},
        }

    prompt = (
        "You are a professional financial news editor. "
        "Reframe the headline under 20 words with corrected grammar, punctuation, and natural phrasing. "
        "Ensure the headline reads like a real newspaper or article headline — complete and senseful, not a fragment. "
        "Do NOT end mid-sentence, with ellipses, or with dangling words like 'for', 'of', or 'to'. "
        "Write a crisp, complete and grammatically correct headline of no more than 20 words. "
        "Always end with a full stop. "
        "Use an action verb and include the main subject. Do NOT truncate sentences, do NOT end with ellipses, and DO end the headline with a single period. "
        "Then write a concise 60-70 word summary focusing on what happened, why it matters, and who is involved. "
        "Exclude addresses, greetings, signatures, or disclaimers. "
        "Return JSON exactly with fields 'headline' and 'summary_60'."
        f"\n\n{text[:4000]}"
    )

    raw = call_llm(prompt)
    if not raw.get("ok"):
        return {
            "headline_final": None,
            "summary_60": None,
            "headline_ai": None,
            "headline_raw": None,
            "llm_meta": {"ok": False, "error": raw.get("error")},
        }

    resp = raw["resp"]
    # defensive access to response text (works with both clients)
    try:
        resp_text = None
        usage_obj = None
        model_used = None

        if isinstance(resp, dict):
            # dictionary-like response (older client or converted)
            choices = resp.get("choices") or []
            if choices:
                first = choices[0]
                # older client: first.get("message", {}).get("content")
                if isinstance(first, dict):
                    msg = first.get("message") or {}
                    resp_text = msg.get("content") or first.get("text") or ""
                else:
                    # unexpected nested shape
                    resp_text = first.get("text") if isinstance(first, dict) else str(first)
            usage_obj = resp.get("usage") if resp.get("usage") is not None else None
            model_used = resp.get("model") if resp.get("model") is not None else None
        else:
            # object-like resp (new client)
            try:
                choices = getattr(resp, "choices", None)
                if choices:
                    first = choices[0]
                    # message may be attribute or dict-like
                    msg = getattr(first, "message", None) or (first.get("message") if isinstance(first, dict) else None)
                    resp_text = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
                    if resp_text is None:
                        # older style 'text' on first
                        resp_text = getattr(first, "text", None) or (first.get("text") if isinstance(first, dict) else None)
                usage_obj = getattr(resp, "usage", None)
                model_used = getattr(resp, "model", None)
            except Exception:
                resp_text = None

        # final fallback: stringify whole resp
        if resp_text is None:
            try:
                resp_text = json.dumps(resp) if not isinstance(resp, str) else str(resp)
            except Exception:
                resp_text = _safe_str(resp)

        serialized_usage = _serialize_usage_field(usage_obj)
    except Exception as e:
        resp_text = _safe_str(resp)
        serialized_usage = {"error": "usage_serialization_failed", "detail": _safe_str(e)}
        model_used = None

    # extract JSON block and parse
    block = extract_json_block(resp_text or "")
    try:
        data = json.loads(block) if block else {}
    except Exception:
        data = {}

    headline_val = (data.get("headline") or "").strip() or None
    summary_val = (data.get("summary_60") or "").strip() or None

    return {
        "headline_final": headline_val,
        "summary_60": summary_val,
        "headline_ai": data.get("headline"),
        "headline_raw": None,
        "llm_meta": {
            "ok": True,
            "model": model_used,
            "usage": serialized_usage,
            "response_text": resp_text,
        },
    }

# ---------- Sentiment ----------
def classify_sentiment(text: str) -> Dict[str, Any]:
    """
    Ask LLM for JSON { "label": Positive|Negative|Neutral, "score": float }
    """
    if not text:
        return {"ok": False, "label": "Unknown", "score": 0.0, "reason": "empty_text"}

    prompt = (
        "Classify sentiment of this corporate disclosure as Positive, Negative, or Neutral. "
        "Respond in JSON with fields 'label' and 'score' (0=negative, 1=positive, neutral≈0.5)."
        f"\n\n{text[:4000]}"
    )

    raw = call_llm(prompt)
    if not raw.get("ok"):
        return {"ok": False, "label": "Unknown", "score": 0.0, "error": raw.get("error")}

    resp = raw["resp"]
    try:
        resp_text = None
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
            if choices:
                m = choices[0].get("message") or {}
                resp_text = m.get("content") or ""
        else:
            choices = getattr(resp, "choices", None)
            if choices:
                first = choices[0]
                msg = getattr(first, "message", None) or (first.get("message") if isinstance(first, dict) else None)
                resp_text = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
        if resp_text is None:
            resp_text = json.dumps(resp) if not isinstance(resp, str) else str(resp)
    except Exception:
        resp_text = _safe_str(resp)

    block = extract_json_block(resp_text)
    try:
        data = json.loads(block) if block else {}
    except Exception:
        data = {}

    label = data.get("label", "Unknown")
    try:
        score = float(data.get("score") or 0.0)
    except Exception:
        score = 0.0

    return {
        "ok": True,
        "model": (resp.get("model") if isinstance(resp, dict) else getattr(resp, "model", None)),
        "label": label,
        "score": score,
        "raw_response": resp_text,
    }

# ---------- If run standalone ----------
if __name__ == "__main__":
    sample = "Lemon Tree Hotels signed a franchise agreement for a new property in Maharashtra."
    print("Import check. OpenAI client available:", _OPENAI_CLIENT_AVAILABLE)
    print("Headline+Summary:", classify_headline_and_summary(sample))
    print("Sentiment:", classify_sentiment(sample))
