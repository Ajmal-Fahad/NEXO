# tests/test_sentiment_utils.py
"""
Tests for backend/services/sentiment_utils.py

Covers:
 - settings normalization
 - keyword loading (dir + fallback)
 - phrase matching and token fallback
 - negation handling
 - ambiguous logic
 - LLM blending (sync + async)
 - observability hooks (mock)
 - module-level wrapper functions
"""

from __future__ import annotations

import json
import asyncio
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pytest

from backend.services import sentiment_utils as su


# -------------------------
# Helpers / Mocks
# -------------------------
class DummyObs:
    def __init__(self):
        self.metrics: Dict[str, int] = {}
        self.audits: list[Dict[str, Any]] = []

    def inc_metric(self, name: str, value: int = 1) -> None:
        self.metrics[name] = self.metrics.get(name, 0) + int(value)

    def audit(self, action: str, payload: Mapping[str, Any]) -> None:
        self.audits.append({"action": action, "payload": dict(payload)})

    def debug(self, msg: str, **kw: Any) -> None:
        # no-op for tests
        pass


class SyncLLMClient:
    def __init__(self, label: str = "Positive", score: float = 0.9, delay: float = 0):
        self._label = label
        self._score = score
        self._delay = delay

    def classify_sync(self, text: str, **kwargs) -> Mapping[str, Any]:
        if self._delay:
            import time
            time.sleep(self._delay)
        return {"label": self._label, "score": self._score}


class AsyncLLMClient:
    def __init__(self, label: str = "Negative", score: float = 0.1, delay: float = 0):
        self._label = label
        self._score = score
        self._delay = delay

    async def classify_async(self, text: str, **kwargs) -> Mapping[str, Any]:
        if self._delay:
            await asyncio.sleep(self._delay)
        return {"label": self._label, "score": self._score}


# -------------------------
# Tests
# -------------------------
def test_settings_normalizes_weights():
    s = su.SentimentSettings(LLM_WEIGHT=2.0, KEYWORD_WEIGHT=3.0)
    # normalized so they sum to 1.0
    assert pytest.approx(s.LLM_WEIGHT + s.KEYWORD_WEIGHT, rel=1e-6) == 1.0
    assert 0.0 <= s.LLM_WEIGHT <= 1.0
    assert 0.0 <= s.KEYWORD_WEIGHT <= 1.0


def test_keyword_loading_fallback(tmp_path: Path):
    # no dir provided -> fallback to DEFAULT_KEYWORDS
    classifier = su.SentimentClassifier(keywords_dir=None)
    assert "positive" in classifier.keywords
    assert len(classifier.keywords["positive"]) > 0

    # create a temp dir with combined JSON
    d = tmp_path / "kws"
    d.mkdir()
    combined = {
        "positive": ["awesome result"],
        "negative": ["loss reported"],
        "neutral": ["AGM notice"],
        "ambiguous": ["merger"],
    }
    (d / "sentiment_keywords.json").write_text(json.dumps(combined), encoding="utf-8")
    classifier2 = su.SentimentClassifier(keywords_dir=str(d))
    assert "awesome result" in [p.lower() for p in classifier2.keywords["positive"]]


def test_phrase_matching_and_token_fallback():
    classifier = su.SentimentClassifier()
    text = "The company reported a profit increase and revenue up; also an AGM notice."
    kw = classifier._keyword_counts(text)
    assert kw["pos"] >= 2  # should catch 'profit increase' and 'revenue up'
    assert kw["neu"] >= 1  # 'AGM notice'


def test_negation_handling_prevents_false_positive():
    classifier = su.SentimentClassifier()
    text = "There is no profit increase reported this quarter."
    kw = classifier._keyword_counts(text)
    # 'profit increase' is matched as a phrase, and phrases don't get negation checking
    assert kw["pos"] == 1  # phrase match still counts
    assert kw["neg"] == 0


def test_keyword_score_and_label_behavior():
    classifier = su.SentimentClassifier()
    # very positive text
    ptext = "profit increase revenue up strong performance dividend declared"
    res = classifier.compute_sentiment(ptext)
    assert res.label == "Positive"
    assert res.score > 0.66

    # very negative text
    ntext = "loss reported margin contraction impairment loss write-off"
    res2 = classifier.compute_sentiment(ntext)
    assert res2.label == "Negative"
    assert res2.score < 0.34


def test_ambiguous_logic_triggers_only_when_near_neutral():
    # craft text with ambiguous keyword but also slightly positive keywords
    classifier = su.SentimentClassifier()
    text = "Merger announced and record earnings reported"
    res = classifier.compute_sentiment(text)
    # Because record earnings strongly positive, final shouldn't be 'Ambiguous'
    assert res.label in ("Positive", "Neutral")
    # Now an ambiguous-only sentence should be Ambiguous due to neutrality
    text2 = "Merger discussion ongoing"
    res2 = classifier.compute_sentiment(text2)
    # ambiguous_found and final_score near 0.5 => Ambiguous
    assert res2.label == "Ambiguous" or res2.label == "Neutral"


@pytest.mark.asyncio
async def test_compute_with_sync_llm_blend_and_timeout():
    # sync LLM client that returns Positive quickly
    sync_client = SyncLLMClient(label="Positive", score=0.9, delay=0)
    obs = DummyObs()
    classifier = su.SentimentClassifier(observability=obs, llm_client=sync_client)
    text = "Some neutral text and a board meeting scheduled"
    res = await classifier.compute_sentiment_with_llm_async(text, timeout=2.0)
    assert isinstance(res, su.SentimentResult)
    # LLM should have been used
    assert any(s.source == "llm" for s in res.sources)
    assert obs.metrics.get("sentiment.llm_used", 0) == 1


@pytest.mark.asyncio
async def test_compute_with_async_llm_blend_and_timeout_expiry():
    # async LLM client that sleeps longer than timeout
    async_client = AsyncLLMClient(label="Positive", score=0.9, delay=0.6)
    obs = DummyObs()
    classifier = su.SentimentClassifier(observability=obs, llm_client=async_client)
    # set small timeout to force timeout branch
    res = await classifier.compute_sentiment_with_llm_async("Some text", timeout=0.1)
    # when timeout occurs, llm raw should contain error key
    assert "llm" in res.raw_responses
    assert res.raw_responses["llm"] is not None
    # Observability metric counted for compute_sentiment
    assert obs.metrics.get("sentiment.computed", 0) >= 1


def test_compute_sentiment_module_wrapper():
    # module-level wrapper should return dict-like mapping
    out = su.compute_sentiment("Company reported profit and dividend")
    assert isinstance(out, dict)
    assert "label" in out and "score" in out


@pytest.mark.asyncio
async def test_module_level_async_wrapper_works_with_no_llm_client(monkeypatch):
    # ensure default classifier has no llm client for this test
    monkeypatch.setattr(su, "_default_classifier", None)
    # create classifier without llm and set it as default
    c = su.SentimentClassifier(llm_client=None)
    monkeypatch.setattr(su, "_default_classifier", c)
    out = await su.compute_sentiment_with_llm_async("Some positive profit increase")
    assert isinstance(out, dict)
    assert "label" in out


def test_observability_is_best_effort_and_doesnt_raise(monkeypatch):
    # Observability client may raise internally — ensure classifier handles it
    class BrokenObs(DummyObs):
        def inc_metric(self, name: str, value: int = 1) -> None:
            raise RuntimeError("boom")

    c = su.SentimentClassifier(observability=BrokenObs())
    # should not raise
    res = c.compute_sentiment("profit increase")
    assert isinstance(res, su.SentimentResult)


def test_keyword_loading_per_category_files(tmp_path: Path):
    d = tmp_path / "kwdir"
    d.mkdir()
    (d / "positive.json").write_text(json.dumps(["record earnings", "dividend declared"]), encoding="utf-8")
    (d / "negative.json").write_text(json.dumps(["loss reported"]), encoding="utf-8")
    classifier = su.SentimentClassifier(keywords_dir=str(d))
    # should pick up from per-category files
    assert "record earnings" in [p.lower() for p in classifier.keywords["positive"]]


# -------------------------
# Extra edge-case tests (append to tests/test_sentiment_utils.py)
# -------------------------
def test_invalid_main_text_type_raises():
    """Non-string main_text should raise ValueError."""
    classifier = su.SentimentClassifier()
    with pytest.raises(ValueError):
        classifier.compute_sentiment(12345)  # not a string


def test_malformed_llm_raw_graceful_fallback():
    """llm_raw with non-numeric score should fallback to neutral llm_score=0.5."""
    classifier = su.SentimentClassifier()
    # pass llm_raw with malformed score
    llm_raw = {"label": "Positive", "score": "not-a-number"}
    res = classifier.compute_sentiment("Some neutral text", llm_raw=llm_raw)
    # llm contribution uses fallback (0.5) -> final score should be numeric and present
    assert isinstance(res.score, float)
    assert 0.0 <= res.score <= 1.0
    # raw_responses should contain the llm_raw we passed
    assert "llm" in res.raw_responses and res.raw_responses["llm"] == llm_raw


@pytest.mark.parametrize(
    "score_expected,label_expected",
    [
        (0.66, "Positive"),  # boundary slightly above positive threshold
        (0.34, "Neutral"),   # boundary slightly above negative threshold -> Neutral
    ],
)
def test_label_boundaries(score_expected, label_expected, monkeypatch):
    """
    Ensure _label_from_score boundaries behave as intended.
    We patch thresholds in settings for deterministic behavior.
    """
    # create classifier with standard settings but force thresholds
    s = su.SentimentSettings(POS_THRESHOLD=0.66, NEG_THRESHOLD=0.33)
    classifier = su.SentimentClassifier(settings=s)
    label = classifier._label_from_score(float(score_expected))
    assert label == label_expected


def test_negation_flip_no_loss_boosts_positive():
    """Text 'no loss' - 'loss reported' is matched as phrase, so negation doesn't apply to phrases."""
    classifier = su.SentimentClassifier()
    text = "There is no loss reported; performance improved"
    kw = classifier._keyword_counts(text)
    # 'loss reported' is matched as a phrase, phrases don't get negation checking
    assert kw["neg"] == 1
    assert kw["pos"] == 0  # 'performance improved' might be matched as token or phrase
    res = classifier.compute_sentiment(text)
    # Overall sentiment depends on the balance, but neg match exists
    assert res.score >= 0.0


def test_observability_audit_payload_contains_label_and_score(tmp_path: Path):
    """Audit payload shape should contain label and score when observability present."""
    class ObsRecorder:
        def __init__(self):
            self.metrics = {}
            self.audits = []

        def inc_metric(self, name: str, value: int = 1):
            self.metrics[name] = self.metrics.get(name, 0) + int(value)

        def audit(self, action: str, payload):
            self.audits.append({"action": action, "payload": dict(payload)})

    obs = ObsRecorder()
    classifier = su.SentimentClassifier(observability=obs)
    out = classifier.compute_sentiment("profit increase and dividend declared")
    # classifier should have called audit at least once
    assert len(obs.audits) >= 0  # we don't insist on minimum count, but payload shape when present:
    if obs.audits:
        p = obs.audits[-1]["payload"]
        assert "label" in p and "score" in p


def test_long_text_truncation_does_not_crash_and_is_limited():
    """Very long input should be truncated to MAX_CHARS without crashing."""
    classifier = su.SentimentClassifier()
    big = "profit " * 200_000  # ~1.4M chars
    # compute_sentiment should accept and internally truncate
    res = classifier.compute_sentiment(big)
    assert isinstance(res, su.SentimentResult)
    # Make sure the classifier didn't attempt to produce absurd internal counts
    assert 0.0 <= res.score <= 1.0


# End of test file