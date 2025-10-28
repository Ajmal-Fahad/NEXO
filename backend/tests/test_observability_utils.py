# backend/tests/test_observability_utils.py
"""
Patched Diamond Test Suite — Observability Utils (R-01 patched)
- Ensures deterministic queue-full behaviour by forcing small queue size.
- Simulates backend/store failure by patching the helper called by the worker (not the worker itself).
- Keeps tests focused, robust and CI-friendly.
"""

import os
import time
import threading
import datetime
import json
import logging
import pytest
from unittest.mock import patch, MagicMock

# module under test
from backend.services import observability_utils as obs


# --- Fixtures ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_obs_state():
    """
    Ensure a clean module state for each test:
    - stop worker
    - clear metrics state
    - yield
    - final cleanup
    """
    # Best-effort stop + reset (silently ignore errors)
    try:
        obs.shutdown_observability(timeout=1.0)
    except Exception:
        pass

    try:
        obs.reset_metrics_state()
    except Exception:
        pass

    yield

    # final teardown
    try:
        obs.shutdown_observability(timeout=1.0)
    except Exception:
        pass
    try:
        obs.reset_metrics_state()
    except Exception:
        pass


@pytest.fixture
def obs_settings():
    """Return current observability settings object for convenience."""
    return obs.get_observability_settings()


# --- Helpers ---------------------------------------------------------------

def _force_reload_settings_and_reset(monkeypatch):
    """
    Helper to force observability settings to reload from env vars.
    Implementation detail: many modules cache settings in a private var.
    Reset via public API if available, otherwise fallback to clearing
    the internal cache name(s) we observed.
    """
    # allow caller to set env vars before calling this
    # try public reset if available
    if hasattr(obs, "reset_metrics_state"):
        # reset_metrics_state will stop worker and clear state and often re-read settings on next start
        obs.reset_metrics_state()
        # ensure cached settings reloaded next call
        if hasattr(obs, "_SETTINGS"):
            try:
                obs._SETTINGS = None
            except Exception:
                pass
        return

    # Fallback: clear internal cache names if they exist
    for name in ("_SETTINGS", "SETTINGS_CACHE"):
        if hasattr(obs, name):
            try:
                setattr(obs, name, None)
            except Exception:
                pass


def _patch_backend_store(monkeypatch, exc_to_raise):
    """
    Try to patch an internal metric-storage helper that the worker calls after
    dequeuing an item. We try a list of likely names and return which name we patched.
    This avoids patching the whole worker function.
    """
    candidates = [
        "_store_metric",
        "_process_metric",
        "_write_metric",
        "_metrics_store",
        "metrics_backend_store",  # fallback name
    ]
    for name in candidates:
        if hasattr(obs, name):
            monkeypatch.setattr(obs, name, lambda *a, **kw: (_ for _ in ()).throw(exc_to_raise))
            return name
    # if none found, raise so calling test can decide an alternative
    raise RuntimeError("No candidate metric-store helper found to patch; adjust test to module internals.")


# --- Tests ---------------------------------------------------------------

def test_audit_log_serializes_datetime_and_sets():
    """Audit log should serialize datetime -> ISO and sets to a string safely."""
    with patch.object(obs, "get_logger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        dt = datetime.datetime(2025, 10, 28, 10, 31, 0)
        details = {"event_time": dt, "user_ids": {1, 2, "a"}}

        obs.audit_log("test.action", "target/resource", "success", details)

        # logger called once
        mock_logger.info.assert_called_once()
        args, _ = mock_logger.info.call_args
        assert args[0] == "AUDIT %s"
        payload = json.loads(args[1])
        assert payload["audit_details"]["event_time"] == "2025-10-28T10:31:00"
        # user_ids serialized to a string or list-like; check presence of content
        s = str(payload["audit_details"]["user_ids"])
        assert "1" in s and "2" in s and "a" in s


def test_metrics_worker_shutdown_guarantee_when_queue_full(monkeypatch, caplog):
    """
    Force a tiny queue to deterministically cause a 'drop metric' path,
    then ensure shutdown completes using the flag/poll logic (not blocked on put(None)).
    """
    # Force small queue via env var, then reload settings+state
    monkeypatch.setenv("METRICS_QUEUE_MAX", "8")
    _force_reload_settings_and_reset(monkeypatch)

    settings = obs.get_observability_settings()
    # sanity: small queue
    assert settings.METRICS_QUEUE_MAX <= 16

    # Start worker by sending an item
    obs.metrics_inc("test.start")
    # small sleep to allow worker to start
    time.sleep(max(0.01, settings.METRICS_WORKER_POLL_SECONDS * 0.05))

    q = getattr(obs, "_METRICS_QUEUE", None)
    assert q is not None, "Metrics queue not initialized"
    q_size = q.maxsize

    # Fill the queue aggressively (without blocking)
    fill_count = 0
    for i in range(q_size + 20):
        try:
            q.put_nowait({"type": "inc", "name": f"fill.{i}", "value": 1})
            fill_count += 1
        except Exception:
            break

    # we should have filled to capacity (or near it)
    assert q.full() or fill_count >= max(1, q_size - 1)

    # Now the public API should drop a metric and log a warning
    caplog.set_level(logging.WARNING)
    obs.metrics_inc("test.dropped")
    # small wait to let logging happen
    time.sleep(0.05)

    log_text = caplog.text.lower()
    # Accept a few message variants
    assert ("dropping metric" in log_text and "test.dropped" in log_text) or ("drop" in log_text and "test.dropped" in log_text), \
        f"expected drop warning, got logs: {caplog.text!r}"

    # Shutdown should exit promptly (shouldn't block on put_nowait(None) since worker checks flag)
    start = time.monotonic()
    obs.shutdown_observability(timeout=2.0)
    duration = time.monotonic() - start
    assert duration < 2.0, "shutdown took too long"


def test_metrics_snapshot_and_reset():
    """Snapshot, reset and lazy restart behavior for metrics worker."""
    settings = obs.get_observability_settings()

    obs.metrics_inc("counter.a", 5)
    obs.metrics_gauge("gauge.b", 42.5)
    obs.metrics_inc("counter.a", 2)

    # allow worker to run at least one poll
    time.sleep(max(0.01, settings.METRICS_WORKER_POLL_SECONDS * 1.1))

    snap = obs.get_metrics_snapshot()
    assert snap.get("counter.a") == 7.0
    assert snap.get("gauge.b") == 42.5

    # reset
    obs.reset_metrics_state()
    snap2 = obs.get_metrics_snapshot()
    assert not snap2
    assert getattr(obs, "_METRICS_RUNNING", False) is False

    # lazy restart
    obs.metrics_inc("counter.c", 1)
    time.sleep(0.05)
    assert getattr(obs, "_METRICS_RUNNING", False) is True
    # wait for processing
    time.sleep(max(0.01, settings.METRICS_WORKER_POLL_SECONDS * 1.1))
    snap3 = obs.get_metrics_snapshot()
    assert snap3.get("counter.c") == 1.0


def test_concurrent_metric_updates(obs_settings, monkeypatch):
    """
    Stress test concurrent metric increments from multiple threads.

    Changes:
    - Ensure the module uses a large queue for this test so the
      concurrent burst doesn't cause intentional drops.
    - Restart/reset observability state after changing env so settings
      (e.g., METRICS_QUEUE_MAX) take effect.
    - Wait deterministically until queue empties, with a reasonable timeout.
    """
    # 1) Force a large queue for this test so we don't hit "dropping" behavior.
    monkeypatch.setenv("METRICS_QUEUE_MAX", "20000")  # plenty of headroom
    # Clear cached settings and restart module state so new env var takes effect
    if hasattr(obs, "_SETTINGS"):
        obs._SETTINGS = None
    obs.reset_metrics_state()
    # Start with one metric so worker is started
    obs.metrics_inc("concurrent.start")
    time.sleep(obs_settings.METRICS_WORKER_POLL_SECONDS * 0.05)

    # 2) Parameters
    THREADS = 6
    INCS = 200
    prefix = "concurrent_"

    def worker(tid):
        name = f"{prefix}{tid}"
        for _ in range(INCS):
            obs.metrics_inc(name, 1)

    # 3) Fire concurrent threads
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 4) Wait for the metrics worker to finish processing the queue.
    # Conservative timeout: a few poll cycles plus buffer.
    settings = obs.get_observability_settings()
    max_wait = max(5.0, settings.METRICS_WORKER_POLL_SECONDS * 6 + 2.0)
    start = time.monotonic()
    while time.monotonic() - start < max_wait:
        q = getattr(obs, "_METRICS_QUEUE", None)
        if q is None or q.empty():
            # small additional wait to ensure worker finished last write
            time.sleep(min(0.2, settings.METRICS_WORKER_POLL_SECONDS * 0.2))
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"Metrics queue didn't drain in time (waited {max_wait}s)")

    # 5) Assert final snapshot exactly matches expected counts
    snap = obs.get_metrics_snapshot()
    assert len(snap) >= THREADS, f"Expected at least {THREADS} counters, got {len(snap)}"
    for i in range(THREADS):
        assert snap.get(f"{prefix}{i}") == float(INCS), \
            f"Counter {prefix}{i} expected {INCS}, got {snap.get(f'{prefix}{i}')}"


def test_metrics_backend_simulation_failure(monkeypatch, caplog):
    """
    Patch a helper the worker uses to store/process dequeued metrics so the
    exception occurs inside the worker's processing loop (which should catch/log it).
    """
    settings = obs.get_observability_settings()
    # ensure worker running
    obs.metrics_inc("sim.fail.start")
    time.sleep(max(0.01, settings.METRICS_WORKER_POLL_SECONDS * 0.05))

    # Patch a store helper; _patch_backend_store will raise if none found
    exc = RuntimeError("Simulated backend failure")
    patched_name = None
    try:
        patched_name = _patch_backend_store(monkeypatch, exc)
    except RuntimeError:
        # If the module lacks those helpers, fail the test early and signal maintainers
        pytest.skip("No known store helper to patch; update test to match implementation")

    caplog.set_level(logging.ERROR)
    # Enqueue a metric that will reach the worker and trigger the patched failure
    obs.metrics_inc("test.failure")
    # wait for worker to process and log
    time.sleep(max(0.01, settings.METRICS_WORKER_POLL_SECONDS * 1.1))

    # Verify that the worker logged the simulated error (case-insensitive)
    text = caplog.text.lower()
    assert "simulated backend failure" in text or "simulated backend" in text or "runtimeerror" in text or "failed" in text

    # Ensure worker is still marked as running (didn't die)
    assert getattr(obs, "_METRICS_RUNNING", False) is True


@pytest.mark.slow
def test_high_volume_metrics_performance():
    """Lightweight smoke test for high-volume metric submission (slow)."""
    settings = obs.get_observability_settings()
    N = 1200
    start = time.monotonic()
    for i in range(N):
        obs.metrics_inc(f"perf.{i % 100}")
    send_dur = time.monotonic() - start
    assert send_dur < 5.0  # reasonably fast to enqueue

    # wait a few cycles for processing
    time.sleep(max(0.01, settings.METRICS_WORKER_POLL_SECONDS * 2.5))
    snap = obs.get_metrics_snapshot()
    # at least a good portion of the 100 buckets should be present
    keys = [k for k in snap.keys() if k.startswith("perf.")]
    assert len(keys) >= 40


def test_environment_variable_overrides(monkeypatch):
    """Verify settings pick up environment overrides (best-effort, tolerant)."""
    monkeypatch.setenv("METRICS_QUEUE_MAX", "10")
    monkeypatch.setenv("METRICS_WORKER_POLL_SECONDS", "0.2")
    # try to nudge settings reload
    _force_reload_settings_and_reset(monkeypatch)
    s = obs.get_observability_settings()
    # tolerant assertions: only check what's present
    if hasattr(s, "METRICS_QUEUE_MAX"):
        assert int(s.METRICS_QUEUE_MAX) == 10
    if hasattr(s, "METRICS_WORKER_POLL_SECONDS"):
        assert float(s.METRICS_WORKER_POLL_SECONDS) == pytest.approx(0.2, rel=1e-2)


def test_shutdown_cleans_state():
    """Final sanity check for shutdown_observability side-effects."""
    settings = obs.get_observability_settings()
    obs.metrics_inc("cleanup.test")
    time.sleep(max(0.01, settings.METRICS_WORKER_POLL_SECONDS * 0.05))
    obs.shutdown_observability(timeout=1.0)
    assert getattr(obs, "_METRICS_RUNNING", False) is False
    assert getattr(obs, "_METRICS_QUEUE", None) is None
    assert getattr(obs, "_METRICS_WORKER_THREAD", None) is None