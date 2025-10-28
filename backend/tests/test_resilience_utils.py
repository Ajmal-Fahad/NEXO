# tests/test_resilience_utils.py
"""
Diamond Test Suite — Resilience Utils (R-03 Final - Diamond Certified)
-----------------------------------------------------------------------
Validates:
- _is_transient_error heuristic accuracy.
- retry_sync / retry_async logic (success, exhaustion, backoff, non-transient handling).
- CircuitBreaker state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED/OPEN) using public API.
- CircuitBreaker decorator behavior under various conditions.
- Global registry thread-safety and management (get, reset, list).
- Observability fallback safety (ensuring no crashes when observability is absent).
- Correct integration between retry helpers and the CircuitBreaker.
- Input validation for retry function parameters (e.g., negative retries).
- Uses conservative timing margins to reduce CI flakiness.
"""

import asyncio
import concurrent.futures # For concurrency test
import logging
import threading
import socket
import time
from unittest import mock

import pytest

# Import the module under test
# Assuming the tests directory is sibling to the 'backend' directory or PYTHONPATH is set
import backend.services.resilience_utils as r


# --- Fixtures ---

@pytest.fixture(autouse=True)
def ensure_clean_state(monkeypatch):
    """
    Fixture ensures a clean state before each test:
    1. Resets the global circuit breaker registry.
    2. Forces the observability fallback path by patching internal flags.
    3. Restores original observability flags after the test.
    """
    # Store original observability state
    orig_obs_available = getattr(r, "_OBSERVABILITY_AVAILABLE", None)
    orig_obs_utils = getattr(r, "observability_utils", None)

    # 1. Reset circuit breakers before the test
    r.reset_all_circuit_breakers()
    # Ensure registry is empty after reset
    assert not r.list_circuit_breakers(), "Registry should be empty after reset"

    # 2. Force observability fallback by default for most tests
    monkeypatch.setattr(r, "_OBSERVABILITY_AVAILABLE", False, raising=False)
    monkeypatch.setattr(r, "observability_utils", None, raising=False)

    yield  # Run the test

    # 3. Reset circuit breakers again after the test (ensure cleanup)
    r.reset_all_circuit_breakers()
    assert not r.list_circuit_breakers(), "Registry should be empty after test cleanup"

    # 4. Restore original observability state
    if orig_obs_available is not None:
        monkeypatch.setattr(r, "_OBSERVABILITY_AVAILABLE", orig_obs_available, raising=False)
    if orig_obs_utils is not None:
        monkeypatch.setattr(r, "observability_utils", orig_obs_utils, raising=False)

def _sleep_margin() -> float:
    """
    Returns a conservative sleep margin (seconds) to add to timeouts in tests.
    Helps prevent flakiness in CI environments due to minor timing variations.
    """
    # Increased margin for potentially slower CI environments
    return 0.20 # 200ms

# --- Test _is_transient_error ---

@pytest.mark.parametrize("exc, expected", [
    (TimeoutError("connection timeout"), True),
    (ConnectionError("connect failed"), True),
    (socket.timeout("socket timed out"), True),
    (RuntimeError("temporary service unavailable 503"), True),
    (Exception("throttling"), True),
    (Exception("Rate limit exceeded"), True),
    (ConnectionRefusedError("refused"), True), # Often transient
    (ConnectionResetError("reset by peer"), True),
    (ValueError("invalid input"), False), # Non-transient
    (TypeError("bad type"), False),       # Non-transient
    (KeyError("missing key"), False),     # Non-transient
    (AttributeError("no attr"), False),   # Non-transient
    (FileNotFoundError("no file"), False), # Non-transient I/O error
])
def test_is_transient_error_heuristic(exc, expected):
    """Verify the transient error detection logic against various exception types."""
    assert r._is_transient_error(exc) is expected


# --- Test retry_sync ---

def test_retry_sync_success_first_try():
    """Test retry_sync succeeds immediately if the function doesn't raise."""
    calls = {"n": 0}
    def succeed():
        calls["n"] += 1
        return "ok"

    result = r.retry_sync(succeed, retries=3, backoff=0.01)
    assert result == "ok"
    assert calls["n"] == 1 # Should only be called once

def test_retry_sync_success_after_retries():
    """Test retry_sync succeeds after encountering transient failures within the retry limit."""
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("temporary timeout") # Simulate transient error
        return "ok"

    result = r.retry_sync(flaky, retries=3, backoff=0.01) # Allows 3 retries (4 attempts total)
    assert result == "ok"
    assert calls["n"] == 3 # Called 3 times (initial + 2 retries)

def test_retry_sync_failure_after_exhaustion():
    """Test retry_sync raises the last exception after all retry attempts fail."""
    calls = {"n": 0}
    def always_fail_transient():
        calls["n"] += 1
        raise ConnectionError("connection reset") # Simulate transient error

    with pytest.raises(ConnectionError):
        # retries=2 means 1 initial call + 2 retries = 3 total attempts
        r.retry_sync(always_fail_transient, retries=2, backoff=0.01)
    assert calls["n"] == 3 # Should have attempted 3 times

def test_retry_sync_fail_fast_on_non_transient():
    """Test retry_sync raises immediately on a non-transient error when raise_on_non_transient=True."""
    calls = {"n": 0}
    def fail_permanent():
        calls["n"] += 1
        raise ValueError("permanent config error") # Simulate non-transient error

    with pytest.raises(ValueError):
        r.retry_sync(fail_permanent, retries=3, backoff=0.01, raise_on_non_transient=True)
    assert calls["n"] == 1 # Should fail on the first attempt

def test_retry_sync_retries_non_transient_when_flag_is_false():
    """Test retry_sync retries even non-transient errors if raise_on_non_transient=False."""
    calls = {"n": 0}
    def fail_permanent():
        calls["n"] += 1
        raise ValueError("permanent config error") # Simulate non-transient error

    with pytest.raises(ValueError):
        # retries=1 means 1 initial call + 1 retry = 2 total attempts
        r.retry_sync(fail_permanent, retries=1, backoff=0.01, raise_on_non_transient=False)
    assert calls["n"] == 2 # Should retry once

def test_retry_sync_invalid_retries_value_raises():
    """Test ValueError is raised for negative retries."""
    with pytest.raises(ValueError, match="retries must be a non-negative integer"):
        r.retry_sync(lambda: "x", retries=-1, backoff=0)

def test_retry_sync_invalid_backoff_value_raises():
    """Test ValueError is raised for negative backoff."""
    with pytest.raises(ValueError, match="backoff must be a non-negative number"):
        r.retry_sync(lambda: "x", retries=1, backoff=-1.0)


# --- Test retry_async ---

@pytest.mark.asyncio
async def test_retry_async_success_first_try():
    """Test async success on the first attempt."""
    calls = {"n": 0}
    async def succeed_async():
        calls["n"] += 1
        await asyncio.sleep(0) # Yield control briefly
        return "ok_async"

    result = await r.retry_async(succeed_async, retries=3, backoff=0.01)
    assert result == "ok_async"
    assert calls["n"] == 1

@pytest.mark.asyncio
async def test_retry_async_success_after_retries():
    """Test async success after transient failures."""
    calls = {"n": 0}
    async def flaky_async():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("async temporary timeout")
        await asyncio.sleep(0)
        return "ok_async"

    result = await r.retry_async(flaky_async, retries=3, backoff=0.01)
    assert result == "ok_async"
    assert calls["n"] == 3

@pytest.mark.asyncio
async def test_retry_async_failure_after_exhaustion():
    """Test async failure when retries are exhausted."""
    calls = {"n": 0}
    async def always_fail_async():
        calls["n"] += 1
        await asyncio.sleep(0)
        raise ConnectionError("async connection reset")

    with pytest.raises(ConnectionError):
        # retries=2 means 3 total attempts
        await r.retry_async(always_fail_async, retries=2, backoff=0.01)
    assert calls["n"] == 3

@pytest.mark.asyncio
async def test_retry_async_fail_fast_on_non_transient():
    """Test async immediate failure on non-transient error."""
    calls = {"n": 0}
    async def fail_permanent_async():
        calls["n"] += 1
        await asyncio.sleep(0)
        raise ValueError("async permanent config error")

    with pytest.raises(ValueError):
        await r.retry_async(fail_permanent_async, retries=3, backoff=0.01, raise_on_non_transient=True)
    assert calls["n"] == 1

@pytest.mark.asyncio
async def test_retry_async_invalid_retries_value_raises():
    """Test ValueError is raised for negative retries in async."""
    async def dummy_coro(): return "x"
    with pytest.raises(ValueError, match="retries must be a non-negative integer"):
        # Correctly await the call to retry_async
        await r.retry_async(dummy_coro, retries=-1, backoff=0)

@pytest.mark.asyncio
async def test_retry_async_invalid_backoff_value_raises():
    """Test ValueError is raised for negative backoff in async."""
    async def dummy_coro(): return "x"
    with pytest.raises(ValueError, match="backoff must be a non-negative number"):
        # Correctly await the call to retry_async
        await r.retry_async(dummy_coro, retries=1, backoff=-1.0)


# --- Test CircuitBreaker Class ---

def test_circuit_breaker_initial_state():
    """Verify initial state is CLOSED and allows requests."""
    cb = r.CircuitBreaker(name="init_test")
    status = cb.status()
    assert status["state"] == "CLOSED"
    assert status["failure_count"] == 0
    assert status["last_failure_time"] is None
    assert cb.allow_request() is True

def test_circuit_breaker_trips_to_open():
    """Verify transition CLOSED -> OPEN on reaching failure threshold."""
    cb = r.CircuitBreaker(name="trip_test", failure_threshold=2, recovery_timeout=60)
    # Failure 1
    cb.record_failure(RuntimeError("fail1"))
    assert cb.status()["state"] == "CLOSED"
    assert cb.status()["failure_count"] == 1
    # Failure 2 (reaches threshold)
    cb.record_failure(RuntimeError("fail2"))
    status = cb.status()
    assert status["state"] == "OPEN"
    assert status["failure_count"] == 2
    assert cb.allow_request() is False # Should block requests now

def test_circuit_breaker_open_to_half_open_after_timeout():
    """Verify transition OPEN -> HALF_OPEN after recovery timeout passes."""
    timeout = 0.1 # Use a slightly larger timeout for reliability
    cb = r.CircuitBreaker(name="half_open_test", failure_threshold=1, recovery_timeout=timeout)
    # Trip to OPEN
    cb.record_failure(RuntimeError("fail"))
    status_open = cb.status()
    assert status_open["state"] == "OPEN"
    # Immediately after opening, it should block
    assert cb.allow_request() is False

    # Wait for timeout + margin
    time.sleep(timeout + _sleep_margin())

    # First call to allow_request *after* timeout should return True and set state to HALF_OPEN
    assert cb.allow_request() is True
    status_half_open = cb.status()
    assert status_half_open["state"] == "HALF_OPEN"
    # Subsequent calls while HALF_OPEN should also return True (allowing the probe to proceed)
    assert cb.allow_request() is True

def test_circuit_breaker_half_open_to_closed_on_success():
    """Verify transition HALF_OPEN -> CLOSED after a successful operation."""
    timeout = 0.1
    cb = r.CircuitBreaker(name="reset_test", failure_threshold=1, recovery_timeout=timeout)
    # Trip to OPEN
    cb.record_failure(RuntimeError("fail"))
    # Wait and transition to HALF_OPEN via allow_request
    time.sleep(timeout + _sleep_margin())
    assert cb.allow_request() is True # This moves state to HALF_OPEN
    assert cb.status()["state"] == "HALF_OPEN"

    # Record a success (simulating successful probe call)
    cb.record_success()
    status = cb.status()
    assert status["state"] == "CLOSED"
    assert status["failure_count"] == 0 # Failure count reset
    assert status["last_failure_time"] is None # Last failure info cleared
    assert cb.allow_request() is True # Should allow requests again

def test_circuit_breaker_half_open_back_to_open_on_failure():
    """Verify transition HALF_OPEN -> OPEN if the probe call fails."""
    timeout = 0.1
    cb = r.CircuitBreaker(name="reopen_test", failure_threshold=1, recovery_timeout=timeout)
    # Trip to OPEN
    cb.record_failure(RuntimeError("f1"))
    # Wait and transition to HALF_OPEN via allow_request
    time.sleep(timeout + _sleep_margin())
    assert cb.allow_request() is True # Moves state to HALF_OPEN
    assert cb.status()["state"] == "HALF_OPEN"

    # Record another failure (simulating failed probe call)
    cb.record_failure(RuntimeError("f2"))
    status = cb.status()
    assert status["state"] == "OPEN" # Should go back to OPEN
    # Failure count increments when moving HALF_OPEN -> OPEN
    assert status["failure_count"] >= 1 # Will be 1 (reset from first fail, then +1)
    # Check blocking immediately after reopening
    assert cb.allow_request() is False

def test_circuit_breaker_success_resets_count_in_closed():
    """Verify success in CLOSED state resets any intermittent failure count to 0."""
    cb = r.CircuitBreaker(name="closed_reset", failure_threshold=3)
    cb.record_failure(RuntimeError("glitch1"))
    assert cb.status()["failure_count"] == 1 # Intermittent failure
    cb.record_success() # Successful operation follows
    # Success should reset the *consecutive* count back to 0
    status = cb.status()
    assert status["failure_count"] == 0
    assert status["last_failure_time"] is None # Also clear time/exception
    assert status["state"] == "CLOSED" # Remains CLOSED

def test_circuit_breaker_decorator_and_flow():
    """Test using the breaker as a decorator and verify full state flow."""
    # Use a longer timeout to prevent race conditions in this test
    timeout = 0.5
    cb = r.CircuitBreaker(name="decorator_test", failure_threshold=1, recovery_timeout=timeout)
    calls = {"n": 0}

    @cb # Apply decorator
    def decorated_flaky():
        calls["n"] += 1
        if calls["n"] == 1: # Fail only on the first call
            raise TimeoutError("first fail")
        return "ok"

    # --- Test Flow ---
    # 1. First call: fails, trips breaker to OPEN
    with pytest.raises(TimeoutError):
        decorated_flaky()
    status1 = cb.status()
    assert status1["state"] == "OPEN"
    assert status1["failure_count"] == 1
    assert calls["n"] == 1

    # 2. Second call: blocked immediately by OPEN breaker (within timeout)
    with pytest.raises(r.CircuitBreakerOpen):
        decorated_flaky()
    assert calls["n"] == 1 # Function body not executed
    assert cb.status()["state"] == "OPEN" # Still OPEN

    # 3. Wait for the recovery timeout + margin
    time.sleep(timeout + _sleep_margin())

    # 4. Third call: This is the probe call.
    #    allow_request() called by decorator -> transitions to HALF_OPEN.
    #    decorated_flaky() runs (calls["n"] becomes 2), succeeds.
    #    record_success() called by decorator -> transitions HALF_OPEN -> CLOSED.
    result = decorated_flaky()
    assert result == "ok"
    assert calls["n"] == 2
    status3 = cb.status()
    assert status3["state"] == "CLOSED"
    assert status3["failure_count"] == 0 # Reset by success

    # 5. Fourth call: Breaker is CLOSED, executes normally.
    result = decorated_flaky()
    assert result == "ok"
    assert calls["n"] == 3
    assert cb.status()["state"] == "CLOSED"

def test_circuit_breaker_force_open_and_close():
    """Test manual administrative control methods force_open and force_close."""
    cb = r.CircuitBreaker(name="force_test", failure_threshold=5)
    assert cb.status()["state"] == "CLOSED"

    # Force Open
    cb.force_open()
    status_open = cb.status()
    assert status_open["state"] == "OPEN"
    assert status_open["last_failure_time"] is not None # Should set time
    assert cb.allow_request() is False # Should block

    # Force Close
    cb.force_close()
    status_closed = cb.status()
    assert status_closed["state"] == "CLOSED"
    assert status_closed["failure_count"] == 0 # Should reset count
    assert status_closed["last_failure_time"] is None # Should clear time
    assert cb.allow_request() is True # Should allow


# --- Test Global Registry ---

def test_get_circuit_breaker_returns_singleton():
    """Verify get_circuit_breaker returns the same instance for the same name."""
    cb1 = r.get_circuit_breaker("shared_breaker", failure_threshold=3, recovery_timeout=10)
    cb2 = r.get_circuit_breaker("shared_breaker", failure_threshold=5, recovery_timeout=20) # Params ignored
    cb3 = r.get_circuit_breaker("another_breaker")

    assert cb1 is cb2 # Same object for same name
    assert cb1.failure_threshold == 3 # Initial parameters are used
    assert cb1 is not cb3 # Different objects for different names

def test_get_circuit_breaker_concurrent_creation():
    """Ensure multiple threads creating the same named breaker get the same instance."""
    results = []
    num_threads = 16
    barrier = threading.Barrier(num_threads) # Synchronize thread start

    def worker():
        barrier.wait() # Wait for all threads to reach this point
        breaker_instance = r.get_circuit_breaker("concurrent_test", failure_threshold=2)
        results.append(breaker_instance)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads: t.start()
    for t in threads: t.join()

    # Check that all threads received the exact same object instance
    assert len(results) == num_threads
    first_id = id(results[0])
    assert all(id(obj) == first_id for obj in results), "Not all threads got the same breaker instance"

def test_reset_all_circuit_breakers_clears_registry_and_closes():
    """Verify reset clears the global registry and forces existing breakers closed."""
    cb_a = r.get_circuit_breaker("a")
    cb_b = r.get_circuit_breaker("b")
    cb_a.force_open() # Manually open one to test reset state
    # Use public API list_circuit_breakers to check state before reset
    assert "a" in r.list_circuit_breakers()
    assert r.list_circuit_breakers()["a"]["state"] == "OPEN"

    r.reset_all_circuit_breakers()

    # After reset, registry should be empty (checked via public API)
    assert not r.list_circuit_breakers(), "Registry should be empty after reset"
    # Internal check as well
    assert not r._BREAKERS

    # Getting breaker again creates a new one, which should be CLOSED by default
    cb_a_new = r.get_circuit_breaker("a")
    assert cb_a_new is not cb_a # Should be a new instance
    assert cb_a_new.status()["state"] == "CLOSED"

def test_list_circuit_breakers_contents():
    """Test listing the status of registered breakers using the public API."""
    # Create some breakers with different params
    r.get_circuit_breaker("list_test1", failure_threshold=1)
    r.get_circuit_breaker("list_test2", recovery_timeout=7)
    statuses = r.list_circuit_breakers()

    assert isinstance(statuses, dict)
    assert "list_test1" in statuses and "list_test2" in statuses
    # Check specific configuration values were captured in status
    assert statuses["list_test1"]["failure_threshold"] == 1
    assert statuses["list_test2"]["recovery_timeout"] == 7
    # Check default state is reported correctly
    assert statuses["list_test1"]["state"] == "CLOSED"
    assert statuses["list_test2"]["state"] == "CLOSED"


# --- Test Observability Fallbacks ---

def test_observability_fallbacks_do_not_raise(monkeypatch, caplog):
    """Verify fallback metric/audit functions log safely when observability is missing."""
    # Fixture 'ensure_clean_state' already forces fallback state by patching _OBSERVABILITY_AVAILABLE = False
    caplog.set_level(logging.DEBUG) # Need DEBUG to see fallback metric log

    # Use getattr to safely get the fallback functions (handles potential naming diffs in module)
    # Prefer _metrics_inc if it exists, otherwise _metric_inc
    metric_fn = getattr(r, "_metrics_inc", None) or getattr(r, "_metric_inc", None)
    # Prefer _audit_event if it exists, otherwise _audit_log
    audit_fn = getattr(r, "_audit_event", None) or getattr(r, "_audit_log", None)

    assert callable(metric_fn), "Fallback metric function not found or not callable"
    assert callable(audit_fn), "Fallback audit function not found or not callable"

    # These calls should execute without raising exceptions
    try:
        metric_fn("fallback.metric.test", 1)
        audit_fn("fallback.audit.test", target="resource", status="ok", details={"key": "value"})
    except Exception as e:
        pytest.fail(f"Observability fallback raised unexpectedly: {e}")

    # Check logs contain fallback messages (use substring matching for robustness)
    log_text = caplog.text.lower() # Check lowercase for case-insensitivity
    assert "metric_inc (fallback)" in log_text
    assert "fallback.metric.test" in log_text
    assert "audit (fallback)" in log_text
    assert "action=fallback.audit.test" in log_text
    assert "target=resource" in log_text


# --- Test Integration of Retry Helpers and Breaker ---

def test_retry_sync_integrates_with_breaker():
    """Verify retry_sync correctly uses the breaker (records success/failure)."""
    # Use a breaker that trips on the first failure
    cb = r.get_circuit_breaker("sync_integration", failure_threshold=1, recovery_timeout=0.1)
    calls = {"n": 0}

    def flaky_once():
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("transient") # First call fails
        return "ok" # Second call succeeds

    # Initial state: CLOSED
    assert cb.status()["state"] == "CLOSED"

    # Call with retry (retries=1 means max 2 attempts).
    # Attempt 1: flaky_once raises TimeoutError. Breaker decorator calls record_failure. State -> OPEN.
    # Retry logic waits (backoff=0), then attempts again.
    # Attempt 2: Breaker is OPEN. Decorator calls allow_request.
    #            If timeout passed, allow_request sets HALF_OPEN, returns True.
    #            flaky_once runs again (calls["n"]=2), succeeds.
    #            Decorator calls record_success. State -> CLOSED.
    #            If timeout NOT passed, allow_request returns False, raises CircuitBreakerOpen.
    # Using backoff=0, assume timeout passes if processing is quick.
    result = r.retry_sync(flaky_once, retries=1, backoff=0.12, breaker=cb)

    assert result == "ok"
    assert calls["n"] == 2
    # Verify breaker state sequence ended in CLOSED after successful retry
    status = cb.status()
    assert status["state"] == "CLOSED"
    assert status["failure_count"] == 0 # Reset by success

def test_retry_sync_blocked_when_breaker_open():
    """Verify retry_sync fails fast with CircuitBreakerOpen if breaker starts OPEN."""
    cb = r.get_circuit_breaker("sync_block", failure_threshold=1)
    cb.force_open() # Start with breaker OPEN
    calls = {"n": 0}

    def should_not_run():
        calls["n"] += 1
        return "fail"

    # retry_sync calls the breaker decorator -> allow_request returns False -> raises CircuitBreakerOpen
    with pytest.raises(r.CircuitBreakerOpen):
        r.retry_sync(should_not_run, retries=3, backoff=0, breaker=cb)
    assert calls["n"] == 0 # Function should not have been executed

@pytest.mark.asyncio
async def test_retry_async_integrates_with_breaker():
    """Verify retry_async correctly interacts with the breaker, including recording state."""
    # Use a breaker that trips on the first failure
    cb = r.get_circuit_breaker("async_integration", failure_threshold=1, recovery_timeout=0.1)
    calls = {"n": 0}

    async def flaky_once_async():
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("async transient") # First call fails
        await asyncio.sleep(0)
        return "ok_async" # Second call succeeds

    # Initial state: CLOSED
    assert cb.status()["state"] == "CLOSED"

    # Call with retry (retries=1 means max 2 attempts).
    # Attempt 1: retry_async checks allow_request (True). Calls coro_fn. Fails. Calls record_failure. State -> OPEN.
    # Retry logic waits (backoff > timeout).
    # Attempt 2: retry_async checks allow_request. Breaker OPEN, timeout passed -> allow_request sets HALF_OPEN, returns True.
    #            Calls coro_fn (calls["n"]=2). Succeeds. Calls record_success. State -> CLOSED.
    result = await r.retry_async(flaky_once_async, retries=1, backoff=0.1 + _sleep_margin(), breaker=cb) # Ensure backoff > timeout

    assert result == "ok_async"
    assert calls["n"] == 2
    # Verify breaker state sequence ended in CLOSED
    status = cb.status()
    assert status["state"] == "CLOSED"
    assert status["failure_count"] == 0 # Reset by success

@pytest.mark.asyncio
async def test_retry_async_blocked_when_breaker_open():
    """Verify retry_async fails fast with CircuitBreakerOpen if breaker starts OPEN."""
    cb = r.get_circuit_breaker("async_block", failure_threshold=1)
    cb.force_open() # Start with breaker OPEN
    calls = {"n": 0}

    async def should_not_run_async():
        calls["n"] += 1
        await asyncio.sleep(0)
        return "fail"

    # retry_async calls allow_request -> returns False -> raises CircuitBreakerOpen
    with pytest.raises(r.CircuitBreakerOpen):
        await r.retry_async(should_not_run_async, retries=3, backoff=0, breaker=cb)
    assert calls["n"] == 0 # Coroutine should not have been executed


# ---------------------
# End of file
# ---------------------