"""
Pytest suite for backend.services.aws_utils (Diamond-grade R-01).

Covers:
 - Option parsing and redaction (_parse_s3_options_from_json, _sanitize_for_logging)
 - Transient error detection (_is_transient_error)
 - Retry wrapper (_retry_op)
 - get_s3_fs caching and concurrency behavior (mocked fsspec)
 - list_s3_prefix / download_prefix_to_local / upload_dir_to_s3
 - put_bytes_to_s3 (boto3 and fsspec fallback)
All fsspec and boto3 calls are mocked — no network required.
"""

import builtins
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.services import aws_utils as a


# ----------------------------
# Fixtures
# ----------------------------

@pytest.fixture(autouse=True)
def clear_module_state():
    """Ensure each test starts with a clean cache and semaphore."""
    a._FSSPEC_FS = None
    a._S3_SEMAPHORE = None
    yield
    a._FSSPEC_FS = None
    a._S3_SEMAPHORE = None


# ----------------------------
# Tests for internal helpers
# ----------------------------

def test_sanitize_for_logging_redacts_sensitive_keys():
    d = {"key": "AAA", "secret": "BBB", "token": "CCC", "endpoint_url": "https://x"}
    result = a._sanitize_for_logging(d)
    assert result["key"] == "***REDACTED***"
    assert result["secret"] == "***REDACTED***"
    assert result["endpoint_url"].startswith("https://")


def test_parse_s3_options_json_filters_and_casts(monkeypatch):
    opts = {"key": "X", "secret": "Y", "anon": "true", "ignored": 123}
    js = json.dumps(opts)
    result = a._parse_s3_options_from_json(js)
    assert "key" in result and "ignored" not in result
    assert isinstance(result["anon"], bool) and result["anon"] is True

    # Invalid JSON returns {}
    result2 = a._parse_s3_options_from_json("{bad}")
    assert result2 == {}


def test_is_transient_error_catches_common_patterns():
    assert a._is_transient_error(TimeoutError("connection timeout"))
    class Dummy(Exception): pass
    assert not a._is_transient_error(Dummy("permanent fail"))


def test_retry_op_retries_transient_and_fails_fast(monkeypatch):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("connection timeout")
        return "ok"

    result = a._retry_op(flaky, retries=3, backoff=0.01)
    assert result == "ok"

    # Non-transient should fail immediately
    calls["n"] = 0
    def hard_fail(): raise ValueError("invalid")
    with pytest.raises(ValueError):
        a._retry_op(hard_fail, retries=3, backoff=0.01)


# ----------------------------
# get_s3_fs
# ----------------------------

def test_get_s3_fs_creates_and_caches(monkeypatch):
    mock_fs = MagicMock()
    mock_filesystem = MagicMock(return_value=mock_fs)
    monkeypatch.setattr(a, "fsspec", MagicMock(filesystem=mock_filesystem))

    fs, sem = a.get_s3_fs('{"anon": true}', concurrency=3)
    assert fs is mock_fs
    assert isinstance(sem, threading.Semaphore)
    # Second call returns cached instance (no re-init)
    fs2, sem2 = a.get_s3_fs()
    assert fs2 is fs
    assert sem2 is sem


def test_get_s3_fs_raises_if_fsspec_missing(monkeypatch):
    monkeypatch.setattr(a, "fsspec", None)
    with pytest.raises(RuntimeError):
        a.get_s3_fs()


# ----------------------------
# list_s3_prefix
# ----------------------------

def test_list_s3_prefix_returns_sorted(monkeypatch):
    fs_mock = MagicMock()
    fs_mock.glob.return_value = ["bucket/x", "bucket/a"]
    monkeypatch.setattr(a, "get_s3_fs", lambda *_a, **_kw: (fs_mock, threading.Semaphore(1)))
    result = a.list_s3_prefix("s3://bucket/")
    assert result == ["s3://bucket/a", "s3://bucket/x"]


# ----------------------------
# download_prefix_to_local
# ----------------------------

def test_download_prefix_to_local_success(monkeypatch, tmp_path):
    fs_mock = MagicMock()
    # glob returns two objects (simulate files)
    fs_mock.glob.return_value = ["bucket/file1.txt", "bucket/file2.txt"]

    # fake open for fs (read data)
    data = b"hello"
    class FakeReader:
        def read(self, n): 
            nonlocal data
            if data: 
                d = data
                data = b""
                return d
            return b""
        def __enter__(self): return self
        def __exit__(self, *a): return False

    fs_mock.open.return_value = FakeReader()

    monkeypatch.setattr(a, "get_s3_fs", lambda *_a, **_kw: (fs_mock, threading.Semaphore(1)))

    out = a.download_prefix_to_local("s3://bucket/prefix", tmp_path)
    assert out is True
    assert any(p.name.endswith(".txt") for p in tmp_path.iterdir())


def test_download_prefix_to_local_handles_runtimeerror(monkeypatch):
    monkeypatch.setattr(a, "get_s3_fs", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("no fs")))
    result = a.download_prefix_to_local("s3://bucket", Path("/tmp/xx"))
    assert result is False


# ----------------------------
# upload_dir_to_s3
# ----------------------------

def test_upload_dir_to_s3_uploads_files(monkeypatch, tmp_path):
    fs_mock = MagicMock()
    fs_mock.open.return_value.__enter__.return_value = MagicMock()
    monkeypatch.setattr(a, "get_s3_fs", lambda *_a, **_kw: (fs_mock, threading.Semaphore(1)))

    # Create dummy file
    file_path = tmp_path / "x.txt"
    file_path.write_text("data")

    ok = a.upload_dir_to_s3(tmp_path, "s3://bucket/prefix")
    assert ok is True
    fs_mock.open.assert_called()


def test_upload_dir_to_s3_handles_no_dir(monkeypatch):
    fs_mock = MagicMock()
    monkeypatch.setattr(a, "get_s3_fs", lambda *_a, **_kw: (fs_mock, threading.Semaphore(1)))
    ok = a.upload_dir_to_s3(Path("/nonexistent"), "s3://bucket/x")
    assert ok is False


# ----------------------------
# put_bytes_to_s3
# ----------------------------

def test_put_bytes_to_s3_prefers_boto3(monkeypatch):
    mock_client = MagicMock()
    mock_boto3 = MagicMock(client=MagicMock(return_value=mock_client))
    monkeypatch.setattr(a, "boto3", mock_boto3)
    data = b"abc"
    ok = a.put_bytes_to_s3("bucket", "key", data)
    assert ok is True
    mock_client.put_object.assert_called_once()


def test_put_bytes_to_s3_fallback_to_fsspec(monkeypatch):
    # Simulate boto3 unavailable
    monkeypatch.setattr(a, "boto3", None)

    # Ensure module sees fsspec as present
    monkeypatch.setattr(a, "fsspec", MagicMock())

    # Create a fake file-like writer that records writes
    mock_writer = MagicMock()
    mock_writer.write = MagicMock()

    # fs_mock.open() should be a context manager returning mock_writer
    fs_mock = MagicMock()
    fs_mock.open.return_value.__enter__.return_value = mock_writer

    # Ensure get_s3_fs returns our fs_mock + a semaphore (no blocking)
    monkeypatch.setattr(a, "get_s3_fs", lambda *_a, **_kw: (fs_mock, threading.Semaphore(1)))

    ok = a.put_bytes_to_s3("bucket", "key", b"data")
    assert ok is True

    # extra assertion to ensure data was attempted to be written
    mock_writer.write.assert_called_once_with(b"data")


def test_put_bytes_to_s3_handles_failures(monkeypatch):
    # Both boto3 and fsspec unavailable
    monkeypatch.setattr(a, "boto3", None)
    monkeypatch.setattr(a, "fsspec", None)
    ok = a.put_bytes_to_s3("bucket", "key", b"x")
    assert ok is False