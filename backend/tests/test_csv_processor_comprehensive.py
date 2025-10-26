# tests/test_csv_processor.py

import os
import sys
import json
import pytest
import pandas as pd
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

# moto mocks
from moto import mock_aws

# --- Setup sys.path to find the backend module ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import csv_processor

# --- Test Constants ---
TEST_BUCKET = "test-csv-bucket"
RAW_PREFIX = "raw-csvs"
PROCESSED_PREFIX = "processed-csvs"
SECRETS_NAME = "test/s3-credentials"

VALID_CSV_CONTENT = (
    "Symbol,Description,Price\n"
    "AAPL,Apple Inc.,150.00\n"
    "GOOG,Alphabet Inc.,2800.00\n"
)
CSV_INJECTION_CONTENT = (
    "Symbol,Description,Price\n"
    "SECURE,A normal value,100\n"
    "=cmd|'/C calc'!A1,An attack,=SUM(1+1)\n"
)
EMPTY_CSV_CONTENT = "Symbol,Description,Price\n"
INVALID_SCHEMA_CONTENT = "Ticker,Name\nMSFT,Microsoft\n"


# --- Pytest Fixtures for Mocking ---

@pytest.fixture(scope="function")
def aws_credentials_env():
    """Set minimal fake AWS creds for moto / boto3."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    yield
    # cleanup: optionally pop env vars
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SECURITY_TOKEN", "AWS_SESSION_TOKEN", "AWS_DEFAULT_REGION"):
        os.environ.pop(k, None)


@pytest.fixture(scope="function")
def mock_aws_env(aws_credentials_env):
    """Combined S3 + Secrets Manager mock context."""
    with mock_aws():
        yield


@pytest.fixture(scope="function")
def mock_settings(monkeypatch):
    """Fixture to set environment variables for each test and reset settings cache."""
    monkeypatch.setenv("S3_RAW_CSV_PATH", f"s3://{TEST_BUCKET}/{RAW_PREFIX}")
    monkeypatch.setenv("S3_PROCESSED_CSV_PATH", f"s3://{TEST_BUCKET}/{PROCESSED_PREFIX}")
    monkeypatch.setenv("AWS_S3_SECRETS_NAME", "")
    monkeypatch.setenv("CSV_COLUMN_MAPPINGS", "")
    monkeypatch.setenv("LOG_FORMAT", "text")  # Use text for easier capture
    csv_processor.reset_cached_settings()
    yield monkeypatch


@pytest.fixture(scope="function")
def s3_client(mock_aws_env):
    """Yields a boto3 S3 client within a mocked AWS environment and creates a bucket."""
    client = csv_processor.boto3.client("s3")
    client.create_bucket(Bucket=TEST_BUCKET)
    yield client


@pytest.fixture(scope="function")
def secretsmanager_client(mock_aws_env):
    """Yields a boto3 Secrets Manager client and creates a mock secret."""
    client = csv_processor.boto3.client("secretsmanager")
    secret_data = json.dumps({"key": "mock_key", "secret": "mock_secret"})
    client.create_secret(Name=SECRETS_NAME, SecretString=secret_data)
    yield client


@pytest.fixture(scope="function")
def local_fs(fs):
    """Uses pyfakefs to create fake filesystem directories used by the processor."""
    fs.create_dir(csv_processor.LOCAL_RAW_DIR)
    fs.create_dir(csv_processor.LOCAL_PROCESSED_DIR)
    yield fs


# --- Test Cases ---

class TestSecurityAndSanitization:
    """CRITICAL: Tests for all security and data sanitization features."""

    def test_sanitize_csv_value_neutralizes_injection(self):
        assert csv_processor._sanitize_csv_value("=SUM(A1:A10)") == "'=SUM(A1:A10)"
        assert csv_processor._sanitize_csv_value("+1-1") == "'+1-1"
        assert csv_processor._sanitize_csv_value("-1+1") == "'-1+1"
        # fixed quoting so the test file parses correctly
        assert csv_processor._sanitize_csv_value("@import('//evil.com')") == "'@import('//evil.com')"
        assert csv_processor._sanitize_csv_value("\t=danger") == "'=danger"
        assert csv_processor._sanitize_csv_value("safe") == "safe"
        assert csv_processor._sanitize_csv_value(None) == ""

    def test_sanitize_dataframe(self):
        df = pd.DataFrame({'A': ["=cmd", "safe"], 'B': [123, 456]})
        sanitized_df = csv_processor._sanitize_dataframe_for_csv(df)
        assert sanitized_df.loc[0, 'A'] == "'=cmd"
        assert sanitized_df.loc[1, 'A'] == "safe"

    def test_path_safety(self, local_fs):
        # Allowed path
        allowed_file = os.path.join(csv_processor.LOCAL_RAW_DIR, "good.csv")
        assert csv_processor._is_path_safe(allowed_file) is True
        # Blocked path traversal attempt
        blocked_file = os.path.join(csv_processor.LOCAL_RAW_DIR, "../../etc/passwd")
        assert csv_processor._is_path_safe(blocked_file) is False
        # S3 considered safe by _is_path_safe
        assert csv_processor._is_path_safe(f"s3://{TEST_BUCKET}/file.csv") is True

    def test_secrets_manager_integration(self, secretsmanager_client, mock_settings):
        # Arrange
        mock_settings.setenv("AWS_S3_SECRETS_NAME", SECRETS_NAME)
        csv_processor.reset_cached_settings()

        # Act
        opts = csv_processor._parse_s3_storage_options()

        # Assert
        assert opts.get("key") == "mock_key"
        assert opts.get("secret") == "mock_secret"


class TestDataIntegrity:
    """CRITICAL: Tests for data integrity features like checksums and hashing."""

    def test_s3_write_integrity_with_boto3_and_md5(self, s3_client, mock_settings):
        # Arrange
        df = pd.DataFrame({"colA": [1], "colB": ["test"]})
        output_path = f"s3://{TEST_BUCKET}/{PROCESSED_PREFIX}/integrity_test.csv"

        # Act
        etag = csv_processor._write_s3_csv_sync(output_path, df)

        # Assert
        assert etag is not None
        # Verify object exists and ETag matches what our function returned
        response = s3_client.get_object(Bucket=TEST_BUCKET, Key=f"{PROCESSED_PREFIX}/integrity_test.csv")
        assert response['ETag'].strip('"') == etag

    def test_local_write_integrity_with_sha256(self, local_fs, mock_settings):
        # Arrange
        local_path = os.path.join(csv_processor.LOCAL_RAW_DIR, "test.csv")
        local_fs.create_file(local_path, contents=VALID_CSV_CONTENT)

        # Act
        result = csv_processor.process_csv_sync(local_path)

        # Assert
        assert result["sha256_hash"] is not None
        manual_hash = csv_processor._calculate_sha256_hash(result["path"])
        assert result["sha256_hash"] == manual_hash


class TestResilienceAndFallbacks:
    """CRITICAL: Tests for circuit breaker, retries, and fallback mechanisms."""

    def test_s3_write_fallback_to_local(self, s3_client, local_fs, mock_settings, mocker):
        # Arrange: put a raw CSV in mocked S3
        s3_path = f"s3://{TEST_BUCKET}/{RAW_PREFIX}/test.csv"
        s3_client.put_object(Bucket=TEST_BUCKET, Key=f"{RAW_PREFIX}/test.csv", Body=VALID_CSV_CONTENT)

        # Mock the S3 filesystem read to return a file-like object
        from io import StringIO
        mock_fs = MagicMock()
        mock_file = StringIO(VALID_CSV_CONTENT)
        mock_fs.open.return_value.__enter__.return_value = mock_file
        mocker.patch.object(csv_processor, '_get_s3_fs', return_value=mock_fs)

        # Force S3 write to fail to trigger fallback
        mocker.patch.object(csv_processor, '_write_s3_csv_sync', side_effect=IOError("S3 is down!"))

        # Act
        result = csv_processor.process_csv_sync(s3_path)

        # Assert fallback to local
        assert result["s3"] is False
        assert result["error"] is not None
        assert "S3 is down!" in result["error"]
        assert result["sha256_hash"] is not None
        assert os.path.exists(result["path"])
        assert str(csv_processor.LOCAL_PROCESSED_DIR) in result["path"]

    def test_circuit_breaker_opens_on_non_transient_error(self, mock_settings, mocker):
        # Arrange
        s = csv_processor.get_csv_settings()
        breaker = csv_processor.CircuitBreaker(failure_threshold=1)
        breaker.state = "CLOSED"
        breaker.failure_count = 0

        failing_func = mocker.Mock(side_effect=ValueError("A permanent error"))

        # Act / Assert: call _retry_call_sync which should call breaker.record_failure and open breaker
        with pytest.raises(ValueError):
            csv_processor._retry_call_sync(
                failing_func, retries=s.S3_RETRIES, backoff=0, timeout=1, breaker=breaker
            )

        failing_func.assert_called()
        assert breaker.state == "OPEN"
        assert breaker.failure_count >= 1


class TestConfigurationAndFlexibility:
    """CRITICAL: Tests for dynamic configuration features."""

    def test_dynamic_column_mapping(self, local_fs, mock_settings):
        # Arrange
        custom_csv = "Ticker,Company Name,Last Price\nMSFT,Microsoft Corp,300.00\n"
        local_path = os.path.join(csv_processor.LOCAL_RAW_DIR, "custom.csv")
        local_fs.create_file(local_path, contents=custom_csv)

        # Set mapping env var and reset cache
        mapping = json.dumps({"Ticker": "Symbol", "Company Name": "Description", "Last Price": "Price"})
        mock_settings.setenv("CSV_COLUMN_MAPPINGS", mapping)
        csv_processor.reset_cached_settings()

        # Act
        result = csv_processor.process_csv_sync(local_path)

        # Assert
        assert result["error"] is None
        assert result["rows"] == 1
        processed_df = pd.read_csv(result["path"])
        assert processed_df.loc[0, 'symbol'] == 'MSFT'
        assert processed_df.loc[0, 'company_name'] == 'Microsoft Corp'


class TestObservability:
    """CRITICAL: Tests for logging, correlation IDs, and metrics."""

    def test_correlation_id_injection_in_logs(self, caplog, capsys, mock_settings):
        caplog.set_level(logging.INFO)
        csv_processor.configure_logging_from_settings(force=True)

        test_id = "test-corr-id-123"
        csv_processor.correlation_id_var.set(test_id)

        # Act
        csv_processor.logger.info("This is a test log.")

        # Assert - check captured stderr for the correlation id and message
        captured = capsys.readouterr()
        assert test_id in captured.err
        assert "This is a test log" in captured.err

    @pytest.mark.asyncio
    async def test_async_task_gets_correlation_id(self, local_fs, mock_settings, capsys):
        csv_processor.configure_logging_from_settings(force=True)
        csv_processor.correlation_id_var.set('standalone')

        local_path = os.path.join(csv_processor.LOCAL_RAW_DIR, "async_test.csv")
        local_fs.create_file(local_path, contents=VALID_CSV_CONTENT)

        # Act
        await csv_processor.async_process_csv(local_path)

        # Assert - ensure a generated task- id appears in logs
        captured = capsys.readouterr()
        assert "task-" in captured.err


class TestCoreFunctionality:
    """Tests for core logic and edge cases."""

    def test_data_conversion_robustness(self):
        assert csv_processor._robust_float_convert("") is None
        assert csv_processor._robust_float_convert(None) is None
        assert csv_processor._robust_float_convert("not a number") is None
        assert csv_processor._robust_float_convert("1,234.56") == 1234.56
        assert csv_processor._robust_float_convert("(123.45)") == -123.45

    def test_find_latest_csv_sync(self, local_fs, mock_settings):
        # Arrange
        path1 = os.path.join(csv_processor.LOCAL_RAW_DIR, "data_2023-01-01.csv")
        path2 = os.path.join(csv_processor.LOCAL_RAW_DIR, "data_2023-01-02.csv")
        local_fs.create_file(path1)
        os.utime(path1, (100, 100))
        local_fs.create_file(path2)
        os.utime(path2, (200, 200))

        # Act
        latest = csv_processor.find_latest_csv_sync()

        # Assert
        assert latest == path2

    def test_edge_case_empty_csv(self, local_fs, mock_settings):
        local_path = os.path.join(csv_processor.LOCAL_RAW_DIR, "empty.csv")
        local_fs.create_file(local_path, contents=EMPTY_CSV_CONTENT)
        result = csv_processor.process_csv_sync(local_path)
        assert result["rows"] == 0

    def test_validation_failure_for_missing_columns(self, local_fs, mock_settings):
        local_path = os.path.join(csv_processor.LOCAL_RAW_DIR, "invalid.csv")
        local_fs.create_file(local_path, contents=INVALID_SCHEMA_CONTENT)
        # If pandera is installed, it will raise SchemaError (singular) or SchemaErrors depending on version;
        # otherwise fallback validation in code raises ValueError.
        try:
            from pandera.errors import SchemaError as PanderaSchemaError
        except Exception:
            PanderaSchemaError = None

        if PanderaSchemaError:
            with pytest.raises(Exception):
                csv_processor.process_csv_sync(local_path)
        else:
            with pytest.raises(ValueError):
                csv_processor.process_csv_sync(local_path)