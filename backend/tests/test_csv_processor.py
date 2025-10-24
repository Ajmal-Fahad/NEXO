#!/usr/bin/env python3
"""
backend/tests/test_csv_processor.py

Comprehensive test suite for csv_processor.py

Tests cover:
- Local CSV processing with column mappings
- S3 CSV processing with moto mocking
- Fallback behavior from S3 to local
- Async wrapper functionality
- Metrics injection for testability
- Error handling and edge cases
"""

import os
import sys
import tempfile
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

# Ensure backend package root is on sys.path so imports work when pytest is run from repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the module under test (package-qualified)
from backend.services.csv_processor import (
    process_csv,
    async_process_csv,
    find_latest_csv,
    set_metrics_collector,
    _increment_fallback_metric,
    _is_s3_uri,
    csv_settings
)

# Test data constants
TEST_CSV_CONTENT = """Symbol,Description,Price,Market capitalization,Price Change % 1 day,Price Change % 1 week,Volume Weighted Average Price 1 day,Volume Change % 1 day,Price * Volume (Turnover) 1 day,Average True Range % (14) 1 day
RELIANCE,Reliance Industries Ltd,2500.50,1500000000000,2.5,-1.2,2480.75,15.3,375000000000,3.2
TCS,Tata Consultancy Services Ltd,3200.25,1200000000000,-0.8,3.1,3220.50,-5.2,280000000000,2.8
INFY,Infosys Ltd,1450.75,600000000000,1.2,0.5,1445.25,8.7,165000000000,1.9"""

EXPECTED_PROCESSED_DATA = {
    "rank": [1, 2, 3],
    "symbol": ["RELIANCE", "TCS", "INFY"],
    "company_name": ["Reliance Industries Ltd", "Tata Consultancy Services Ltd", "Infosys Ltd"],
    "mcap_rs_cr": [150000.0, 120000.0, 60000.0],  # Converted to crores
    "price": [2500.50, 3200.25, 1450.75],
    "change_1d_pct": [2.5, -0.8, 1.2],
    "change_1w_pct": [-1.2, 3.1, 0.5],
    "vwap": [2480.75, 3220.50, 1445.25],
    "volume_change_24h_pct": [15.3, -5.2, 8.7],
    "Volume_24H": [37500.0, 28000.0, 16500.0],  # Turnover converted to crores
    "atr_14d": [3.2, 2.8, 1.9]
}


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file with test data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(TEST_CSV_CONTENT)
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_processed_dir():
    """Create a temporary directory for processed files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_s3_env():
    """Mock S3 environment variables."""
    with patch.dict(os.environ, {
        'S3_RAW_CSV_PATH': 's3://test-bucket/raw/',
        'S3_PROCESSED_CSV_PATH': 's3://test-bucket/processed/',
        'S3_STORAGE_OPTIONS': '{"anon": true}'
    }):
        yield


@pytest.fixture
def reset_metrics():
    """Reset metrics collector and counter before each test."""
    set_metrics_collector(None)
    # Reset the global counter by directly accessing the module
    import backend.services.csv_processor as csv_mod
    csv_mod._fallback_count = 0
    yield


class TestCSVProcessorLocal:
    """Test local CSV processing functionality."""

    @pytest.mark.asyncio
    async def test_process_csv_basic(self, temp_csv_file, temp_processed_dir, reset_metrics):
        """Test basic CSV processing with local files."""
        with patch('backend.services.csv_processor.PROCESSED_DIR', temp_processed_dir):
            result = await process_csv(temp_csv_file)

            # Check result structure
            assert 'path' in result
            assert 's3' in result
            assert 'rows' in result
            assert result['s3'] is False
            assert result['rows'] == 3

            # Check output file exists
            assert Path(result['path']).exists()

            # Verify processed data
            df = pd.read_csv(result['path'])
            assert len(df) == 3
            # Check required columns exist (order-insensitive)
            for col in EXPECTED_PROCESSED_DATA.keys():
                assert col in df.columns

            # Check specific values
            assert df['symbol'].tolist() == EXPECTED_PROCESSED_DATA['symbol']
            assert pytest.approx(df['price'].tolist(), rel=1e-6) == EXPECTED_PROCESSED_DATA['price']
            assert df['volume_change_24h_pct'].tolist() == EXPECTED_PROCESSED_DATA['volume_change_24h_pct']
            assert df['Volume_24H'].tolist() == EXPECTED_PROCESSED_DATA['Volume_24H']
            assert df['atr_14d'].tolist() == EXPECTED_PROCESSED_DATA['atr_14d']

    @pytest.mark.asyncio
    async def test_process_csv_column_mappings(self, temp_csv_file, temp_processed_dir, reset_metrics):
        """Test that column mappings work correctly."""
        with patch('backend.services.csv_processor.PROCESSED_DIR', temp_processed_dir):
            result = await process_csv(temp_csv_file)
            df = pd.read_csv(result['path'])

            # Verify new column mappings exist and have correct values
            assert 'volume_change_24h_pct' in df.columns
            assert 'Volume_24H' in df.columns
            assert 'atr_14d' in df.columns

            # Check values match expected
            assert df['volume_change_24h_pct'].tolist() == [15.3, -5.2, 8.7]
            assert df['Volume_24H'].tolist() == [37500.0, 28000.0, 16500.0]  # crores
            assert df['atr_14d'].tolist() == [3.2, 2.8, 1.9]

    @pytest.mark.asyncio
    async def test_process_csv_rounding(self, temp_csv_file, temp_processed_dir, reset_metrics):
        """Test that numeric columns are rounded to 2 decimals."""
        with patch('backend.services.csv_processor.PROCESSED_DIR', temp_processed_dir):
            result = await process_csv(temp_csv_file)
            df = pd.read_csv(result['path'])

            # Check that numeric columns are rounded
            numeric_cols = [col for col in df.columns if col not in ['rank', 'symbol', 'company_name']]
            for col in numeric_cols:
                series = df[col]
                # Check that values are either NaN or satisfy round(val,2) == val
                for val in series.dropna():
                    assert round(float(val), 2) == float(val), f"Column {col} has value {val} with more than 2 decimals"

    def test_process_csv_validation_error(self):
        """Test that missing required columns raise ValueError."""
        # Create CSV with missing required column
        invalid_csv = "Symbol,Description\nTEST,Test Company"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(invalid_csv)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="CSV missing required columns"):
                # process_csv is async now
                asyncio.get_event_loop().run_until_complete(process_csv(temp_path))
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestCSVProcessorS3:
    """Test S3 functionality with moto mocking."""

    @pytest.fixture
    def s3_mock(self):
        """Set up moto S3 mock."""
        import moto
        with moto.mock_aws():
            import boto3
            s3_client = boto3.client('s3', region_name='us-east-1')
            s3_client.create_bucket(Bucket='test-bucket')

            # Upload test CSV to S3
            s3_client.put_object(
                Bucket='test-bucket',
                Key='raw/test_data.csv',
                Body=TEST_CSV_CONTENT
            )
            yield s3_client

    def test_s3_uri_detection(self):
        """Test S3 URI detection."""
        assert _is_s3_uri('s3://bucket/path/file.csv') is True
        assert _is_s3_uri('/local/path/file.csv') is False
        assert _is_s3_uri('file.csv') is False

    def test_find_latest_csv_s3(self, s3_mock, mock_s3_env):
        """Test finding latest CSV from S3."""
        with patch('backend.services.csv_processor.RAW_CSV_DIR', 's3://test-bucket/raw/'), \
             patch('backend.services.csv_processor._list_s3_csvs') as mock_list:
            mock_list.return_value = ['s3://test-bucket/raw/test_data.csv']
            # find_latest_csv is async now
            result = asyncio.get_event_loop().run_until_complete(find_latest_csv())
            assert result == 's3://test-bucket/raw/test_data.csv'

    def test_process_csv_s3_success(self, s3_mock, mock_s3_env, temp_processed_dir, reset_metrics):
        """Test successful S3 processing."""
        # Create test dataframe
        from io import StringIO
        test_df = pd.read_csv(StringIO(TEST_CSV_CONTENT), dtype=str, keep_default_na=False)

        with patch('backend.services.csv_processor.PROCESSED_DIR', 's3://test-bucket/processed/'), \
             patch('backend.services.csv_processor._read_csv', return_value=test_df) as mock_read, \
             patch('backend.services.csv_processor._write_s3_csv', new_callable=AsyncMock) as mock_write:

            result = asyncio.get_event_loop().run_until_complete(process_csv('s3://test-bucket/raw/test_data.csv'))

            # Verify functions were called
            mock_read.assert_called_once_with('s3://test-bucket/raw/test_data.csv')
            mock_write.assert_awaited_once()

            # Check result indicates S3 processing
            assert result['s3'] is True
            assert 's3://test-bucket/processed/' in result['path']


class TestCSVProcessorFallback:
    """Test fallback behavior from S3 to local."""

    def test_s3_fallback_on_write_failure(self, temp_csv_file, temp_processed_dir, reset_metrics):
        """Test fallback to local when S3 write fails."""
        collected_metrics = []

        def mock_collector(name, value):
            collected_metrics.append((name, value))

        set_metrics_collector(mock_collector)

        with patch('backend.services.csv_processor.PROCESSED_DIR', 's3://test-bucket/processed/'), \
             patch('backend.services.csv_processor.LOCAL_PROCESSED_DIR', Path(temp_processed_dir)), \
             patch('backend.services.csv_processor._write_s3_csv', new_callable=AsyncMock, side_effect=Exception("S3 write failed")):

            result = asyncio.get_event_loop().run_until_complete(process_csv(temp_csv_file))

            # Should fallback to local
            assert result['s3'] is False
            assert temp_processed_dir in result['path']

            # Should have incremented fallback metric
            assert len(collected_metrics) == 1
            assert collected_metrics[0] == ('csv_processor.fallback_to_local', 1)


class TestCSVProcessorAsync:
    """Test async wrapper functionality."""

    @pytest.mark.asyncio
    async def test_async_process_csv(self, temp_csv_file, temp_processed_dir, reset_metrics):
        """Test async wrapper calls sync function."""
        with patch('backend.services.csv_processor.PROCESSED_DIR', temp_processed_dir), \
             patch('backend.services.csv_processor.process_csv', new_callable=AsyncMock) as mock_process:

            mock_process.return_value = {'path': '/test/path.csv', 's3': False, 'rows': 3}

            result = await async_process_csv(temp_csv_file)

            # Verify async wrapper called sync function (now async)
            mock_process.assert_awaited_once_with(temp_csv_file, False)
            assert result == mock_process.return_value


class TestMetricsInjection:
    """Test metrics injection functionality."""

    def test_metrics_collector_injection(self, reset_metrics):
        """Test that metrics collector can be injected and called."""
        collected_metrics = []

        def mock_collector(name, value):
            collected_metrics.append((name, value))

        # Initially no collector
        initial_count = len(collected_metrics)
        _increment_fallback_metric()
        assert len(collected_metrics) == initial_count

        # Inject collector
        set_metrics_collector(mock_collector)

        # Clear previous metrics and test
        collected_metrics.clear()
        _increment_fallback_metric()
        assert len(collected_metrics) == 1
        assert collected_metrics[0][0] == 'csv_processor.fallback_to_local'
        assert isinstance(collected_metrics[0][1], int)  # counter value

        # Test multiple calls
        _increment_fallback_metric()
        assert len(collected_metrics) == 2
        assert collected_metrics[1][0] == 'csv_processor.fallback_to_local'
        assert isinstance(collected_metrics[1][1], int)  # counter value

    def test_metrics_collector_failure_handling(self, reset_metrics):
        """Test that metrics collector failures don't break processing."""
        def failing_collector(name, value):
            raise Exception("Collector failed")

        set_metrics_collector(failing_collector)

        # Should not raise exception despite collector failure
        _increment_fallback_metric()

        # No exception = pass


class TestCSVProcessorEdgeCases:
    """Test edge cases and error conditions."""

    def test_find_latest_csv_no_files(self):
        """Test find_latest_csv when no files exist."""
        with patch('backend.services.csv_processor.LOCAL_RAW_DIR', Path('/nonexistent')), \
             patch('backend.services.csv_processor._is_s3_uri', return_value=False):
            result = asyncio.get_event_loop().run_until_complete(find_latest_csv())
            assert result is None

    def test_process_csv_file_not_found(self):
        """Test processing non-existent file."""
        with pytest.raises(FileNotFoundError):
            asyncio.get_event_loop().run_until_complete(process_csv('/nonexistent/file.csv'))

    def test_process_csv_invalid_csv(self, temp_processed_dir, reset_metrics):
        """Test processing malformed CSV."""
        invalid_csv = "Symbol,Description\nTEST"  # Missing required Price column
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(invalid_csv)
            temp_path = f.name

        try:
            with patch('backend.services.csv_processor.PROCESSED_DIR', temp_processed_dir):
                # Should raise ValueError for missing required columns
                with pytest.raises(ValueError, match="CSV missing required columns"):
                    asyncio.get_event_loop().run_until_complete(process_csv(temp_path))
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == '__main__':
    pytest.main([__file__])
