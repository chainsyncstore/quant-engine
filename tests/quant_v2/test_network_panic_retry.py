"""Tests for Pre-Flight Fix 1: Network panic retry in BinanceClient._request_with_backoff."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import requests
import requests.exceptions

from quant.data.binance_client import BinanceClient
from quant.config import BinanceAPIConfig


@pytest.fixture
def client():
    """Create a BinanceClient with test config."""
    cfg = BinanceAPIConfig(
        api_key="test_key",
        api_secret="test_secret",
        base_url="https://fapi.binance.com",
        symbol="BTCUSDT",
        interval="1h",
    )
    c = BinanceClient(config=cfg)
    # Disable throttling for fast tests
    c._MIN_REQUEST_INTERVAL = 0.0
    c._BACKOFF_BASE_DELAY = 0.01  # Very small for test speed
    return c


class TestNetworkPanicRetry:
    """Verify that socket-level exceptions are caught and retried."""

    def test_connection_error_retries_then_succeeds(self, client):
        """ConnectionError on first attempt should retry and succeed on second."""
        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.headers = {}

        with patch("requests.get", side_effect=[
            requests.exceptions.ConnectionError("DNS resolution failed"),
            mock_success,
        ]):
            resp = client._request_with_backoff("get", "https://example.com", {})
            assert resp.status_code == 200

    def test_timeout_retries_then_succeeds(self, client):
        """Timeout on first attempt should retry and succeed."""
        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.headers = {}

        with patch("requests.get", side_effect=[
            requests.exceptions.Timeout("read timed out"),
            mock_success,
        ]):
            resp = client._request_with_backoff("get", "https://example.com", {})
            assert resp.status_code == 200

    def test_ssl_error_retries_then_succeeds(self, client):
        """SSLError should be retried."""
        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.headers = {}

        with patch("requests.get", side_effect=[
            requests.exceptions.SSLError("handshake failed"),
            mock_success,
        ]):
            resp = client._request_with_backoff("get", "https://example.com", {})
            assert resp.status_code == 200

    def test_connection_error_exhausts_retries_then_raises(self, client):
        """If all retries fail with ConnectionError, should raise."""
        client._BACKOFF_MAX_RETRIES = 3

        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("DNS fail")):
            with pytest.raises(requests.exceptions.ConnectionError):
                client._request_with_backoff("get", "https://example.com", {})

    def test_mixed_network_and_http_errors(self, client):
        """ConnectionError followed by HTTP 500 followed by success."""
        mock_500 = MagicMock()
        mock_500.status_code = 500
        mock_500.headers = {}

        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.headers = {}

        with patch("requests.get", side_effect=[
            requests.exceptions.ConnectionError("reset"),
            mock_500,
            mock_200,
        ]):
            resp = client._request_with_backoff("get", "https://example.com", {})
            assert resp.status_code == 200

    def test_gaierror_via_connection_error(self, client):
        """socket.gaierror is wrapped as ConnectionError by requests."""
        import socket

        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.headers = {}

        # requests wraps socket.gaierror in ConnectionError
        gaierror = requests.exceptions.ConnectionError(
            socket.gaierror(8, "nodename nor servname provided")
        )
        with patch("requests.get", side_effect=[gaierror, mock_success]):
            resp = client._request_with_backoff("get", "https://example.com", {})
            assert resp.status_code == 200
