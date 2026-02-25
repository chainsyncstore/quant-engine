"""Tests for Fix 1: Exponential backoff and HTTP 429 handling in BinanceClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from quant.config import BinanceAPIConfig
from quant.data.binance_client import BinanceClient


def _make_client() -> BinanceClient:
    """Create a BinanceClient with dummy config for testing."""
    cfg = BinanceAPIConfig(
        api_key="test_key",
        api_secret="test_secret",
        base_url="https://fapi.binance.com",
    )
    return BinanceClient(config=cfg)


def _mock_response(status_code: int = 200, json_data=None, headers=None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = headers or {}
    resp.text = ""
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            response=resp
        )
    return resp


class TestHTTP429ExponentialBackoff:
    """Test that 429 responses trigger retries with exponential backoff."""

    @patch("quant.data.binance_client.time.sleep")
    @patch("quant.data.binance_client.time.time", return_value=1000.0)
    @patch("quant.data.binance_client.requests")
    def test_429_triggers_exponential_backoff(self, mock_requests, mock_time, mock_sleep):
        """On 429, client should honour Retry-After and retry, eventually succeeding."""
        client = _make_client()

        rate_limited = _mock_response(
            429,
            headers={"Retry-After": "2", "X-MBX-USED-WEIGHT-1M": "2200"},
        )
        success = _mock_response(200, json_data={"result": "ok"})

        mock_requests.get = MagicMock(side_effect=[rate_limited, success])

        result = client._get("https://fapi.binance.com/fapi/v1/test", {})

        assert result == {"result": "ok"}
        assert mock_requests.get.call_count == 2
        # Should have slept for the Retry-After duration (2s)
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list if c.args]
        assert any(s >= 2 for s in sleep_calls), f"Expected a sleep >= 2s, got {sleep_calls}"


class TestHTTP418IPBan:
    """Test that 418 (IP ban) raises immediately without retry."""

    @patch("quant.data.binance_client.time.sleep")
    @patch("quant.data.binance_client.time.time", return_value=1000.0)
    @patch("quant.data.binance_client.requests")
    def test_418_ip_ban_raises_immediately(self, mock_requests, mock_time, mock_sleep):
        """On 418, client should NOT retry — raise immediately."""
        client = _make_client()

        banned = _mock_response(418)
        mock_requests.get = MagicMock(return_value=banned)

        with pytest.raises(requests.HTTPError):
            client._get("https://fapi.binance.com/fapi/v1/test", {})

        # Only 1 attempt — no retries
        assert mock_requests.get.call_count == 1


class TestHTTP5xxRetry:
    """Test that 5xx responses retry up to 3 times then succeed."""

    @patch("quant.data.binance_client.time.sleep")
    @patch("quant.data.binance_client.time.time", return_value=1000.0)
    @patch("quant.data.binance_client.requests")
    def test_5xx_retries_then_succeeds(self, mock_requests, mock_time, mock_sleep):
        """Two 503s followed by a success should result in 3 total calls."""
        client = _make_client()

        error_503 = _mock_response(503)
        success = _mock_response(200, json_data={"data": "ok"})

        mock_requests.get = MagicMock(side_effect=[error_503, error_503, success])

        result = client._get("https://fapi.binance.com/fapi/v1/test", {})

        assert result == {"data": "ok"}
        assert mock_requests.get.call_count == 3


class TestWeightThrottling:
    """Test proactive throttling when weight exceeds threshold."""

    def test_weight_tracking_from_response_header(self):
        """_update_rate_limit_weight should parse X-MBX-USED-WEIGHT-1M."""
        client = _make_client()
        resp = _mock_response(200, headers={"X-MBX-USED-WEIGHT-1M": "1900"})
        client._update_rate_limit_weight(resp)
        assert client._used_weight_1m == 1900
