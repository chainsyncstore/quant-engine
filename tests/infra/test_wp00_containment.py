from __future__ import annotations

import io
import logging
from pathlib import Path
from urllib.parse import quote

import yaml

from quant.telebot.main import _configure_secure_logging


ROOT = Path(__file__).resolve().parents[2]


def test_telegram_transport_logs_redact_canary_token_and_authenticated_urls() -> None:
    token = "123456789:CANARY_token_value_abcdefghijklmnopqrstuvwxyz"
    encoded_token = quote(token, safe="")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    original_handlers = root.handlers[:]
    original_root_level = root.level
    original_httpx_level = httpx_logger.level
    original_httpcore_level = httpcore_logger.level
    try:
        root.handlers = [handler]
        root.setLevel(logging.INFO)
        httpx_logger.setLevel(logging.NOTSET)
        httpcore_logger.setLevel(logging.NOTSET)
        _configure_secure_logging(token)

        logging.getLogger("tests.wp00.transport").warning(
            "raw https://api.telegram.org/bot%s/getMe encoded "
            "https://api.telegram.org/bot%s/getUpdates auth "
            "redis://user:password@redis:6379",
            token,
            encoded_token,
        )
        try:
            raise RuntimeError(
                f"request failed for https://api.telegram.org/bot{token}/sendMessage"
            )
        except RuntimeError:
            logging.getLogger("tests.wp00.exception").exception("Telegram request failed")
        rendered = stream.getvalue()
    finally:
        root.handlers = original_handlers
        root.setLevel(original_root_level)
        httpx_logger.setLevel(original_httpx_level)
        httpcore_logger.setLevel(original_httpcore_level)

    assert token not in rendered
    assert encoded_token not in rendered
    assert "user:password" not in rendered
    assert "[REDACTED]" in rendered


def test_telegram_http_transport_info_logging_is_suppressed() -> None:
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() >= logging.WARNING


def test_compose_keeps_redis_private_and_acl_authenticated() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    redis = compose["services"]["redis"]
    command = "\n".join(redis["command"])
    healthcheck = " ".join(redis["healthcheck"]["test"])

    assert "ports" not in redis
    assert redis["expose"] == ["6379"]
    assert redis["secrets"] == [
        {
            "source": "redis_password",
            "target": "redis_password",
            "uid": "999",
            "gid": "999",
            "mode": 288,
        }
    ]
    assert "user default off" in command
    assert 'username="$${REDIS_ACL_USERNAME}"' in command
    assert "Invalid Redis ACL username format" in command
    assert "case \"$$username\" in ''|*[!A-Za-z0-9_-]*)" in command
    assert "--dir /data" in command
    assert "--aclfile" in command
    assert "+@all" not in command
    assert "REDISCLI_AUTH" in healthcheck
    assert "/run/secrets/redis_password" in healthcheck
    assert compose["secrets"]["redis_password"]["file"].startswith(
        "${REDIS_PASSWORD_FILE:"
    )
    for service_name in ("telegram_bot", "retrain_scheduler", "model_evaluator"):
        assert compose["services"][service_name]["depends_on"]["redis"] == {
            "condition": "service_healthy"
        }


def test_incident_helpers_are_excluded_from_build_context() -> None:
    ignored = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())
    assert {
        "check_breach.py",
        "check_decisions.py",
        "clear_all_breach.py",
        "clear_breach.py",
        "clear_breach.sql",
        "patch_eligible.py",
        "patch_status.py",
    } <= ignored
