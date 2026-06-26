from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DOCS = (
    REPO_ROOT / "DEPLOY.md",
    REPO_ROOT / "AWS_DEPLOY.md",
    REPO_ROOT / "docs" / "PRODUCTION_READINESS.md",
)


def _doc_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in DEPLOY_DOCS)


def test_tracked_secret_state_and_archive_paths_are_removed() -> None:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = result.stdout.splitlines()
    blocked_patterns = (
        re.compile(r"(^|/)\.env($|[./])"),
        re.compile(r"\.(tar\.gz|tgz|zip|pem|key|db|sqlite|sqlite3)$"),
    )

    blocked = [
        path
        for path in tracked
        if any(pattern.search(path) for pattern in blocked_patterns)
    ]

    assert blocked == []


def test_deployment_docs_reject_dirty_tree_and_broad_access_patterns() -> None:
    text = _doc_text()
    lowered = text.lower()

    forbidden = [
        "0.0.0.0/0",
        "allow ssh traffic from anywhere",
        "scp -r",
        "quant-key.pem",
        "current active server",
        "repo root, gitignored",
        "create .env",
        "upload .env",
        ".env upload",
    ]

    for phrase in forbidden:
        assert phrase not in lowered

    assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)


def test_deployment_docs_require_scanned_clean_release_flow() -> None:
    text = _doc_text()

    assert "tools/security/scan_artifacts.py" in text
    assert "tools/security/build_release.py --output release.tar.gz" in text
    assert "known administrator IPs" in text
    assert "Never transfer a recursive local working tree" in text
    assert "Keep runtime secrets outside the repo checkout" in text


def test_live_enablement_docs_require_full_safety_checklist() -> None:
    text = _doc_text()

    required_phrases = [
        "Every P0 ledger row",
        "production_readiness.py --profile live",
        "Production Readiness",
        "BOT_V2_ENFORCE_GO_NO_GO=true",
        "BOT_V2_LIVE_GO_NO_GO=true",
        "BOT_V2_FORCE_ROLLBACK",
        "valid manifests and checksums",
        "Live mark freshness checks",
        "Redis is private",
        "command messages are authenticated",
        "restricted to approved operators",
        "rollback drills have been rehearsed",
        "rotated",
    ]

    for phrase in required_phrases:
        assert phrase in text
