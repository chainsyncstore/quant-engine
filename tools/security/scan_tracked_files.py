"""Scan tracked files for committed secret/state leakage.

This gate is intentionally separate from the release artifact scanner. Artifact
scans use broad diagnostic-content rules on assembled bundles; tracked-file
scans use path rules plus high-confidence secret signatures so test fixtures and
scanner source do not create noisy false positives.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]

TRACKED_DENY_GLOBS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.zip",
    "*.log",
    "signal_log.json",
    "debug_credentials.py",
    "test_capital_auth.py",
    "redis_analyzer_local.py",
    "analyze_sqlite_local.py",
    "trade_analyzer_local.py",
    "debug_*.py",
    "*.pkl",
    "*.pickle",
    "*.joblib",
    "*audit_report*.md",
    "*audit_report*.txt",
)

TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".conf",
    ".env",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

HIGH_CONFIDENCE_CONTENT_PATTERNS = (
    ("private_key_block", re.compile("-----BEGIN " + r"[A-Z0-9 ]*PRIVATE KEY-----")),
    ("ssh_private_key_block", re.compile("-----BEGIN " + r"OPENSSH PRIVATE KEY-----")),
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("aws_secret_access_key_assignment", re.compile(r"\bAWS_SECRET_ACCESS_KEY\s*=\s*['\"]?[A-Za-z0-9/+]{35,}")),
    ("telegram_bot_token", re.compile(r"\b\d{6,}:" + r"[A-Za-z0-9_-]{30,}\b")),
    ("capital_session_header", re.compile(r"\bX-SECURITY-TOKEN\s*[:=]\s*['\"]?[A-Za-z0-9._-]{20,}", re.IGNORECASE)),
)


@dataclass(frozen=True)
class Finding:
    path: str
    reason: str


def _matches_glob(name: str, pattern: str) -> bool:
    normalized = name.replace("\\", "/")
    basename = Path(normalized).name
    return fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(basename, pattern)


def _tracked_path_deny_reasons(name: str) -> list[str]:
    return [
        f"denied_path:{pattern}"
        for pattern in TRACKED_DENY_GLOBS
        if _matches_glob(name, pattern)
    ]


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or "git ls-files failed")
    names = [name for name in result.stdout.decode("utf-8", errors="replace").split("\0") if name]
    return [REPO_ROOT / name for name in names]


def _is_text_candidate(path: Path) -> bool:
    if path.name == ".env" or path.name.startswith(".env."):
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def _scan_content(path: Path, label: str) -> list[Finding]:
    if not _is_text_candidate(path):
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return [Finding(label, f"read_failed:{exc.__class__.__name__}")]
    findings: list[Finding] = []
    for reason, pattern in HIGH_CONFIDENCE_CONTENT_PATTERNS:
        if pattern.search(text):
            findings.append(Finding(label, f"denied_content:{reason}"))
    return findings


def scan_tracked_paths(paths: Iterable[Path], *, repo_root: Path = REPO_ROOT) -> list[Finding]:
    findings: list[Finding] = []
    root = repo_root.resolve()
    for path in paths:
        resolved = path.resolve()
        try:
            label = resolved.relative_to(root).as_posix()
        except ValueError:
            label = str(path)

        if not resolved.exists():
            # Clean-tree readiness checks catch dirty deletions. This scanner is
            # focused on leakage in files that are present in the tracked set.
            continue

        for reason in _tracked_path_deny_reasons(label):
            findings.append(Finding(label, reason))
        findings.extend(_scan_content(resolved, label))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan git-tracked files for secret/state leakage.")
    parser.parse_args(argv)

    try:
        paths = _tracked_files()
    except RuntimeError as exc:
        print(f"Tracked-file secret scan failed: {exc}", file=sys.stderr)
        return 1

    findings = scan_tracked_paths(paths)
    if findings:
        print("Tracked-file secret scan failed:")
        for finding in findings:
            print(f"- {finding.path}: {finding.reason}")
        return 1

    print("Tracked-file secret scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
