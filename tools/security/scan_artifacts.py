"""Release artifact scanner for secret/state leakage paths.

The scanner checks filesystem trees and archive member names without extracting
archives. It reports path/reason pairs only; it never prints file contents.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DENY_GLOBS = (
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
    "models/*",
    "*.pkl",
    "*.pickle",
    "*.joblib",
    "*audit_report*.md",
    "*audit_report*.txt",
)

CONTENT_PATTERNS = (
    re.compile(r"\b[A-Z0-9_]*API_KEY\s*=\s*['\"][^'\"]{16,}['\"]", re.IGNORECASE),
    re.compile(r"\b[A-Z0-9_]*PASSWORD\s*=\s*['\"][^'\"]{8,}['\"]", re.IGNORECASE),
    re.compile(r"\bX-SECURITY-TOKEN\b\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE),
    re.compile(r"\bCST\b\s*[:=]\s*['\"][^'\"]+['\"]"),
    re.compile(r"\bhgetall\s*\(", re.IGNORECASE),
    re.compile(r"\bexecute\s*\(\s*['\"]SELECT\s+\*", re.IGNORECASE),
    re.compile(r"\bprint\s*\([^)]*(?:response|resp)\.text", re.IGNORECASE),
)

TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".json", ".yml", ".yaml", ".env"}
MAX_TEXT_SCAN_BYTES = 1_048_576
CONTENT_SCAN_EXEMPT_GLOBS = (
    "*.md",
    "tests/*",
    "*/tests/*",
    "test_*.py",
    "*_test.py",
)


@dataclass(frozen=True)
class Finding:
    path: str
    reason: str


def _normalize(name: str) -> str:
    normalized = name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _matches_glob(name: str, pattern: str) -> bool:
    normalized = _normalize(name)
    basename = Path(normalized).name
    return fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(basename, pattern)


def _is_content_scan_exempt(name: str) -> bool:
    comparable = name.rsplit("!", 1)[-1]
    if Path(_normalize(comparable)).name == ".env":
        return False
    return any(_matches_glob(comparable, pattern) for pattern in CONTENT_SCAN_EXEMPT_GLOBS)


def deny_reasons(name: str, *, top_level_archive: bool = False) -> list[str]:
    reasons: list[str] = []
    for pattern in DENY_GLOBS:
        if top_level_archive and pattern in {"*.tar", "*.tar.gz", "*.tgz", "*.zip"}:
            continue
        if _matches_glob(name, pattern):
            reasons.append(f"denied_path:{pattern}")
    return reasons


def deny_reason(name: str, *, top_level_archive: bool = False) -> str | None:
    reasons = deny_reasons(name, top_level_archive=top_level_archive)
    return reasons[0] if reasons else None


def _is_archive(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith((".tar", ".tar.gz", ".tgz", ".zip"))


def _is_text_name(name: str) -> bool:
    path = Path(_normalize(name))
    if path.name == ".env" or path.name.startswith(".env."):
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def _is_text_candidate(path: Path) -> bool:
    return _is_text_name(path.name)


def _scan_text(text: str, label: str) -> list[Finding]:
    if _is_content_scan_exempt(label):
        return []
    findings: list[Finding] = []
    for pattern in CONTENT_PATTERNS:
        if pattern.search(text):
            findings.append(Finding(label, f"denied_content:{pattern.pattern}"))
    return findings


def _scan_text_bytes(data: bytes, label: str) -> list[Finding]:
    return _scan_text(data.decode("utf-8", errors="ignore"), label)


def _scan_text_file(path: Path, label: str) -> list[Finding]:
    if not _is_text_candidate(path):
        return []
    try:
        data = path.read_bytes()
    except OSError as exc:
        return [Finding(label, f"read_failed:{exc.__class__.__name__}")]
    if len(data) > MAX_TEXT_SCAN_BYTES:
        return [Finding(label, "text_too_large_for_content_scan")]
    return _scan_text_bytes(data, label)


def scan_archive(path: Path, *, top_level: bool = True) -> list[Finding]:
    findings: list[Finding] = []
    try:
        if tarfile.is_tarfile(path):
            with tarfile.open(path) as archive:
                for member in archive.getmembers():
                    normalized = _normalize(member.name)
                    label = f"{path}!{normalized}"
                    for reason in deny_reasons(member.name, top_level_archive=False):
                        findings.append(Finding(label, reason))
                    if not member.isfile() or not _is_text_name(member.name):
                        continue
                    if member.size > MAX_TEXT_SCAN_BYTES:
                        findings.append(Finding(label, "text_too_large_for_content_scan"))
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        findings.append(Finding(label, "archive_member_read_failed"))
                        continue
                    findings.extend(_scan_text_bytes(extracted.read(MAX_TEXT_SCAN_BYTES + 1), label))
        elif zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    normalized = _normalize(member.filename)
                    label = f"{path}!{normalized}"
                    for reason in deny_reasons(member.filename, top_level_archive=False):
                        findings.append(Finding(label, reason))
                    if member.is_dir() or not _is_text_name(member.filename):
                        continue
                    if member.file_size > MAX_TEXT_SCAN_BYTES:
                        findings.append(Finding(label, "text_too_large_for_content_scan"))
                        continue
                    with archive.open(member) as extracted:
                        findings.extend(
                            _scan_text_bytes(extracted.read(MAX_TEXT_SCAN_BYTES + 1), label)
                        )
        else:
            return [Finding(str(path), "archive_unreadable")]
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        return [Finding(str(path), f"archive_read_failed:{exc.__class__.__name__}")]

    return findings


def scan_path(path: Path, *, top_level: bool = True) -> list[Finding]:
    findings: list[Finding] = []
    if not path.exists():
        return [Finding(str(path), "missing_path")]

    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_dir():
                continue
            rel = _normalize(str(child.relative_to(path)))
            for reason in deny_reasons(rel):
                findings.append(Finding(str(child), reason))
            if _is_archive(child):
                findings.extend(scan_archive(child, top_level=False))
            findings.extend(_scan_text_file(child, str(child)))
        return findings

    for reason in deny_reasons(path.name, top_level_archive=top_level and _is_archive(path)):
        findings.append(Finding(str(path), reason))
    if _is_archive(path):
        findings.extend(scan_archive(path, top_level=top_level))
    else:
        findings.extend(_scan_text_file(path, str(path)))
    return findings


def scan_many(paths: Iterable[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        findings.extend(scan_path(path))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan release artifacts for secret/state leakage paths.")
    parser.add_argument("paths", nargs="+", help="Filesystem paths or archives to scan")
    args = parser.parse_args(argv)

    findings = scan_many(Path(raw) for raw in args.paths)
    if findings:
        print("Artifact scan failed:")
        for finding in findings:
            print(f"- {finding.path}: {finding.reason}")
        return 1
    print("Artifact scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
