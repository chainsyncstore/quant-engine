from __future__ import annotations

import io
import fnmatch
import json
import tarfile
from pathlib import Path

import pytest

from tools.collect_wp01_source_provenance import normalize_tar, safe_member_name
from tools.verify_wp01 import archive_content_errors, generate, read_archive, verify


def _tar_bytes(members: list[tuple[str, bytes]]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return stream.getvalue()


@pytest.mark.parametrize("name", ["../secret", "/absolute", "a/../../secret"])
def test_safe_member_name_rejects_traversal(name: str) -> None:
    with pytest.raises(ValueError):
        safe_member_name(name)


def test_normalized_archive_is_stable_and_sorted(tmp_path: Path) -> None:
    raw = _tar_bytes([("z.py", b"z\r\n"), ("a.py", b"a\n")])
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    first_manifest = normalize_tar(raw, first)
    second_manifest = normalize_tar(raw, second)

    assert first_manifest == second_manifest
    assert first.read_bytes() == second.read_bytes()
    assert read_archive(first) == read_archive(second)
    assert list(read_archive(first)) == ["a.py", "z.py"]


def test_normalize_tar_rejects_duplicate_members(tmp_path: Path) -> None:
    raw = _tar_bytes([("same.py", b"one"), ("same.py", b"two")])
    with pytest.raises(ValueError, match="duplicate"):
        normalize_tar(raw, tmp_path / "out.tar.gz")


def test_retained_wp01_evidence_verifies(tmp_path: Path) -> None:
    repo_root = Path.cwd()
    evidence = repo_root / "ubuntu_audit_20260622" / "source_provenance"
    if not evidence.exists():
        pytest.skip("WP-01 production evidence has not been collected")
    output = tmp_path / "verified"
    errors = verify(repo_root, evidence, output)

    assert errors == []
    identities = json.loads((output / "identities.json").read_text())
    assert identities["bindings"]["all_app_containers_use_inspected_image"] is True
    assert all(
        all(container_matches.values())
        for container_matches in identities["bindings"]["critical_host_container_matches"].values()
    )


def test_full_regeneration_is_deterministic(tmp_path: Path) -> None:
    repo_root = Path.cwd()
    evidence = repo_root / "ubuntu_audit_20260622" / "source_provenance"
    if not evidence.exists():
        pytest.skip("WP-01 production evidence has not been collected")
    first = tmp_path / "first"
    second = tmp_path / "second"

    generate(repo_root, evidence, first)
    generate(repo_root, evidence, second)

    first_files = {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files


@pytest.mark.parametrize(
    "content",
    [
        b"-----BEGIN PRIVATE KEY-----\nnot-a-real-key\n",
        b"123456:AAAAAAAAAAAAAAAAAAAA_fake",
        b"https://canary-user:not-a-real-password@example.invalid/path",
    ],
)
def test_archive_content_credentials_are_rejected_without_values(content: bytes) -> None:
    errors = archive_content_errors("canary.tar.gz", {"src/example.py": content})

    assert errors == ["prohibited credential content in canary.tar.gz: src/example.py"]
    assert content.decode("ascii").strip() not in errors[0]


def test_archive_content_accepts_benign_source() -> None:
    errors = archive_content_errors(
        "benign.tar.gz",
        {
            "src/example.py": b"URL = 'https://example.invalid/path'\n",
            "README.md": b"Credentials must come from the environment.\n",
        },
    )

    assert errors == []


def test_all_non_keep_paths_have_explicit_dispositions() -> None:
    repo_root = Path.cwd()
    reconciliation = json.loads(
        (repo_root / "docs" / "wp01" / "source_reconciliation.json").read_text()
    )
    dispositions = json.loads(
        (repo_root / "docs" / "wp01" / "dispositions.json").read_text()
    )
    patterns = [record["path"] for record in dispositions["records"]]
    uncovered = [
        item["path"]
        for item in reconciliation
        if item["classification"] != "keep_verified_runtime_behavior"
        and not any(fnmatch.fnmatch(item["path"], pattern) for pattern in patterns)
    ]
    assert uncovered == []
