#!/usr/bin/env python3
"""Regenerate and verify deterministic WP-01 provenance deliverables."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


TEXT_SUFFIXES = {
    ".cfg", ".css", ".html", ".in", ".ini", ".js", ".json", ".md",
    ".py", ".service", ".sh", ".toml", ".txt", ".yaml", ".yml",
}
SECRET_NAME_PARTS = (".env", ".key", ".pem", ".p12", "credential", "secret")
SECRET_CONTENT_PATTERNS = (
    re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(rb"(?<![A-Za-z0-9_-])\d{5,}:[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"),
    re.compile(rb"https?://[^\s/@:]+:[^\s/@]+@"),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_sha(path: str, data: bytes) -> str:
    if Path(path).suffix.lower() in TEXT_SUFFIXES or Path(path).name in {"Dockerfile"}:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256(data)


def read_archive(path: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or not member.isfile():
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.name in result:
                raise ValueError(f"duplicate archive member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"unreadable archive member: {member.name}")
            result[member.name] = stream.read()
    return result


def archive_content_errors(archive_name: str, archive_files: dict[str, bytes]) -> list[str]:
    """Return value-silent credential findings for archive member content."""

    return [
        f"prohibited credential content in {archive_name}: {name}"
        for name, data in sorted(archive_files.items())
        if any(pattern.search(data) for pattern in SECRET_CONTENT_PATTERNS)
    ]


def git_bytes(repo: Path, revision: str, path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def local_bytes(repo: Path, path: str) -> bytes | None:
    candidate = repo / Path(path)
    return candidate.read_bytes() if candidate.is_file() else None


def digest_record(path: str, data: bytes | None) -> dict[str, object] | None:
    if data is None:
        return None
    return {"sha256": sha256(data), "normalized_sha256": normalized_sha(path, data), "size": len(data)}


def parse_sections(metadata: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in metadata.splitlines():
        if line.startswith("WP01_") and line.endswith("_BEGIN"):
            current = line.removeprefix("WP01_").removesuffix("_BEGIN").lower()
            sections[current] = []
        elif line.startswith("WP01_") and line.endswith("_END"):
            current = None
        elif current is not None:
            sections[current].append(line)
    return sections


def parse_scalar(metadata: str, marker: str) -> str:
    lines = metadata.splitlines()
    index = lines.index(marker)
    return lines[index + 1]


def parse_hash_lines(lines: list[str], container: bool = False) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in lines:
        digest, path = line.split("  ", 1)
        result[path if container else path.lstrip("*")] = digest
    return dict(sorted(result.items()))


def parse_docker(lines: list[str]) -> dict[str, object]:
    containers = []
    for line in lines[:-1]:
        name, container_id, image_id, workdir, labels = line.split("|", 4)
        containers.append({
            "name": json.loads(name).lstrip("/"),
            "container_id": json.loads(container_id),
            "image_id": json.loads(image_id),
            "workdir": json.loads(workdir),
            "labels": json.loads(labels),
        })
    image_id, repo_digests, created, os_name, architecture, workdir, labels = lines[-1].split("|", 6)
    return {
        "containers": sorted(containers, key=lambda item: item["name"]),
        "image": {
            "architecture": json.loads(architecture),
            "created": json.loads(created),
            "id": json.loads(image_id),
            "labels": json.loads(labels),
            "os": json.loads(os_name),
            "repo_digests": json.loads(repo_digests),
            "workdir": json.loads(workdir),
        },
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def generate(repo: Path, evidence: Path, output: Path) -> None:
    metadata_text = (evidence / "production_metadata.txt").read_text(encoding="utf-8")
    sections = parse_sections(metadata_text)
    host = read_archive(evidence / "host_source.tar.gz")
    container = read_archive(evidence / "container_source.tar.gz")
    paths = sorted(set(host) | set(container))

    reconciliation = []
    for path in paths:
        head_data = git_bytes(repo, "HEAD", path)
        worktree_data = local_bytes(repo, path)
        records = {
            "local_head": digest_record(path, head_data),
            "local_worktree": digest_record(path, worktree_data),
            "ubuntu_host": digest_record(path, host.get(path)),
            "running_container": digest_record(path, container.get(path)),
        }
        normalized = {
            key: value["normalized_sha256"] if value else None for key, value in records.items()
        }
        if normalized["ubuntu_host"] is not None and normalized["running_container"] is None:
            classification = "host_only_deployment_source"
        elif normalized["ubuntu_host"] is None and normalized["running_container"] is not None:
            classification = "unresolved_container_only_source"
        elif normalized["ubuntu_host"] != normalized["running_container"]:
            classification = "unresolved_host_container_mismatch"
        elif normalized["local_worktree"] == normalized["ubuntu_host"]:
            classification = "keep_verified_runtime_behavior"
        elif normalized["local_head"] == normalized["ubuntu_host"]:
            classification = "replace_with_reviewed_local_change"
        else:
            classification = "unresolved_production_only_behavior"
        reconciliation.append({"classification": classification, "path": path, "surfaces": records})

    host_hashes = parse_hash_lines(sections["critical_hashes"])
    container_hashes = parse_hash_lines(sections["container_hashes"], container=True)
    docker = parse_docker(sections["docker"])
    image_ids = {item["image_id"] for item in docker["containers"]}
    bindings = {
        "all_app_containers_use_inspected_image": image_ids == {docker["image"]["id"]},
        "critical_host_container_matches": {},
    }
    runtime_paths = sorted({key.split(":", 1)[1] for key in container_hashes})
    for host_path in runtime_paths:
        digest = host_hashes[host_path]
        bindings["critical_host_container_matches"][host_path] = {
            item["name"]: container_hashes.get(f'{item["name"]}:{host_path}') == digest
            for item in docker["containers"]
        }

    identities = {
        "docker": docker,
        "git": {
            "branch": parse_scalar(metadata_text, "WP01_BRANCH"),
            "local_head": subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip(),
            "ubuntu_head": parse_scalar(metadata_text, "WP01_HEAD"),
        },
        "bindings": bindings,
    }
    write_json(output / "identities.json", identities)
    write_json(output / "source_reconciliation.json", reconciliation)
    write_json(output / "critical_hashes.json", {"containers": container_hashes, "host": host_hashes})
    write_json(
        output / "evidence_inventory.json",
        {
            path.relative_to(repo).as_posix(): {"sha256": sha256(path.read_bytes()), "size": path.stat().st_size}
            for path in sorted(evidence.iterdir()) if path.is_file()
        },
    )


def verify(repo: Path, evidence: Path, output: Path) -> list[str]:
    errors: list[str] = []
    generate(repo, evidence, output)
    identities = json.loads((output / "identities.json").read_text(encoding="utf-8"))
    if not identities["bindings"]["all_app_containers_use_inspected_image"]:
        errors.append("application containers do not all use the inspected image")
    for path, matches in identities["bindings"]["critical_host_container_matches"].items():
        for container, matched in matches.items():
            if not matched:
                errors.append(f"critical hash mismatch: {path} on {container}")
    for archive_name, manifest_name in (
        ("host_source.tar.gz", "host_archive_manifest.json"),
        ("container_source.tar.gz", "container_archive_manifest.json"),
    ):
        archive_files = read_archive(evidence / archive_name)
        expected_manifest = json.loads((evidence / manifest_name).read_text(encoding="utf-8"))
        actual_manifest = {
            name: {"sha256": sha256(data), "size": len(data)}
            for name, data in sorted(archive_files.items())
        }
        if actual_manifest != expected_manifest:
            errors.append(f"archive manifest mismatch: {archive_name}")
        errors.extend(archive_content_errors(archive_name, archive_files))
        for name in archive_files:
            lowered = name.lower()
            if any(part in lowered for part in SECRET_NAME_PARTS):
                errors.append(f"prohibited secret-like path in {archive_name}: {name}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--evidence", type=Path, default=Path("ubuntu_audit_20260622/source_provenance"))
    parser.add_argument("--output", type=Path, default=Path("docs/wp01"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    evidence = (repo / args.evidence).resolve() if not args.evidence.is_absolute() else args.evidence
    output = (repo / args.output).resolve() if not args.output.is_absolute() else args.output
    errors = verify(repo, evidence, output)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("WP-01 provenance verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
