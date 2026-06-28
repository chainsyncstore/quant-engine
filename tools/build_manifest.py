#!/usr/bin/env python3
"""Generate deterministic source/build identity for WP-02 releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCKS = (
    "requirements/build.lock",
    "requirements/runtime.lock",
    "requirements/test.lock",
)
BUILD_INPUTS = (
    ".dockerignore",
    ".github/workflows/wp02-image.yml",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.override.yml",
    "docker-compose.prod.yml",
    "pyproject.toml",
    "requirements/runtime.in",
    "requirements/test.in",
    "requirements/build.in",
    *LOCKS,
    "bootstrap_registry.py",
    "tools/build_manifest.py",
    "tools/image_smoke.py",
    "tools/image_attestation.py",
    "tools/verify_external_artifacts.py",
    "scripts/build_release.ps1",
    "scripts/verify_immutable_image.py",
)
SOURCE_DIRECTORIES = ("quant", "quant_v2", "tests")


def _git(*args: str, root: Path = ROOT) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reviewed_context(root: Path = ROOT) -> list[str]:
    """Return only source and build inputs that can affect the release image."""
    paths = {path for path in BUILD_INPUTS if (root / path).is_file()}
    for directory in SOURCE_DIRECTORIES:
        base = root / directory
        if not base.exists():
            continue
        paths.update(
            path.relative_to(root).as_posix()
            for path in base.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    return sorted(paths)


def source_digest(root: Path = ROOT) -> tuple[str, list[dict[str, str]]]:
    records = []
    for relative in reviewed_context(root):
        path = root / relative
        if not path.is_file():
            continue
        records.append({"path": relative.replace("\\", "/"), "sha256": sha256(path.read_bytes())})
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload), records


def dirty_paths(root: Path = ROOT) -> list[str]:
    output = _git("status", "--porcelain=v1", "--untracked-files=all", root=root)
    return sorted(line[3:].replace("\\", "/") for line in output.splitlines() if line)


def generate(root: Path = ROOT, *, release: bool) -> dict[str, object]:
    dirty = dirty_paths(root)
    if release and dirty:
        raise RuntimeError("release manifest requires a clean Git tree")
    source, records = source_digest(root)
    lock_hashes = {
        path: sha256((root / path).read_bytes())
        for path in LOCKS
        if (root / path).is_file()
    }
    if len(lock_hashes) != len(LOCKS):
        raise RuntimeError("dependency locks are missing")
    inputs = {
        path: sha256((root / path).read_bytes())
        for path in BUILD_INPUTS
        if (root / path).is_file()
    }
    return {
        "schema_version": 1,
        "mode": "release" if release else "development",
        "git_sha": _git("rev-parse", "HEAD", root=root),
        "dirty": bool(dirty),
        "dirty_paths": dirty,
        "python": "3.11.9",
        "platform": "linux/amd64",
        "source_manifest_sha256": source,
        "lock_sha256": sha256(
            json.dumps(lock_hashes, sort_keys=True, separators=(",", ":")).encode()
        ),
        "locks": lock_hashes,
        "build_inputs": inputs,
        "source_files": records,
        "model_compatibility_gate": "deferred_to_wp10",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / ".build" / "wp02-manifest.json")
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    try:
        manifest = generate(ROOT, release=args.release)
    except RuntimeError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
