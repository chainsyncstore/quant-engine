#!/usr/bin/env python3
"""Collect read-only WP-01 source provenance from the configured Ubuntu host."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import posixpath
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


REPO = "/home/admin-4arm/hypothesis-research-engine"
APP_CONTAINER = "quant_telegram"
APP_CONTAINERS = ("quant_telegram", "quant_model_eval", "quant_retrain")
ROOT_FILES = (
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.override.yml",
    "docker-compose.prod.yml",
    "pyproject.toml",
    "bootstrap_registry.py",
    "requirements.in",
    "requirements.lock",
    "requirements-ci.in",
    "requirements-ci.lock",
)
CRITICAL_FILES = (
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.override.yml",
    "pyproject.toml",
    "bootstrap_registry.py",
    "quant/telebot/main.py",
    "quant_v2/data/multi_symbol_dataset.py",
    "quant_v2/execution/adapters.py",
    "quant_v2/execution/service.py",
    "quant_v2/model_registry.py",
    "quant_v2/portfolio/risk_policy.py",
    "quant_v2/research/model_evaluator.py",
    "quant_v2/research/scheduled_retrain.py",
)
RUNTIME_CRITICAL_FILES = tuple(path for path in CRITICAL_FILES if path.startswith(("quant/", "quant_v2/")))


def run_ssh(host: str, script: str, *, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, "sh", "-s"],
        input=script.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        error = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"read-only SSH command failed ({result.returncode}): {error}")
    return result.stdout if binary else result.stdout.decode("utf-8", "strict")


def safe_member_name(name: str) -> str:
    normalized = posixpath.normpath(name.removeprefix("./"))
    path = PurePosixPath(normalized)
    if not normalized or normalized == "." or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name!r}")
    return normalized


def normalize_tar(raw: bytes, destination: Path) -> dict[str, dict[str, object]]:
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as source:
        for member in source:
            name = safe_member_name(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"non-regular archive member rejected: {member.name!r}")
            if name in files:
                raise ValueError(f"duplicate archive member: {name!r}")
            extracted = source.extractfile(member)
            if extracted is None:
                raise ValueError(f"unreadable archive member: {name!r}")
            files[name] = extracted.read()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw_target:
        with gzip.GzipFile(fileobj=raw_target, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as target:
                for name in sorted(files):
                    data = files[name]
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    target.addfile(info, io.BytesIO(data))

    return {
        name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        for name, data in sorted(files.items())
    }


def source_tar_script(prefix: str = "") -> str:
    root = f"{prefix}{REPO}" if prefix else REPO
    if prefix:
        root = "/app"
    quoted_roots = " ".join(f"'{name}'" for name in ROOT_FILES)
    return f"""set -eu
cd '{root}'
{{
  find quant quant_v2 -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0
  if [ -d deploy ]; then find deploy -type f \\( -name '*.sh' -o -name '*.service' -o -name '*.yml' -o -name '*.yaml' \\) -print0; fi
  for path in {quoted_roots}; do [ ! -f "$path" ] || printf '%s\\0' "$path"; done
}} | LC_ALL=C sort -zu | tar --no-recursion --null --files-from=- --create --file=-
"""


def host_metadata_script() -> str:
    critical = " ".join(f"'{path}'" for path in CRITICAL_FILES)
    return f"""set -eu
repo='{REPO}'
printf '%s\\n' 'WP01_HEAD'
git -C "$repo" rev-parse HEAD
printf '%s\\n' 'WP01_BRANCH'
git -C "$repo" branch --show-current
printf '%s\\n' 'WP01_STATUS_BEGIN'
git -C "$repo" -c core.quotepath=true status --porcelain=v1 --untracked-files=all
printf '%s\\n' 'WP01_STATUS_END'
printf '%s\\n' 'WP01_DIFF_STAT_BEGIN'
git -C "$repo" diff --stat --no-ext-diff
git -C "$repo" diff --cached --stat --no-ext-diff
printf '%s\\n' 'WP01_DIFF_STAT_END'
printf '%s\\n' 'WP01_CRITICAL_HASHES_BEGIN'
cd "$repo"
for path in {critical}; do [ ! -f "$path" ] || sha256sum -- "$path"; done
printf '%s\\n' 'WP01_CRITICAL_HASHES_END'
printf '%s\\n' 'WP01_DOCKER_BEGIN'
for container in {' '.join(APP_CONTAINERS)}; do
  docker inspect "$container" --format '{{{{json .Name}}}}|{{{{json .Id}}}}|{{{{json .Image}}}}|{{{{json .Config.WorkingDir}}}}|{{{{json .Config.Labels}}}}'
done
docker image inspect quant_bot:latest --format '{{{{json .Id}}}}|{{{{json .RepoDigests}}}}|{{{{json .Created}}}}|{{{{json .Os}}}}|{{{{json .Architecture}}}}|{{{{json .Config.WorkingDir}}}}|{{{{json .Config.Labels}}}}'
printf '%s\\n' 'WP01_DOCKER_END'
printf '%s\\n' 'WP01_CONTAINER_HASHES_BEGIN'
for container in {' '.join(APP_CONTAINERS)}; do
  for path in {critical}; do
    docker exec "$container" sh -c '[ ! -f "/app/$1" ] || sha256sum -- "/app/$1"' sh "$path" | sed "s#  /app/#  $container:#"
  done
done
printf '%s\\n' 'WP01_CONTAINER_HASHES_END'
"""


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="4arm-ubuntu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ubuntu_audit_20260622/source_provenance"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    metadata = run_ssh(args.host, host_metadata_script())
    (output / "production_metadata.txt").write_text(metadata, encoding="utf-8", newline="\n")

    host_raw = run_ssh(args.host, source_tar_script(), binary=True)
    host_manifest = normalize_tar(host_raw, output / "host_source.tar.gz")

    container_script = f"docker exec -i {APP_CONTAINER} sh -s <<'WP01'\n{source_tar_script('/app')}WP01\n"
    container_raw = run_ssh(args.host, container_script, binary=True)
    container_manifest = normalize_tar(container_raw, output / "container_source.tar.gz")

    write_json(output / "host_archive_manifest.json", host_manifest)
    write_json(output / "container_archive_manifest.json", container_manifest)
    write_json(
        output / "collection_scope.json",
        {
            "app_container": APP_CONTAINER,
            "app_containers_hashed": list(APP_CONTAINERS),
            "critical_files": list(CRITICAL_FILES),
            "runtime_critical_files": list(RUNTIME_CRITICAL_FILES),
            "excluded_categories": [
                "credentials and environment files",
                "private keys",
                "databases and state",
                "logs",
                "models and registry artifacts",
                "datasets and experiments",
                "archives, caches, bytecode, and generated files",
            ],
            "host": args.host,
            "remote_repo": REPO,
            "root_files": list(ROOT_FILES),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
