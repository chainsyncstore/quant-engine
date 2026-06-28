#!/usr/bin/env python3
"""Fail-closed runtime image checks that do not need credentials or models."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


EXPECTED = {
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "scikit-learn": "1.8.0",
    "lightgbm": "4.6.0",
    "joblib": "1.5.3",
    "torch": "2.9.1+cpu",
    "chronos-forecasting": "2.3.0",
}


def run(
    metadata_path: Path,
    require_non_root: bool = True,
    require_release_marker: bool = False,
) -> dict[str, object]:
    if sys.version_info[:3] != (3, 11, 9):
        raise RuntimeError(f"unexpected Python version: {sys.version.split()[0]}")
    if require_non_root and hasattr(os, "geteuid") and os.geteuid() == 0:
        raise RuntimeError("runtime must not run as root")
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    if versions != EXPECTED:
        raise RuntimeError(f"dependency mismatch: {versions!r}")
    for module in ("quant", "quant_v2", "numpy", "pandas", "sklearn", "lightgbm", "joblib"):
        importlib.import_module(module)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in ("git_sha", "source_manifest_sha256", "lock_sha256"):
        if not metadata.get(key):
            raise RuntimeError(f"missing build metadata: {key}")
    prohibited_commands = [name for name in ("uv", "pytest", "ruff") if shutil.which(name)]
    prohibited_paths = [
        path
        for path in (
            Path("/app/tests"),
            Path("/app/requirements/build.lock"),
            Path("/build/requirements/build.lock"),
            Path("/opt/build-tools"),
        )
        if path.exists()
    ]
    if prohibited_commands or prohibited_paths:
        raise RuntimeError(
            f"runtime contains build/test tooling: commands={prohibited_commands}, "
            f"paths={[str(path) for path in prohibited_paths]}"
        )
    marker = Path("/app/.wp02-tests-passed")
    if require_release_marker and not marker.is_file():
        raise RuntimeError("release image is missing the test-stage success marker")
    from sqlalchemy import create_engine, inspect

    from quant.telebot.models import Base

    with tempfile.TemporaryDirectory() as temporary:
        database = Path(temporary) / "readiness.db"
        engine = create_engine(f"sqlite:///{database}")
        Base.metadata.create_all(engine)
        if not inspect(engine).get_table_names():
            raise RuntimeError("database readiness schema created no tables")
        engine.dispose()
    return {
        "status": "ok",
        "uid": os.geteuid() if hasattr(os, "geteuid") else None,
        "versions": versions,
        "database_readiness": "application_schema_create_all_ok",
        "model_compatibility": "deferred_to_wp10",
        "runtime_exclusions": "uv_pytest_ruff_tests_build_locks_absent",
        "test_stage_marker": marker.is_file(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("/app/build-manifest.json"))
    parser.add_argument("--allow-root", action="store_true", help="local test mode only")
    parser.add_argument("--require-release-marker", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.metadata, not args.allow_root, args.require_release_marker),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
