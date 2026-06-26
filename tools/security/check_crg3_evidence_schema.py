from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.security.finalize_crg3 import (
    CRG3_EVIDENCE_SCHEMA_VERSION,
    _expected_crg3_evidence_checks,
    _expected_runtime_sbom_components,
    _runtime_lock_metadata,
    _validate_crg3_evidence_record,
)


SAMPLE_SHA = "0123456789abcdef0123456789abcdef01234567"


def main() -> int:
    sample = {
        "checks": _expected_crg3_evidence_checks(),
        "commit_sha": SAMPLE_SHA,
        "ledger_id": "CRG3",
        "local_head": SAMPLE_SHA,
        "local_working_tree": "clean",
        "repository": "chainsyncstore/hypothesis-research-engine",
        "repository_binding_source": "origin fetch remote",
        "run_attempt": 1,
        "run_id": "123456789",
        "run_url": "https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
        "runtime_lock": _runtime_lock_metadata(),
        "schema": "docs/CRG3_EVIDENCE_SCHEMA.json",
        "schema_version": CRG3_EVIDENCE_SCHEMA_VERSION,
        "sbom_artifact": "production-sbom",
        "sbom_artifact_metadata": {
            "artifact_sha256": "0" * 64,
            "component_count": len(_expected_runtime_sbom_components()),
            "download_url": (
                "https://api.github.com/repos/chainsyncstore/"
                "hypothesis-research-engine/actions/artifacts/42/zip"
            ),
            "download_size_in_bytes": 2048,
            "id": 42,
            "sbom_sha256": "0" * 64,
            "size_in_bytes": 2048,
        },
        "verified_date": "2026-06-04",
        "workflow": {
            "job": "readiness",
            "live_execution_env": "0",
            "name": "Production Readiness",
            "path": ".github/workflows/production-readiness.yml",
            "sbom_path": "build/security/sbom.cdx.json",
            "wrapper_command": "python tools/security/production_readiness.py --profile ci",
        },
    }
    _validate_crg3_evidence_record(sample)
    print("CRG3 evidence schema validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
