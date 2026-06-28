#!/usr/bin/env python3
"""Bind exact local image identity to WP-02 build and scan evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "image_id": args.image_id,
        "immutable_registry_digest": None,
        "exact_image_certified": False,
        "build_manifest_sha256": digest(args.manifest),
        "evidence": {
            path.name: digest(path) for path in sorted(args.evidence) if path.is_file()
        },
        "promotion_gate": "requires_registry_digest_and_wp15_attestation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
