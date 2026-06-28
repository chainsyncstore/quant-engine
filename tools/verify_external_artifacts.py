#!/usr/bin/env python3
"""Verify pinned Docker Hub tag manifests still resolve to reviewed digests."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request


PINNED_IMAGES = {
    "library/python:3.11.9-slim-bookworm": "sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317",
    "library/redis:7-alpine": "sha256:6ab0b6e7381779332f97b8ca76193e45b0756f38d4c0dcda72dbb3c32061ab99",
}
MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    )
)


def resolved_digest(repository: str, tag: str) -> str:
    scope = urllib.parse.quote(f"repository:{repository}:pull")
    token_url = f"https://auth.docker.io/token?service=registry.docker.io&scope={scope}"
    with urllib.request.urlopen(token_url, timeout=20) as response:
        token = json.load(response)["token"]
    request = urllib.request.Request(
        f"https://registry-1.docker.io/v2/{repository}/manifests/{tag}",
        method="HEAD",
        headers={"Authorization": f"Bearer {token}", "Accept": MANIFEST_ACCEPT},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        digest = response.headers.get("Docker-Content-Digest")
    if not digest:
        raise RuntimeError(f"registry returned no digest for {repository}:{tag}")
    return digest


def verify() -> list[dict[str, str]]:
    records = []
    for reference, expected in sorted(PINNED_IMAGES.items()):
        repository, tag = reference.rsplit(":", 1)
        actual = resolved_digest(repository, tag)
        if actual != expected:
            raise RuntimeError(f"digest mismatch for {reference}: expected {expected}, got {actual}")
        records.append({"reference": reference, "digest": actual, "status": "verified"})
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=argparse.FileType("w"))
    args = parser.parse_args()
    payload = {"schema_version": 1, "images": verify()}
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
