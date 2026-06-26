from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import uuid4

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename


def _component_from_requirement_line(raw_line: str) -> dict[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#") or line.startswith("--"):
        return None
    line = line.rstrip("\\").strip()
    try:
        requirement = Requirement(line)
    except Exception:
        return None

    version: str | None = None
    for specifier in requirement.specifier:
        if specifier.operator == "==":
            version = specifier.version
            break

    if requirement.url:
        filename = Path(unquote(urlparse(requirement.url).path)).name
        parsed_name, parsed_version, _, _ = parse_wheel_filename(filename)
        if canonicalize_name(parsed_name) != canonicalize_name(requirement.name):
            raise ValueError(f"lockfile URL package name mismatch: {line}")
        version = str(parsed_version)

    if version is None:
        raise ValueError(f"lockfile requirement has no pinned version: {line}")

    purl_name = canonicalize_name(requirement.name)
    return {
        "type": "library",
        "name": requirement.name,
        "version": version,
        "purl": f"pkg:pypi/{purl_name}@{version}",
    }


def _components(requirements_path: Path) -> list[dict[str, str]]:
    components: list[dict[str, str]] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        component = _component_from_requirement_line(raw_line)
        if component is not None:
            components.append(component)
    return components


def build_sbom(requirements_path: Path) -> dict[str, object]:
    components = _components(requirements_path)
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "component": {
                "type": "application",
                "name": "quant-research-engine",
            },
            "tools": [
                {
                    "vendor": "quant-research-engine",
                    "name": "tools/security/generate_sbom.py",
                }
            ],
        },
        "components": components,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a minimal CycloneDX SBOM from a lockfile.")
    parser.add_argument("--requirements", default="requirements.lock")
    parser.add_argument("--output", default="build/security/sbom.cdx.json")
    args = parser.parse_args(argv)

    requirements_path = Path(args.requirements)
    output_path = Path(args.output)
    sbom = build_sbom(requirements_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path} with {len(sbom['components'])} components")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
