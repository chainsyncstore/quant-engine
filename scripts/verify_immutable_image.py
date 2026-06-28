#!/usr/bin/env python3
"""Validate the Compose application image contract before deployment."""

from __future__ import annotations

import os
import re
import sys


PATTERN = re.compile(r"^[a-z0-9][a-z0-9._/-]*(?::[a-zA-Z0-9._-]+)?@sha256:[0-9a-f]{64}$")


def main() -> int:
    image = os.environ.get("QUANT_IMAGE", "")
    if not PATTERN.fullmatch(image):
        print("QUANT_IMAGE must be an immutable registry reference ending in @sha256:<64 hex>", file=sys.stderr)
        return 2
    print(image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
