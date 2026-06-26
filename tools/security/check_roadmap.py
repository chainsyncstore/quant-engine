from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


LEDGER_HEADER = "| ID | Pass | Priority | Status | Work item | Files | Proof required |"
VERIFIED_RE = re.compile(r"^Verified \d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class LedgerRow:
    row_id: str
    pass_name: str
    priority: str
    status: str
    work_item: str
    files: str
    proof: str


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_ledger_text(text: str, *, source: str = "<roadmap>") -> list[LedgerRow]:
    lines = text.splitlines()
    try:
        header_index = lines.index(LEDGER_HEADER)
    except ValueError as exc:
        raise ValueError(f"ledger header not found in {source}") from exc

    rows: list[LedgerRow] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("| "):
            break
        cells = _split_markdown_row(line)
        if len(cells) != 7:
            raise ValueError(f"malformed ledger row: {line}")
        rows.append(LedgerRow(*cells))
    return rows


def parse_ledger(path: Path) -> list[LedgerRow]:
    return parse_ledger_text(path.read_text(encoding="utf-8"), source=str(path))


def validate_p0_rows(rows: list[LedgerRow], *, allow_open_ids: set[str] | None = None) -> list[str]:
    allow_open_ids = allow_open_ids or set()
    errors: list[str] = []
    for row in rows:
        if row.priority != "P0":
            continue
        if row.row_id in allow_open_ids:
            continue
        prefix = f"{row.row_id}:"
        if not VERIFIED_RE.match(row.status):
            errors.append(f"{prefix} P0 status must be 'Verified YYYY-MM-DD', got {row.status!r}")
        if not row.files or row.files in {"-", "`-`"}:
            errors.append(f"{prefix} missing files/evidence scope")
        proof = row.proof.lower()
        if not row.proof or re.search(r"\b(todo|ready)\b", proof):
            errors.append(f"{prefix} proof field is missing or not final evidence")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate production refactor roadmap ledger evidence.")
    parser.add_argument("roadmap", nargs="?", default="PRODUCTION_REFACTOR_ROADMAP.md")
    parser.add_argument(
        "--allow-open-id",
        action="append",
        default=[],
        help="Allow a specific P0 ledger ID to remain open for self-bootstrapping CI evidence.",
    )
    args = parser.parse_args(argv)

    try:
        rows = parse_ledger(Path(args.roadmap))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    errors = validate_p0_rows(rows, allow_open_ids=set(args.allow_open_id))
    if errors:
        print("Roadmap P0 ledger validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Roadmap P0 ledger validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
