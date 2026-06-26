from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.security.scan_artifacts import scan_path


def _git_status_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        print(result.stderr.strip(), file=sys.stderr)
        return False
    if result.stdout.strip():
        print("Refusing to build release from a dirty working tree.", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        return False
    return True


def _run_git(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _archive_ref(output: Path, *, ref: str) -> int:
    archive_cmd = ["git", "archive", "--format=tar.gz", f"--output={output}", ref]
    print("+ " + " ".join(str(part) for part in archive_cmd))
    return subprocess.run(archive_cmd, cwd=REPO_ROOT, check=False).returncode


def _archive_worktree(output: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="quant-release-index-") as temp_dir:
        temp_index = Path(temp_dir) / "index"
        repo_index = REPO_ROOT / ".git" / "index"
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(temp_index)
        if repo_index.exists():
            shutil.copy2(repo_index, temp_index)
        else:
            command = ["git", "read-tree", "HEAD"]
            print("+ " + " ".join(command))
            result = _run_git(command, env=env)
            if result.returncode:
                if result.stderr.strip():
                    print(result.stderr.strip(), file=sys.stderr)
                return result.returncode

        command = ["git", "add", "-A"]
        print("+ " + " ".join(command))
        result = _run_git(command, env=env)
        if result.returncode:
            if result.stderr.strip():
                print(result.stderr.strip(), file=sys.stderr)
            return result.returncode

        write_tree = ["git", "write-tree"]
        print("+ " + " ".join(write_tree))
        result = _run_git(write_tree, env=env)
        if result.returncode:
            if result.stderr.strip():
                print(result.stderr.strip(), file=sys.stderr)
            return result.returncode
        tree = result.stdout.strip()
        return _archive_ref(output, ref=tree)


def build_release(
    output: Path,
    *,
    ref: str = "HEAD",
    allow_dirty: bool = False,
    from_worktree: bool = False,
) -> int:
    output = output.resolve()
    if from_worktree and not allow_dirty:
        print("--from-worktree requires --allow-dirty.", file=sys.stderr)
        return 2
    if not allow_dirty and not _git_status_clean():
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    archive_returncode = _archive_worktree(output) if from_worktree else _archive_ref(output, ref=ref)
    if archive_returncode:
        return archive_returncode

    findings = scan_path(output)
    if findings:
        output.unlink(missing_ok=True)
        print("Release artifact scan failed; removed unsafe archive.", file=sys.stderr)
        for finding in findings:
            print(f"- {finding.path}: {finding.reason}", file=sys.stderr)
        return 1

    print(f"Release artifact ready: {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a scanned release artifact from tracked files.")
    parser.add_argument("--output", required=True, help="Output .tar.gz path")
    parser.add_argument("--ref", default="HEAD", help="Git ref to archive")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Developer-only escape hatch for tests/implementation; do not use for production release.",
    )
    parser.add_argument(
        "--from-worktree",
        action="store_true",
        help="Developer-only: archive a temporary-index snapshot of current tracked/untracked worktree changes.",
    )
    args = parser.parse_args(argv)
    return build_release(
        Path(args.output),
        ref=args.ref,
        allow_dirty=args.allow_dirty,
        from_worktree=args.from_worktree,
    )


if __name__ == "__main__":
    raise SystemExit(main())
