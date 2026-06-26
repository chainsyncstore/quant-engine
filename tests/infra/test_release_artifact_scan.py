from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

from tools.security import build_release
from tools.security.scan_artifacts import scan_path
from tools.security.scan_tracked_files import scan_tracked_paths


def _reasons(findings):
    return {finding.reason for finding in findings}


def test_release_scan_accepts_clean_tree(tmp_path: Path) -> None:
    root = tmp_path / "release"
    package = root / "quant"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "README.release.txt").write_text("clean release notes\n", encoding="utf-8")

    assert scan_path(root) == []


def test_release_scan_rejects_secret_state_and_archive_paths(tmp_path: Path) -> None:
    root = tmp_path / "release"
    root.mkdir()
    for name in [
        ".env",
        "quant-key.pem",
        "quant_bot.db",
        "signal_log.json",
        "debug_credentials.py",
        "deploy_latest.tar.gz",
    ]:
        (root / name).write_text("placeholder\n", encoding="utf-8")

    findings = scan_path(root)
    reasons = _reasons(findings)

    assert "denied_path:.env" in reasons
    assert "denied_path:*.pem" in reasons
    assert "denied_path:*.db" in reasons
    assert "denied_path:signal_log.json" in reasons
    assert "denied_path:debug_credentials.py" in reasons
    assert "denied_path:*.tar.gz" in reasons


def test_release_scan_rejects_unsafe_diagnostic_content(tmp_path: Path) -> None:
    script = tmp_path / "ops_probe.py"
    script.write_text(
        "API_KEY = 'sentinel_live_api_key_123456'\n"
        "print(resp.text)\n"
        "r.hgetall('exec:sessions')\n",
        encoding="utf-8",
    )

    findings = scan_path(script)
    reasons = _reasons(findings)

    assert any(reason.startswith("denied_content:") for reason in reasons)
    assert all("sentinel" not in finding.reason for finding in findings)


def test_tracked_file_scan_rejects_high_confidence_secret_content(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    key_file = root / "app.py"
    token = "123456789:" + "abcdefghijklmnopqrstuvwxyzABCDE"
    key_file.write_text(
        f"TOKEN = {token!r}\n",
        encoding="utf-8",
    )

    findings = scan_tracked_paths([key_file], repo_root=root)

    assert findings
    assert findings[0].path == "app.py"
    assert findings[0].reason == "denied_content:telegram_bot_token"
    assert "abcdefghijklmnopqrstuvwxyzABCDE" not in findings[0].reason


def test_tracked_file_scan_rejects_secret_state_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    env_file = root / ".env"
    pem_file = root / "quant-key.pem"
    env_file.write_text("placeholder\n", encoding="utf-8")
    pem_file.write_text("placeholder\n", encoding="utf-8")

    findings = scan_tracked_paths([env_file, pem_file], repo_root=root)
    reasons = _reasons(findings)

    assert "denied_path:.env" in reasons
    assert "denied_path:*.pem" in reasons


def test_release_scan_rejects_archive_members_without_extraction(tmp_path: Path) -> None:
    archive_path = tmp_path / "release.tar.gz"
    safe_file = tmp_path / "app.py"
    env_file = tmp_path / ".env"
    nested_archive = tmp_path / "nested.tar.gz"
    safe_file.write_text("print('ok')\n", encoding="utf-8")
    env_file.write_text("TOKEN=secret\n", encoding="utf-8")
    nested_archive.write_text("not a real archive\n", encoding="utf-8")

    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(safe_file, arcname="app.py")
        archive.add(env_file, arcname=".env")
        archive.add(nested_archive, arcname="nested.tar.gz")

    findings = scan_path(archive_path)
    reasons = _reasons(findings)

    assert "denied_path:.env" in reasons
    assert "denied_path:*.tar.gz" in reasons
    assert all("TOKEN=secret" not in finding.path for finding in findings)


def test_release_scan_rejects_archive_member_unsafe_text_content(tmp_path: Path) -> None:
    archive_path = tmp_path / "release.tar.gz"
    unsafe_script = tmp_path / "ops_probe.py"
    unsafe_script.write_text(
        "API_KEY = 'sentinel_live_api_key_123456'\n"
        "print(resp.text)\n",
        encoding="utf-8",
    )

    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(unsafe_script, arcname="ops_probe.py")

    findings = scan_path(archive_path)
    reasons = _reasons(findings)

    assert any(reason.startswith("denied_content:") for reason in reasons)
    assert all("sentinel" not in finding.reason for finding in findings)
    assert all("sentinel" not in finding.path for finding in findings)


def test_release_scan_allows_benign_secret_name_references(tmp_path: Path) -> None:
    source = tmp_path / "service.py"
    source.write_text(
        "api_key = request.credentials.get('binance_api_key', '').strip()\n"
        "msg = body.get('msg', resp.text)\n"
        "HEADER_RE = r'X-SECURITY-TOKEN'\n",
        encoding="utf-8",
    )

    assert scan_path(source) == []


def test_release_scan_exempts_docs_and_tests_from_diagnostic_content_patterns(
    tmp_path: Path,
) -> None:
    root = tmp_path / "release"
    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True)
    (root / "ROADMAP.md").write_text("r.hgetall('exec:sessions')\n", encoding="utf-8")
    (tests_dir / "test_fixture.py").write_text(
        "API_KEY = 'sentinel_live_api_key_123456'\n"
        "print(resp.text)\n",
        encoding="utf-8",
    )

    assert scan_path(root) == []


def test_release_scan_rejects_model_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "release"
    model_dir = root / "models" / "production"
    model_dir.mkdir(parents=True)
    (model_dir / "model_4m.joblib").write_text("placeholder\n", encoding="utf-8")

    findings = scan_path(root)
    reasons = _reasons(findings)

    assert "denied_path:models/*" in reasons or "denied_path:*/models/*" in reasons
    assert "denied_path:*.joblib" in reasons


def test_release_scan_allows_source_model_package_paths(tmp_path: Path) -> None:
    root = tmp_path / "release"
    source_model_dir = root / "quant_v2" / "models"
    source_model_dir.mkdir(parents=True)
    (source_model_dir / "__init__.py").write_text("", encoding="utf-8")
    (source_model_dir / "trainer.py").write_text("class Trainer: pass\n", encoding="utf-8")

    assert scan_path(root) == []


def test_build_release_rejects_dirty_tree_by_default(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "release.tar.gz"
    monkeypatch.setattr(build_release, "_git_status_clean", lambda: False)

    assert build_release.build_release(output) == 1
    assert not output.exists()


def test_build_release_allows_explicit_dirty_developer_bypass(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "release.tar.gz"
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        with tarfile.open(output, "w:gz") as archive:
            safe = tmp_path / "README.md"
            safe.write_text("release\n", encoding="utf-8")
            archive.add(safe, arcname="README.md")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(build_release.subprocess, "run", fake_run)

    assert build_release.build_release(output, allow_dirty=True) == 0
    assert output.exists()
    assert calls[0][:3] == ["git", "archive", "--format=tar.gz"]


def test_build_release_removes_archive_when_scan_fails(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "release.tar.gz"

    def fake_run(command, **kwargs):
        env_file = tmp_path / ".env"
        env_file.write_text("TOKEN=secret\n", encoding="utf-8")
        with tarfile.open(output, "w:gz") as archive:
            archive.add(env_file, arcname=".env")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(build_release.subprocess, "run", fake_run)

    assert build_release.build_release(output, allow_dirty=True) == 1
    assert not output.exists()


def test_build_release_from_clean_git_commit_accepts_source_model_packages(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    source_model_dir = repo / "quant_v2" / "models"
    source_model_dir.mkdir(parents=True)
    (source_model_dir / "__init__.py").write_text("", encoding="utf-8")
    (source_model_dir / "trainer.py").write_text("class Trainer: pass\n", encoding="utf-8")
    (repo / "README.md").write_text("release\n", encoding="utf-8")

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "clean release"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    output = tmp_path / "release.tar.gz"
    monkeypatch.setattr(build_release, "REPO_ROOT", repo)

    assert build_release.build_release(output) == 0
    assert output.exists()
    assert scan_path(output) == []


def test_build_release_from_worktree_uses_temporary_index_without_mutating_real_index(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "audit_report.md").write_text("legacy report\n", encoding="utf-8")

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "legacy unsafe artifact"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    (repo / "audit_report.md").unlink()
    source_model_dir = repo / "quant_v2" / "models"
    source_model_dir.mkdir(parents=True)
    (source_model_dir / "__init__.py").write_text("", encoding="utf-8")
    (source_model_dir / "trainer.py").write_text("class Trainer: pass\n", encoding="utf-8")

    output = tmp_path / "release.tar.gz"
    monkeypatch.setattr(build_release, "REPO_ROOT", repo)

    assert build_release.build_release(output, allow_dirty=True, from_worktree=True) == 0
    assert output.exists()
    assert scan_path(output) == []

    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert " D audit_report.md" in status
    assert "?? quant_v2/" in status


def test_build_release_from_worktree_honors_current_index_staged_deletions(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("*.tar.gz\n", encoding="utf-8")
    archive = repo / "deploy_optimized.tar.gz"
    archive.write_text("legacy archive\n", encoding="utf-8")

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "-f", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "legacy archive"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "rm", "--cached", "deploy_optimized.tar.gz"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    output = tmp_path / "release.tar.gz"
    monkeypatch.setattr(build_release, "REPO_ROOT", repo)

    assert archive.exists()
    assert build_release.build_release(output, allow_dirty=True, from_worktree=True) == 0
    assert output.exists()
    assert scan_path(output) == []
