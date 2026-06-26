from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from tools.security.check_roadmap import parse_ledger_text, validate_p0_rows


REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP = REPO_ROOT / "PRODUCTION_REFACTOR_ROADMAP.md"
CRG3_PREFIX = "| CRG3 | CI, Regression, and Production Readiness Gate | P0 |"
REQUIRED_SBOM_ARTIFACT = "production-sbom"
GITHUB_OWNER_PATTERN = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
GITHUB_REPO_PATTERN = r"[A-Za-z0-9._-]+"
GITHUB_RUN_RE = re.compile(
    rf"^https://github\.com/(?P<owner>{GITHUB_OWNER_PATTERN})/"
    rf"(?P<repo>{GITHUB_REPO_PATTERN})/actions/runs/\d+/?$"
)
GITHUB_REMOTE_PATTERNS = (
    re.compile(
        rf"^https://github\.com/(?P<owner>{GITHUB_OWNER_PATTERN})/"
        rf"(?P<repo>{GITHUB_REPO_PATTERN}?)(?:\.git)?$"
    ),
    re.compile(
        rf"^git@github\.com:(?P<owner>{GITHUB_OWNER_PATTERN})/"
        rf"(?P<repo>{GITHUB_REPO_PATTERN}?)(?:\.git)?$"
    ),
)
EXPECTED_WORKFLOW_NAME = "Production Readiness"
EXPECTED_WORKFLOW_PATH = ".github/workflows/production-readiness.yml"
EXPECTED_WORKFLOW_JOB = "readiness"
EXPECTED_WORKFLOW_WRAPPER_STEP = "Run production readiness wrapper"
EXPECTED_WORKFLOW_WRAPPER_COMMAND = "python tools/security/production_readiness.py --profile ci"
EXPECTED_WORKFLOW_UPLOAD_STEP = "Upload SBOM"
EXPECTED_WORKFLOW_UPLOAD_ACTION = "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
EXPECTED_WORKFLOW_SBOM_PATH = "build/security/sbom.cdx.json"
EXPECTED_WORKFLOW_CHECKOUT_STEP = "Check out repository"
EXPECTED_WORKFLOW_CHECKOUT_ACTION = "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5"
EXPECTED_WORKFLOW_SETUP_STEP = "Set up Python"
EXPECTED_WORKFLOW_SETUP_ACTION = "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065"
EXPECTED_WORKFLOW_CI_INSTALL_STEP = "Install pinned CI tools"
EXPECTED_WORKFLOW_CI_INSTALL_COMMAND = "python -m pip install --require-hashes -r requirements-ci.lock"
EXPECTED_WORKFLOW_RUNTIME_INSTALL_STEP = "Install locked runtime dependencies"
EXPECTED_WORKFLOW_RUNTIME_INSTALL_COMMAND = "python -m pip install --require-hashes -r requirements.lock"
EXPECTED_RUNTIME_REQUIREMENTS_LOCK = REPO_ROOT / "requirements.lock"
EXPECTED_WORKFLOW_RUNNER = "ubuntu-latest"
EXPECTED_WORKFLOW_TIMEOUT_MINUTES = 45
EXPECTED_WORKFLOW_CONCURRENCY_GROUP = "production-readiness-${{ github.ref }}"
CRG3_EVIDENCE_SCHEMA = "docs/CRG3_EVIDENCE_SCHEMA.json"
CRG3_EVIDENCE_SCHEMA_VERSION = 9
APPROVED_FINALIZER_EVENTS = {"push", "workflow_dispatch"}
APPROVED_RELEASE_BRANCHES = {"main", "master"}
MAX_GITHUB_ARTIFACT_PAGES = 10
MAX_GITHUB_ARTIFACTS_PER_PAGE = 100
MAX_GITHUB_JSON_BYTES = 1024 * 1024
MAX_SBOM_ARTIFACT_BYTES = 5 * 1024 * 1024
MAX_SBOM_JSON_BYTES = 2 * 1024 * 1024
MAX_SBOM_ZIP_MEMBERS = 64
CRG3_EVIDENCE_ONLY_CHECKS = (
    "production-sbom artifact upload",
    "downloaded CycloneDX SBOM artifact validation",
)


@dataclass(frozen=True)
class RunRef:
    owner: str
    repo: str
    run_id: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def run_api_url(self) -> str:
        return f"https://api.github.com/repos/{self.slug}/actions/runs/{self.run_id}"

    @property
    def html_url(self) -> str:
        return f"https://github.com/{self.slug}/actions/runs/{self.run_id}"

    @property
    def artifacts_api_url(self) -> str:
        return f"https://api.github.com/repos/{self.slug}/actions/runs/{self.run_id}/artifacts"


@dataclass(frozen=True)
class RepositoryApproval:
    repo: str
    source: str


@dataclass(frozen=True)
class GitHubRunEvidence:
    run_attempt: int
    artifact_id: int
    artifact_size_in_bytes: int
    artifact_download_size_in_bytes: int
    artifact_download_url: str
    artifact_sha256: str
    sbom_sha256: str
    sbom_component_count: int


GitHubApiFetcher = Callable[[str], dict[str, Any]]
GitHubBytesFetcher = Callable[[str], bytes]
SbomComponent = tuple[str, str, str, str]


def _remove_request_header(request: urllib.request.Request, header_name: str) -> None:
    normalized = header_name.lower()
    for header_store in (request.headers, request.unredirected_hdrs):
        for key in list(header_store):
            if key.lower() == normalized:
                del header_store[key]


class _CrossHostAuthStrippingRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None

        original = urllib.parse.urlsplit(req.full_url)
        target = urllib.parse.urlsplit(redirected.full_url)
        if (target.scheme, target.netloc.lower()) != (original.scheme, original.netloc.lower()):
            _remove_request_header(redirected, "authorization")
        return redirected


def _github_urlopen(request: urllib.request.Request, *, timeout: int):
    opener = urllib.request.build_opener(_CrossHostAuthStrippingRedirectHandler())
    return opener.open(request, timeout=timeout)


def _artifact_page_url(url: str, *, page: int) -> str:
    parsed = urllib.parse.urlsplit(url)
    query = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if key not in {"per_page", "page"}
    ]
    query.extend([("per_page", str(MAX_GITHUB_ARTIFACTS_PER_PAGE)), ("page", str(page))])
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urllib.parse.urlencode(query),
            parsed.fragment,
        )
    )


def _run_ref_from_url(run_url: str) -> RunRef:
    match = GITHUB_RUN_RE.fullmatch(run_url)
    if not match:
        raise ValueError("run URL must look like https://github.com/<owner>/<repo>/actions/runs/<id>")
    return RunRef(
        owner=match.group("owner"),
        repo=match.group("repo"),
        run_id=run_url.rstrip("/").rsplit("/", 1)[-1],
    )


def _repo_from_remote_url(remote_url: str) -> str | None:
    for pattern in GITHUB_REMOTE_PATTERNS:
        match = pattern.fullmatch(remote_url.strip())
        if match:
            return f"{match.group('owner')}/{match.group('repo')}"
    return None


def _allowed_github_repos(*, repo_root: Path = REPO_ROOT) -> set[str]:
    result = subprocess.run(
        ["git", "remote", "-v"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("could not read git remotes for CRG3 repository validation")

    repos: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "origin" and parts[2] == "(fetch)":
            repo = _repo_from_remote_url(parts[1])
            if repo:
                repos.add(repo)
    if not repos:
        raise ValueError("no GitHub origin fetch remote found for CRG3 repository validation")
    return repos


def _validate_run_url(
    run_url: str, *, allowed_repos: set[str] | None = None
) -> RepositoryApproval:
    run_ref = _run_ref_from_url(run_url)
    approved_repos = allowed_repos if allowed_repos is not None else _allowed_github_repos()
    if run_ref.slug not in approved_repos:
        approved = ", ".join(sorted(approved_repos))
        raise ValueError(f"run URL repository {run_ref.slug!r} is not approved; expected one of: {approved}")
    source = "explicit allowed repository set" if allowed_repos is not None else "origin fetch remote"
    return RepositoryApproval(repo=run_ref.slug, source=source)


def _validate_sha(commit_sha: str) -> None:
    if not re.fullmatch(r"^[0-9a-fA-F]{40}$", commit_sha):
        raise ValueError("commit SHA must be a 40-character hex SHA")


def _current_git_head(*, repo_root: Path = REPO_ROOT) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("could not read local git HEAD for CRG3 commit validation")
    head = result.stdout.strip()
    _validate_sha(head)
    return head


def _validate_local_head(commit_sha: str, *, local_head: str | None = None) -> str:
    head = local_head or _current_git_head()
    _validate_sha(head)
    if head.lower() != commit_sha.lower():
        raise ValueError("local git HEAD does not match supplied CRG3 commit SHA")
    return head


def _git_status_short(*, repo_root: Path = REPO_ROOT) -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("could not read local git status for CRG3 clean-checkout validation")
    return result.stdout


def _validate_clean_checkout(*, git_status: str | None = None) -> None:
    status = _git_status_short() if git_status is None else git_status
    if status.strip():
        raise ValueError("local working tree is not clean; CRG3 finalization requires a clean checkout")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_local_workflow(workflow_path: Path) -> tuple[str, dict[str, Any]]:
    try:
        text = workflow_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError("local production readiness workflow file is missing") from exc
    try:
        workflow = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError("local production readiness workflow YAML is invalid") from exc
    if not isinstance(workflow, dict):
        raise ValueError("local production readiness workflow must be a YAML object")
    return text, workflow


def _workflow_env_value_is_disabled(value: Any) -> bool:
    return isinstance(value, str) and value.strip() == "0"


def _workflow_step_by_name(
    steps: list[Any],
    name: str,
    *,
    description: str,
) -> tuple[int, dict[str, Any]]:
    matches = [
        (index, step)
        for index, step in enumerate(steps)
        if isinstance(step, dict) and step.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"local production readiness workflow must have exactly one {description} step"
        )
    return matches[0]


def _require_step_before(first: int, second: int, *, message: str) -> None:
    if first >= second:
        raise ValueError(message)


def _workflow_triggers(workflow: dict[str, Any]) -> Any:
    if "on" in workflow:
        return workflow["on"]
    return workflow.get(True)


def _validate_local_workflow_execution_controls(workflow: dict[str, Any]) -> None:
    if workflow.get("name") != EXPECTED_WORKFLOW_NAME:
        raise ValueError("local production readiness workflow name must be Production Readiness")

    triggers = _workflow_triggers(workflow)
    if not isinstance(triggers, dict):
        raise ValueError("local production readiness workflow triggers are malformed")
    if "workflow_dispatch" not in triggers:
        raise ValueError("local production readiness workflow must support workflow_dispatch")
    if "pull_request" not in triggers:
        raise ValueError("local production readiness workflow must keep pull_request probes")
    push = triggers.get("push")
    if not isinstance(push, dict) or push.get("branches") != ["main", "master"]:
        raise ValueError("local production readiness workflow push branches must be main and master")

    permissions = workflow.get("permissions")
    if permissions != {"contents": "read"}:
        raise ValueError("local production readiness workflow permissions must be contents: read only")

    concurrency = workflow.get("concurrency")
    if not isinstance(concurrency, dict):
        raise ValueError("local production readiness workflow concurrency is malformed")
    if concurrency.get("group") != EXPECTED_WORKFLOW_CONCURRENCY_GROUP:
        raise ValueError("local production readiness workflow concurrency group is invalid")
    if concurrency.get("cancel-in-progress") is not False:
        raise ValueError("local production readiness workflow must not cancel in-progress evidence")


def _validate_local_workflow_live_disabled(
    workflow_path: Path = REPO_ROOT / EXPECTED_WORKFLOW_PATH,
) -> None:
    text, workflow = _load_local_workflow(workflow_path)

    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        raise ValueError("local production readiness workflow must define jobs")
    readiness_job = jobs.get(EXPECTED_WORKFLOW_JOB)
    if not isinstance(readiness_job, dict):
        raise ValueError("local production readiness workflow must define readiness job")
    if readiness_job.get("runs-on") != EXPECTED_WORKFLOW_RUNNER:
        raise ValueError("local production readiness workflow readiness job must run on ubuntu-latest")
    if readiness_job.get("timeout-minutes") != EXPECTED_WORKFLOW_TIMEOUT_MINUTES:
        raise ValueError("local production readiness workflow readiness job timeout must be 45 minutes")

    job_env = readiness_job.get("env")
    if not isinstance(job_env, dict) or not _workflow_env_value_is_disabled(
        job_env.get("BOT_V2_ALLOW_LIVE_EXECUTION")
    ):
        raise ValueError(
            "local production readiness workflow readiness job must set "
            "BOT_V2_ALLOW_LIVE_EXECUTION to string '0'"
        )

    steps = readiness_job.get("steps")
    if not isinstance(steps, list):
        raise ValueError("local production readiness workflow readiness job must define steps")

    checkout_index, checkout_step = _workflow_step_by_name(
        steps,
        EXPECTED_WORKFLOW_CHECKOUT_STEP,
        description="checkout",
    )
    if checkout_step.get("uses") != EXPECTED_WORKFLOW_CHECKOUT_ACTION:
        raise ValueError("local production readiness workflow checkout action must be pinned")
    checkout_with = checkout_step.get("with")
    if not isinstance(checkout_with, dict) or checkout_with.get("persist-credentials") is not False:
        raise ValueError(
            "local production readiness workflow checkout must set persist-credentials false"
        )

    setup_index, setup_step = _workflow_step_by_name(
        steps,
        EXPECTED_WORKFLOW_SETUP_STEP,
        description="Python setup",
    )
    if setup_step.get("uses") != EXPECTED_WORKFLOW_SETUP_ACTION:
        raise ValueError("local production readiness workflow setup-python action must be pinned")
    setup_with = setup_step.get("with")
    if not isinstance(setup_with, dict):
        raise ValueError("local production readiness workflow setup-python step must define with")
    if str(setup_with.get("python-version", "")) != "3.11":
        raise ValueError("local production readiness workflow setup-python must use Python 3.11")
    if setup_with.get("cache") != "pip":
        raise ValueError("local production readiness workflow setup-python must enable pip cache")

    ci_install_index, ci_install_step = _workflow_step_by_name(
        steps,
        EXPECTED_WORKFLOW_CI_INSTALL_STEP,
        description="CI tool install",
    )
    if ci_install_step.get("run") != EXPECTED_WORKFLOW_CI_INSTALL_COMMAND:
        raise ValueError("local production readiness workflow must install hashed CI tools")

    runtime_install_index, runtime_install_step = _workflow_step_by_name(
        steps,
        EXPECTED_WORKFLOW_RUNTIME_INSTALL_STEP,
        description="runtime dependency install",
    )
    if runtime_install_step.get("run") != EXPECTED_WORKFLOW_RUNTIME_INSTALL_COMMAND:
        raise ValueError("local production readiness workflow must install hashed runtime dependencies")

    wrapper_index, wrapper_step = _workflow_step_by_name(
        steps,
        EXPECTED_WORKFLOW_WRAPPER_STEP,
        description="wrapper",
    )
    if wrapper_step.get("run") != EXPECTED_WORKFLOW_WRAPPER_COMMAND:
        raise ValueError("local production readiness workflow wrapper step must run CI profile")

    _require_step_before(
        checkout_index,
        setup_index,
        message="local production readiness workflow checkout must run before Python setup",
    )
    _require_step_before(
        setup_index,
        ci_install_index,
        message="local production readiness workflow Python setup must run before installs",
    )
    _require_step_before(
        ci_install_index,
        runtime_install_index,
        message="local production readiness workflow CI tools must install before runtime dependencies",
    )
    _require_step_before(
        runtime_install_index,
        wrapper_index,
        message="local production readiness workflow runtime dependencies must install before wrapper",
    )

    step_env = wrapper_step.get("env")
    if isinstance(step_env, dict) and "BOT_V2_ALLOW_LIVE_EXECUTION" in step_env:
        if not _workflow_env_value_is_disabled(step_env.get("BOT_V2_ALLOW_LIVE_EXECUTION")):
            raise ValueError(
                "local production readiness workflow wrapper step must not enable live execution"
            )
    elif step_env is not None:
        raise ValueError("local production readiness workflow wrapper step env is malformed")

    for step in steps:
        if not isinstance(step, dict):
            continue
        step_env = step.get("env")
        if not isinstance(step_env, dict) or "BOT_V2_ALLOW_LIVE_EXECUTION" not in step_env:
            continue
        if not _workflow_env_value_is_disabled(step_env.get("BOT_V2_ALLOW_LIVE_EXECUTION")):
            step_name = step.get("name") or "<unnamed>"
            raise ValueError(
                "local production readiness workflow step "
                f"{step_name!r} must not override BOT_V2_ALLOW_LIVE_EXECUTION"
            )

    upload_index, upload_step = _workflow_step_by_name(
        steps,
        EXPECTED_WORKFLOW_UPLOAD_STEP,
        description="SBOM upload",
    )
    _require_step_before(
        wrapper_index,
        upload_index,
        message="local production readiness workflow wrapper must run before SBOM upload",
    )
    if upload_step.get("uses") != EXPECTED_WORKFLOW_UPLOAD_ACTION:
        raise ValueError("local production readiness workflow SBOM upload action must be pinned")
    if upload_step.get("if") != "always()":
        raise ValueError("local production readiness workflow SBOM upload step must run with always()")

    upload_with = upload_step.get("with")
    if not isinstance(upload_with, dict):
        raise ValueError("local production readiness workflow SBOM upload step must define with")
    if upload_with.get("name") != REQUIRED_SBOM_ARTIFACT:
        raise ValueError("local production readiness workflow SBOM artifact name must be production-sbom")
    if upload_with.get("path") != EXPECTED_WORKFLOW_SBOM_PATH:
        raise ValueError("local production readiness workflow SBOM artifact path is invalid")
    if upload_with.get("if-no-files-found") != "error":
        raise ValueError("local production readiness workflow SBOM upload must fail when missing")

    _validate_local_workflow_execution_controls(workflow)


def _github_request(url: str, *, max_bytes: int) -> bytes:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    try:
        with _github_urlopen(request, timeout=30) as response:
            payload = response.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"GitHub API verification failed for {url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"GitHub API verification failed for {url}: {exc.reason}") from exc
    if len(payload) > max_bytes:
        raise ValueError(f"GitHub API verification failed for {url}: response is too large")
    return payload


def _github_api_json(url: str) -> dict[str, Any]:
    try:
        payload = _github_request(url, max_bytes=MAX_GITHUB_JSON_BYTES).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"GitHub API verification failed for {url}: response is not UTF-8") from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"GitHub API verification failed for {url}: invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError(f"GitHub API verification failed for {url}: expected object response")
    return data


def _github_api_bytes(url: str) -> bytes:
    return _github_request(url, max_bytes=MAX_SBOM_ARTIFACT_BYTES)


def _expected_runtime_sbom_components(
    requirements_path: Path = EXPECTED_RUNTIME_REQUIREMENTS_LOCK,
) -> list[dict[str, str]]:
    from tools.security.generate_sbom import _components

    components = _components(requirements_path)
    if not components:
        raise ValueError("requirements.lock does not define runtime components for SBOM validation")
    return components


def _runtime_lock_metadata(
    requirements_path: Path = EXPECTED_RUNTIME_REQUIREMENTS_LOCK,
) -> dict[str, str]:
    return {
        "path": "requirements.lock",
        "sha256": hashlib.sha256(requirements_path.read_bytes()).hexdigest(),
    }


def _sbom_component_identity(component: dict[str, Any]) -> SbomComponent:
    return (
        str(component.get("type")),
        str(component.get("name")),
        str(component.get("version")),
        str(component.get("purl")),
    )


def _validate_sbom_matches_runtime_lock(components: list[dict[str, Any]]) -> None:
    expected = {
        _sbom_component_identity(component)
        for component in _expected_runtime_sbom_components()
    }
    actual_components = [_sbom_component_identity(component) for component in components]
    duplicates = sorted(
        {component for component in actual_components if actual_components.count(component) > 1}
    )
    if duplicates:
        duplicate_names = ", ".join(
            f"{name}=={version}" for _, name, version, _ in duplicates[:10]
        )
        raise ValueError(
            "production-sbom artifact components must be unique "
            f"(duplicates: {duplicate_names})"
        )
    actual = set(actual_components)

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(
                "missing locked components: "
                + ", ".join(f"{name}=={version}" for _, name, version, _ in missing[:10])
            )
        if unexpected:
            details.append(
                "unexpected components: "
                + ", ".join(f"{name}=={version}" for _, name, version, _ in unexpected[:10])
            )
        raise ValueError(
            "production-sbom artifact components must match requirements.lock exactly"
            f" ({'; '.join(details)})"
        )


def _verify_sbom_artifact_payload(
    *, download_url: str, fetch_bytes: GitHubBytesFetcher
) -> tuple[int, str, str, int]:
    try:
        payload = fetch_bytes(download_url)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"GitHub SBOM artifact download failed for {download_url}: {exc.__class__.__name__}"
        ) from exc
    if len(payload) > MAX_SBOM_ARTIFACT_BYTES:
        raise ValueError("production-sbom artifact download is too large")

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) > MAX_SBOM_ZIP_MEMBERS:
                raise ValueError("production-sbom artifact contains too many zip members")
            candidate_names = [
                member.filename
                for member in members
                if Path(member.filename).name == "sbom.cdx.json" and not member.filename.endswith("/")
            ]
            if not candidate_names:
                raise ValueError("production-sbom artifact does not contain sbom.cdx.json")
            if len(candidate_names) != 1:
                raise ValueError("production-sbom artifact must contain exactly one sbom.cdx.json")
            with archive.open(candidate_names[0]) as sbom_file:
                sbom_payload = sbom_file.read(MAX_SBOM_JSON_BYTES + 1)
                if len(sbom_payload) > MAX_SBOM_JSON_BYTES:
                    raise ValueError("production-sbom sbom.cdx.json is too large")
                try:
                    sbom_text = sbom_payload.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError("production-sbom artifact sbom.cdx.json is not UTF-8") from exc
                sbom = json.loads(sbom_text)
    except zipfile.BadZipFile as exc:
        raise ValueError("production-sbom artifact download is not a zip archive") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("production-sbom artifact does not contain valid JSON") from exc

    if not isinstance(sbom, dict):
        raise ValueError("production-sbom artifact JSON must be an object")
    if sbom.get("bomFormat") != "CycloneDX":
        raise ValueError("production-sbom artifact must be a CycloneDX SBOM")
    components = sbom.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("production-sbom artifact must include non-empty components")
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("production-sbom artifact components must be objects")
        for key in ("type", "name", "version", "purl"):
            if not isinstance(component.get(key), str) or not component.get(key):
                raise ValueError(
                    "production-sbom artifact components must include type, name, version, and purl"
                )
    _validate_sbom_matches_runtime_lock(components)
    return (
        len(payload),
        hashlib.sha256(payload).hexdigest(),
        hashlib.sha256(sbom_payload).hexdigest(),
        len(components),
    )


def _validate_artifact_download_url(download_url: str, *, run_ref: RunRef) -> int:
    parsed = urllib.parse.urlsplit(download_url)
    if parsed.scheme != "https" or parsed.netloc != "api.github.com":
        raise ValueError("GitHub production-sbom artifact download URL must use https://api.github.com")
    if parsed.query or parsed.fragment:
        raise ValueError("GitHub production-sbom artifact download URL must be canonical")

    prefix = f"/repos/{run_ref.slug}/actions/artifacts/"
    if not parsed.path.startswith(prefix) or not parsed.path.endswith("/zip"):
        raise ValueError(
            "GitHub production-sbom artifact download URL must belong to the workflow repository"
        )

    artifact_id = parsed.path[len(prefix) : -len("/zip")]
    if not artifact_id.isdigit():
        raise ValueError("GitHub production-sbom artifact download URL has an invalid artifact id")
    return int(artifact_id)


def _validate_run_artifacts_url(artifacts_url: Any, *, run_ref: RunRef) -> str:
    if not isinstance(artifacts_url, str) or not artifacts_url:
        raise ValueError("GitHub run artifacts_url is missing")
    if artifacts_url.rstrip("/") != run_ref.artifacts_api_url:
        raise ValueError("GitHub run artifacts_url does not match supplied workflow run URL")
    return artifacts_url


def _validate_artifact_size_metadata(artifact: dict[str, Any]) -> int:
    size = artifact.get("size_in_bytes")
    if type(size) is not int or size <= 0:
        raise ValueError("GitHub production-sbom artifact has invalid size_in_bytes")
    if size > MAX_SBOM_ARTIFACT_BYTES:
        raise ValueError("GitHub production-sbom artifact size_in_bytes is too large")
    return size


def _validate_artifact_id_metadata(
    artifact: dict[str, Any], *, download_artifact_id: int
) -> int:
    artifact_id = artifact.get("id")
    if type(artifact_id) is not int or artifact_id <= 0:
        raise ValueError("GitHub production-sbom artifact has invalid id")
    if artifact_id != download_artifact_id:
        raise ValueError("GitHub production-sbom artifact id does not match archive_download_url")
    return artifact_id


def _validate_run_attempt(run_attempt: Any) -> int:
    if type(run_attempt) is not int or run_attempt <= 0:
        raise ValueError("GitHub run_attempt must be a positive integer")
    return run_attempt


def _validate_run_id_metadata(run_id: Any, *, run_ref: RunRef) -> int:
    if type(run_id) is not int or run_id <= 0:
        raise ValueError("GitHub run id must be a positive integer")
    if str(run_id) != run_ref.run_id:
        raise ValueError("GitHub run id does not match supplied workflow run URL")
    return run_id


def _validate_run_repository_metadata(repository: Any, *, run_ref: RunRef) -> None:
    if not isinstance(repository, dict):
        raise ValueError("GitHub run repository metadata must be an object")
    full_name = repository.get("full_name")
    if not isinstance(full_name, str) or full_name != run_ref.slug:
        raise ValueError("GitHub run repository metadata does not match supplied workflow run URL")


def _validate_workflow_path(path: Any, *, head_branch: str) -> None:
    if path == EXPECTED_WORKFLOW_PATH:
        return
    if not isinstance(path, str) or not path.startswith(f"{EXPECTED_WORKFLOW_PATH}@"):
        raise ValueError(
            f"GitHub run workflow path must be {EXPECTED_WORKFLOW_PATH!r}, got {path!r}"
        )

    workflow_ref = path.removeprefix(f"{EXPECTED_WORKFLOW_PATH}@")
    if workflow_ref != head_branch or workflow_ref not in APPROVED_RELEASE_BRANCHES:
        raise ValueError(
            f"GitHub run workflow path must use release ref for {EXPECTED_WORKFLOW_PATH!r}, got {path!r}"
        )


def _verify_github_run_evidence(
    *,
    run_ref: RunRef,
    commit_sha: str,
    sbom_artifact: str,
    fetcher: GitHubApiFetcher,
    artifact_fetcher: GitHubBytesFetcher,
) -> GitHubRunEvidence:
    run = fetcher(run_ref.run_api_url)
    if str(run.get("url", "")).rstrip("/") != run_ref.run_api_url:
        raise ValueError("GitHub run API url does not match supplied workflow run URL")
    if str(run.get("html_url", "")).rstrip("/") != run_ref.html_url:
        raise ValueError("GitHub run html_url does not match supplied workflow run URL")
    _validate_run_id_metadata(run.get("id"), run_ref=run_ref)
    _validate_run_repository_metadata(run.get("repository"), run_ref=run_ref)
    if run.get("name") != EXPECTED_WORKFLOW_NAME:
        raise ValueError(
            f"GitHub run workflow must be {EXPECTED_WORKFLOW_NAME!r}, got {run.get('name')!r}"
        )
    if run.get("event") not in APPROVED_FINALIZER_EVENTS:
        allowed = ", ".join(sorted(APPROVED_FINALIZER_EVENTS))
        raise ValueError(f"GitHub run event must be one of {allowed}, got {run.get('event')!r}")
    if run.get("head_branch") not in APPROVED_RELEASE_BRANCHES:
        allowed = ", ".join(sorted(APPROVED_RELEASE_BRANCHES))
        raise ValueError(
            f"GitHub run head_branch must be one of {allowed}, got {run.get('head_branch')!r}"
        )
    _validate_workflow_path(run.get("path"), head_branch=str(run.get("head_branch")))
    if run.get("status") != "completed":
        raise ValueError(f"GitHub run must be completed, got {run.get('status')!r}")
    if run.get("conclusion") != "success":
        raise ValueError(f"GitHub run conclusion must be success, got {run.get('conclusion')!r}")
    if str(run.get("head_sha", "")).lower() != commit_sha.lower():
        raise ValueError("GitHub run head_sha does not match supplied commit SHA")
    run_attempt = _validate_run_attempt(run.get("run_attempt"))

    artifacts_url = _validate_run_artifacts_url(run.get("artifacts_url"), run_ref=run_ref)
    page = 1
    seen_artifacts = 0
    matching_artifacts: list[dict[str, Any]] = []
    while True:
        if page > MAX_GITHUB_ARTIFACT_PAGES:
            raise ValueError("GitHub artifact pagination exceeded the CRG3 page limit")
        artifacts = fetcher(_artifact_page_url(artifacts_url, page=page))
        if "artifacts" not in artifacts:
            raise ValueError("GitHub artifacts response is missing artifacts")
        artifact_entries = artifacts.get("artifacts")
        if not isinstance(artifact_entries, list):
            raise ValueError("GitHub artifacts response is malformed")
        total_count = artifacts.get("total_count")
        if type(total_count) is not int or total_count < 0:
            raise ValueError("GitHub artifacts response has invalid total_count")
        if len(artifact_entries) > MAX_GITHUB_ARTIFACTS_PER_PAGE:
            raise ValueError("GitHub artifacts response has too many entries for one page")
        seen_artifacts += len(artifact_entries)
        if seen_artifacts > total_count:
            raise ValueError("GitHub artifacts response entries exceed total_count")
        if not artifact_entries and seen_artifacts < total_count:
            raise ValueError("GitHub artifacts response ended before total_count")
        for artifact in artifact_entries:
            if not isinstance(artifact, dict):
                raise ValueError("GitHub artifacts response entries must be objects")
            if artifact.get("name") == sbom_artifact and artifact.get("expired") is False:
                matching_artifacts.append(artifact)

        if seen_artifacts >= total_count or not artifact_entries:
            break
        page += 1

    if not matching_artifacts:
        raise ValueError(f"GitHub run does not have a non-expired {sbom_artifact!r} artifact")
    if len(matching_artifacts) != 1:
        raise ValueError(f"GitHub run must have exactly one non-expired {sbom_artifact!r} artifact")
    artifact_size_in_bytes = _validate_artifact_size_metadata(matching_artifacts[0])
    download_url = str(matching_artifacts[0].get("archive_download_url") or "")
    if not download_url:
        raise ValueError("GitHub production-sbom artifact is missing archive_download_url")
    download_artifact_id = _validate_artifact_download_url(download_url, run_ref=run_ref)
    artifact_id = _validate_artifact_id_metadata(
        matching_artifacts[0],
        download_artifact_id=download_artifact_id,
    )
    artifact_download_size_in_bytes, artifact_sha256, sbom_sha256, sbom_component_count = (
        _verify_sbom_artifact_payload(
            download_url=download_url,
            fetch_bytes=artifact_fetcher,
        )
    )
    return GitHubRunEvidence(
        run_attempt=run_attempt,
        artifact_id=artifact_id,
        artifact_size_in_bytes=artifact_size_in_bytes,
        artifact_download_size_in_bytes=artifact_download_size_in_bytes,
        artifact_download_url=download_url,
        artifact_sha256=artifact_sha256,
        sbom_sha256=sbom_sha256,
        sbom_component_count=sbom_component_count,
    )


def _validate_evidence_json_path(evidence_json_path: Path, *, roadmap_path: Path) -> None:
    if evidence_json_path.suffix.lower() != ".json":
        raise ValueError("CRG3 evidence JSON path must use a .json suffix")
    if evidence_json_path.exists() and evidence_json_path.is_dir():
        raise ValueError("CRG3 evidence JSON path must be a file path, not a directory")
    parent = evidence_json_path.parent
    if parent != Path(".") and not parent.exists():
        raise ValueError("CRG3 evidence JSON parent directory does not exist")
    roadmap_dir = roadmap_path.resolve().parent
    resolved_evidence_path = evidence_json_path.resolve()
    if not resolved_evidence_path.is_relative_to(roadmap_dir):
        raise ValueError("CRG3 evidence JSON path must stay under the roadmap directory")
    relative_evidence_path = resolved_evidence_path.relative_to(roadmap_dir)
    if any(part.lower() == ".git" for part in relative_evidence_path.parts):
        raise ValueError("CRG3 evidence JSON path must not be inside .git")
    if any(part.startswith(".") for part in relative_evidence_path.parts[:-1]):
        raise ValueError("CRG3 evidence JSON path must not be inside hidden directories")
    if resolved_evidence_path.name.lower() == Path(EXPECTED_WORKFLOW_SBOM_PATH).name:
        raise ValueError("CRG3 evidence JSON path must not overwrite sbom.cdx.json")
    if resolved_evidence_path == _crg3_evidence_schema_path().resolve():
        raise ValueError("CRG3 evidence JSON path must not be the evidence schema path")
    if resolved_evidence_path == roadmap_path.resolve():
        raise ValueError("CRG3 evidence JSON path must not be the roadmap path")
    if resolved_evidence_path.name.lower() != "crg3-evidence.json":
        raise ValueError("CRG3 evidence JSON filename must be crg3-evidence.json")


def _build_evidence_record(
    *,
    verified_date: date,
    run_repo: str,
    repository_binding_source: str,
    run_ref: RunRef,
    run_url: str,
    github_evidence: GitHubRunEvidence,
    commit_sha: str,
    checked_head: str,
    sbom_artifact: str,
) -> dict[str, Any]:
    return {
        "checks": _expected_crg3_evidence_checks(),
        "commit_sha": commit_sha,
        "ledger_id": "CRG3",
        "local_head": checked_head,
        "local_working_tree": "clean",
        "repository": run_repo,
        "repository_binding_source": repository_binding_source,
        "run_attempt": github_evidence.run_attempt,
        "run_id": run_ref.run_id,
        "run_url": run_url,
        "schema": CRG3_EVIDENCE_SCHEMA,
        "sbom_artifact": sbom_artifact,
        "sbom_artifact_metadata": {
            "download_url": github_evidence.artifact_download_url,
            "id": github_evidence.artifact_id,
            "size_in_bytes": github_evidence.artifact_size_in_bytes,
            "download_size_in_bytes": github_evidence.artifact_download_size_in_bytes,
            "artifact_sha256": github_evidence.artifact_sha256,
            "sbom_sha256": github_evidence.sbom_sha256,
            "component_count": github_evidence.sbom_component_count,
        },
        "runtime_lock": _runtime_lock_metadata(),
        "schema_version": CRG3_EVIDENCE_SCHEMA_VERSION,
        "verified_date": verified_date.isoformat(),
        "workflow": {
            "job": EXPECTED_WORKFLOW_JOB,
            "live_execution_env": "0",
            "name": EXPECTED_WORKFLOW_NAME,
            "path": EXPECTED_WORKFLOW_PATH,
            "sbom_path": EXPECTED_WORKFLOW_SBOM_PATH,
            "wrapper_command": EXPECTED_WORKFLOW_WRAPPER_COMMAND,
        },
    }


def _expected_crg3_evidence_checks() -> list[str]:
    from tools.security.production_readiness import _checks as production_readiness_checks

    readiness_check_names = [
        check.name for check in production_readiness_checks(allow_open_ledger_ids=["CRG3"])
    ]
    return [*readiness_check_names, *CRG3_EVIDENCE_ONLY_CHECKS]


def _format_crg3_evidence_check_list(checks: list[str]) -> str:
    return ", ".join(f"`{check}`" for check in checks)


def _format_crg3_evidence_check_bullets(checks: list[str]) -> str:
    return "\n".join(f"- `{check}`" for check in checks)


def _crg3_evidence_schema_path() -> Path:
    schema_path = Path(CRG3_EVIDENCE_SCHEMA)
    if schema_path.is_absolute():
        return schema_path
    return REPO_ROOT / schema_path


def _load_crg3_evidence_schema() -> dict[str, Any]:
    schema_path = _crg3_evidence_schema_path()
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("CRG3 evidence schema file is missing") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("CRG3 evidence schema file is invalid JSON") from exc
    if not isinstance(schema, dict):
        raise ValueError("CRG3 evidence schema must be a JSON object")
    return schema


def _validate_schema_value(value: Any, schema: dict[str, Any], *, path: str) -> None:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            raise ValueError(f"{path} schema properties must be an object")
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(key, str) for key in required):
            raise ValueError(f"{path} schema required keys must be strings")
        missing = [key for key in required if key not in value]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{path} is missing required keys: {joined}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                joined = ", ".join(extra)
                raise ValueError(f"{path} has unexpected keys: {joined}")
        for key, nested_value in value.items():
            nested_schema = properties.get(key)
            if not isinstance(nested_schema, dict):
                raise ValueError(f"{path}.{key} schema is missing")
            _validate_schema_value(nested_value, nested_schema, path=f"{path}.{key}")
    elif expected_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array")
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            raise ValueError(f"{path} must contain at least {min_items} items")
        if schema.get("uniqueItems") is True:
            seen = set()
            duplicates = []
            for item in value:
                if item in seen and item not in duplicates:
                    duplicates.append(item)
                seen.add(item)
            if duplicates:
                joined = ", ".join(repr(item) for item in duplicates)
                raise ValueError(f"{path} must contain unique items; duplicates: {joined}")
        item_schema = schema.get("items")
        if item_schema is not None:
            if not isinstance(item_schema, dict):
                raise ValueError(f"{path} items schema must be an object")
            for index, item in enumerate(value):
                _validate_schema_value(item, item_schema, path=f"{path}[{index}]")
    elif expected_type == "integer":
        if type(value) is not int:
            raise ValueError(f"{path} must be an integer")
        minimum = schema.get("minimum")
        if isinstance(minimum, int) and value < minimum:
            raise ValueError(f"{path} must be at least {minimum}")
        maximum = schema.get("maximum")
        if isinstance(maximum, int) and value > maximum:
            raise ValueError(f"{path} must be at most {maximum}")
    elif expected_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            raise ValueError(f"{path} must have length at least {min_length}")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and not re.fullmatch(pattern, value):
            raise ValueError(f"{path} does not match required pattern")
    elif expected_type is not None:
        raise ValueError(f"{path} schema has unsupported type {expected_type!r}")

    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} must be {schema['const']!r}")


def _validate_crg3_evidence_checks(record: dict[str, Any]) -> None:
    checks = record.get("checks")
    if not isinstance(checks, list) or not all(isinstance(check, str) for check in checks):
        return

    expected = _expected_crg3_evidence_checks()
    duplicates = sorted({check for check in checks if checks.count(check) > 1})
    missing = [check for check in expected if check not in checks]
    unexpected = [check for check in checks if check not in expected]

    if duplicates or missing or unexpected:
        details = []
        if duplicates:
            details.append("duplicates: " + ", ".join(duplicates))
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ValueError(
            "CRG3 evidence checks must exactly match production readiness gates plus "
            f"workflow-only artifact proof ({'; '.join(details)})"
        )


def _validate_crg3_evidence_provenance(record: dict[str, Any]) -> None:
    runtime_lock = record.get("runtime_lock")
    if isinstance(runtime_lock, dict) and runtime_lock != _runtime_lock_metadata():
        raise ValueError(
            "CRG3 evidence runtime_lock does not match current requirements.lock metadata"
        )

    sbom_artifact_metadata = record.get("sbom_artifact_metadata")
    if not isinstance(sbom_artifact_metadata, dict):
        return
    expected_component_count = len(_expected_runtime_sbom_components())
    if sbom_artifact_metadata.get("component_count") != expected_component_count:
        raise ValueError(
            "CRG3 evidence sbom_artifact_metadata.component_count does not match "
            "requirements.lock component count"
        )


def _validate_crg3_evidence_identity(record: dict[str, Any]) -> None:
    run_ref = _run_ref_from_url(str(record["run_url"]))

    if record.get("repository") != run_ref.slug:
        raise ValueError("CRG3 evidence repository does not match run_url")
    if str(record.get("run_id")) != run_ref.run_id:
        raise ValueError("CRG3 evidence run_id does not match run_url")

    commit_sha = str(record.get("commit_sha"))
    local_head = str(record.get("local_head"))
    if commit_sha.lower() != local_head.lower():
        raise ValueError("CRG3 evidence local_head does not match commit_sha")

    sbom_artifact_metadata = record.get("sbom_artifact_metadata")
    if not isinstance(sbom_artifact_metadata, dict):
        return
    download_url = sbom_artifact_metadata.get("download_url")
    if isinstance(download_url, str):
        artifact_id = _validate_artifact_download_url(download_url, run_ref=run_ref)
        if sbom_artifact_metadata.get("id") != artifact_id:
            raise ValueError(
                "CRG3 evidence sbom_artifact_metadata.id does not match download_url"
            )


def _validate_crg3_verified_date_not_future(verified_date: date) -> None:
    if verified_date > datetime.now(UTC).date():
        raise ValueError("CRG3 verified date must not be in the future")


def _validate_crg3_evidence_temporal(record: dict[str, Any]) -> None:
    try:
        verified_date = date.fromisoformat(str(record["verified_date"]))
    except ValueError as exc:
        raise ValueError("CRG3 evidence verified_date must be a valid ISO calendar date") from exc
    try:
        _validate_crg3_verified_date_not_future(verified_date)
    except ValueError as exc:
        raise ValueError("CRG3 evidence verified_date must not be in the future") from exc


def _validate_crg3_evidence_record(record: dict[str, Any]) -> None:
    schema = _load_crg3_evidence_schema()
    try:
        _validate_schema_value(record, schema, path="CRG3 evidence")
        _validate_crg3_evidence_checks(record)
        _validate_crg3_evidence_provenance(record)
        _validate_crg3_evidence_identity(record)
        _validate_crg3_evidence_temporal(record)
    except ValueError as exc:
        raise ValueError(f"CRG3 evidence JSON does not match schema: {exc}") from exc


def _atomic_write_text(path: Path, text: str) -> None:
    target_dir = path.parent if path.parent != Path("") else Path(".")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=target_dir,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as temp_file:
            temp_file.write(text)
            temp_path = Path(temp_file.name)
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def finalize(
    *,
    run_url: str,
    commit_sha: str,
    sbom_artifact: str,
    verified_date: date | None = None,
    roadmap_path: Path = ROADMAP,
    allowed_repos: set[str] | None = None,
    github_api_fetcher: GitHubApiFetcher = _github_api_json,
    github_artifact_fetcher: GitHubBytesFetcher = _github_api_bytes,
    local_head: str | None = None,
    git_status: str | None = None,
    allow_reverify: bool = False,
    dry_run: bool = False,
    evidence_json_path: Path | None = None,
) -> None:
    repository_approval = _validate_run_url(run_url, allowed_repos=allowed_repos)
    run_repo = repository_approval.repo
    run_ref = _run_ref_from_url(run_url)
    _validate_sha(commit_sha)
    verified_date = verified_date or datetime.now(UTC).date()
    _validate_crg3_verified_date_not_future(verified_date)
    if sbom_artifact.strip() != REQUIRED_SBOM_ARTIFACT:
        raise ValueError(f"SBOM artifact must be {REQUIRED_SBOM_ARTIFACT!r}")
    if evidence_json_path is not None:
        _validate_evidence_json_path(evidence_json_path, roadmap_path=roadmap_path)
    checked_head = _validate_local_head(commit_sha, local_head=local_head)
    _validate_clean_checkout(git_status=git_status)
    _validate_local_workflow_live_disabled()
    github_evidence = _verify_github_run_evidence(
        run_ref=run_ref,
        commit_sha=commit_sha,
        sbom_artifact=sbom_artifact,
        fetcher=github_api_fetcher,
        artifact_fetcher=github_artifact_fetcher,
    )

    completed_checks = _expected_crg3_evidence_checks()
    completed_checks_text = _format_crg3_evidence_check_list(completed_checks)
    completed_checks_bullets = _format_crg3_evidence_check_bullets(completed_checks)
    runtime_lock = _runtime_lock_metadata()
    text = roadmap_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    for index, line in enumerate(lines):
        if line.startswith(CRG3_PREFIX):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) != 7:
                raise ValueError("CRG3 ledger row is malformed")
            if cells[3].startswith("Verified") and not allow_reverify:
                raise ValueError(
                    "CRG3 is already verified; pass --allow-reverify to replace existing evidence"
                )
            cells[3] = f"Verified {verified_date.isoformat()}"
            cells[6] = (
                "Production Readiness workflow passed with exact completed checks recorded in "
                f"CRG3 evidence JSON: {completed_checks_text}; artifact ID/download URL binding, "
                "canonical artifact URL validation, and cross-host redirect authorization stripping; "
                f"repo: `{run_repo}`; repo binding: `{repository_approval.source}`; "
                f"run: {run_url}; commit: `{commit_sha}`; "
                f"run id: `{run_ref.run_id}`; run attempt: `{github_evidence.run_attempt}`; "
                f"artifact: `{sbom_artifact}`."
            )
            lines[index] = "| " + " | ".join(cells) + " |"
            break
    else:
        raise ValueError("CRG3 ledger row not found")

    report = f"""

### {verified_date.isoformat()} Pass 6 CRG3 Workflow Evidence Finalization

Ledger IDs changed:
- `CRG3` verified from successful Docker-capable GitHub Actions evidence.

Files changed:
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Production readiness workflow ran on a clean GitHub Actions runner.
- Finalizer-generated roadmap and report proof use the same exact completed check names as CRG3 evidence JSON.
- Downloaded `production-sbom` artifact contains exactly one valid CycloneDX `sbom.cdx.json` whose unique `type`/`name`/`version`/`purl` components match `requirements.lock` exactly.
- CRG3 evidence records the `requirements.lock` SHA-256 digest used for SBOM matching.
- `production-sbom` artifact metadata proves explicit `expired: false`, bounded `size_in_bytes`, and artifact `id` matching `archive_download_url`.
- Artifact download URL is repository-bound and canonical with no query string or fragment.
- GitHub API/artifact response reads are bounded, and cross-host artifact download redirects strip `Authorization`.
- Local production readiness workflow keeps manual dispatch, pull-request probes, release-branch push triggers, read-only permissions, same-ref concurrency without cancellation, Ubuntu runner, and 45-minute timeout.
- Local production readiness workflow pins job-level `BOT_V2_ALLOW_LIVE_EXECUTION=0`.
- Local production readiness workflow uses pinned checkout/setup actions, disables persisted checkout credentials, installs hash-locked CI tools before hash-locked runtime dependencies, and runs the wrapper before SBOM upload.
- Local production readiness workflow uploads `build/security/sbom.cdx.json` as exactly one `production-sbom` artifact with missing-file failure enabled.

Completed checks:
{completed_checks_bullets}

Verification evidence:
- Repository: `{run_repo}`
- Repository binding source: `{repository_approval.source}`
- GitHub API repository.full_name: `{run_repo}`
- Workflow run: {run_url}
- Workflow run ID: `{run_ref.run_id}`
- Workflow run attempt: `{github_evidence.run_attempt}`
- Commit SHA: `{commit_sha}`
- Local checkout HEAD: `{checked_head}`
- Local working tree: clean
- SBOM artifact: `{sbom_artifact}`
- SBOM artifact ID: `{github_evidence.artifact_id}`
- SBOM artifact size: `{github_evidence.artifact_size_in_bytes}`
- SBOM artifact downloaded size: `{github_evidence.artifact_download_size_in_bytes}`
- SBOM artifact download URL: `{github_evidence.artifact_download_url}`
- SBOM artifact SHA-256: `{github_evidence.artifact_sha256}`
- SBOM JSON SHA-256: `{github_evidence.sbom_sha256}`
- SBOM component count: `{github_evidence.sbom_component_count}`
- Runtime lockfile: `{runtime_lock["path"]}`
- Runtime lockfile SHA-256: `{runtime_lock["sha256"]}`

Rollback plan:
- If this evidence is later invalidated, set `CRG3` back to `In progress`, keep `BOT_V2_ALLOW_LIVE_EXECUTION=0`, and rerun `python tools/security/production_readiness.py --profile ci` on a fixed commit.

Residual risk:
- Strict live readiness must still pass from a clean Docker-capable checkout before live enablement.

Live-block status:
- Live remains blocked until `python tools/security/production_readiness.py --profile live` passes after this ledger update.
"""
    updated_text = "\n".join(lines).rstrip() + report + "\n"
    validation_errors = validate_p0_rows(
        parse_ledger_text(updated_text, source=str(roadmap_path))
    )
    if validation_errors:
        joined_errors = "; ".join(validation_errors)
        raise ValueError(f"updated roadmap P0 ledger validation failed: {joined_errors}")

    evidence_json = None
    if evidence_json_path is not None:
        evidence_record = _build_evidence_record(
            verified_date=verified_date,
            run_repo=run_repo,
            repository_binding_source=repository_approval.source,
            run_ref=run_ref,
            run_url=run_url,
            github_evidence=github_evidence,
            commit_sha=commit_sha,
            checked_head=checked_head,
            sbom_artifact=sbom_artifact,
        )
        _validate_crg3_evidence_record(evidence_record)
        evidence_json = (
            json.dumps(
                evidence_record,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    if dry_run:
        return

    _atomic_write_text(roadmap_path, updated_text)
    if evidence_json_path is not None and evidence_json is not None:
        _atomic_write_text(evidence_json_path, evidence_json)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finalize CRG3 from successful workflow evidence.")
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--sbom-artifact", default="production-sbom")
    parser.add_argument("--date", help="Verification date in YYYY-MM-DD format")
    parser.add_argument("--roadmap", default=str(ROADMAP))
    parser.add_argument(
        "--allow-reverify",
        action="store_true",
        help="Replace an existing CRG3 Verified row with new workflow evidence.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify CRG3 evidence and the would-be roadmap update without writing.",
    )
    parser.add_argument(
        "--evidence-json",
        help="Optional path for a machine-readable CRG3 evidence JSON file.",
    )
    args = parser.parse_args(argv)

    verified_date = date.fromisoformat(args.date) if args.date else None
    finalize(
        run_url=args.run_url,
        commit_sha=args.commit_sha,
        sbom_artifact=args.sbom_artifact,
        verified_date=verified_date,
        roadmap_path=Path(args.roadmap),
        allow_reverify=args.allow_reverify,
        dry_run=args.dry_run,
        evidence_json_path=Path(args.evidence_json) if args.evidence_json else None,
    )
    if args.dry_run:
        print("CRG3 finalizer dry run passed; roadmap not modified.")
    else:
        print("CRG3 finalized from workflow evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
