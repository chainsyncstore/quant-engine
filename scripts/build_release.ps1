param(
    [string]$Image = "quant-bot:wp02",
    [switch]$Dev
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Manifest = Join-Path $Root ".build/wp02-manifest.json"
$Mode = if ($Dev) { @() } else { @("--release") }

& $Python "$Root/tools/build_manifest.py" --output $Manifest @Mode
$metadata = Get-Content -Raw $Manifest | ConvertFrom-Json

docker build --target release `
  --build-arg "VCS_REF=$($metadata.git_sha)" `
  --build-arg "SOURCE_MANIFEST_SHA256=$($metadata.source_manifest_sha256)" `
  --build-arg "LOCK_SHA256=$($metadata.lock_sha256)" `
  --build-arg "CI_RUN_ID=$($env:CI_RUN_ID)" `
  --build-arg "BUILD_DATE=$($env:BUILD_DATE)" `
  --tag $Image $Root

docker run --rm --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m `
  --cap-drop ALL --security-opt no-new-privileges:true $Image `
  python /app/tools/image_smoke.py --require-release-marker

docker run --rm --read-only --tmpfs /tmp:rw,noexec,nosuid,size=16m `
  --cap-drop ALL --security-opt no-new-privileges:true $Image `
  sh -ec 'test "$(id -u)" = 10001; test -f /app/.wp02-tests-passed; test ! -d /app/tests; test ! -d /opt/build-tools; test ! -e /app/requirements/build.lock; test ! -e /build/requirements/build.lock; ! command -v uv; ! command -v pytest; ! command -v ruff'

$digest = docker image inspect $Image --format '{{index .RepoDigests 0}}'
if (-not $digest) {
    Write-Warning "Local Docker did not assign a registry digest; push once and deploy the resulting @sha256 reference."
}
