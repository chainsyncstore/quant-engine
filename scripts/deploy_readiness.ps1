param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,
    [ValidateSet("preflight", "deploy", "rollback")]
    [string]$Mode = "preflight",
    [switch]$Execute,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

$args = @(
    "$Root/tools/deploy_readiness.py",
    "--manifest", $Manifest,
    "--mode", $Mode
)

if ($Execute) {
    $args += "--execute"
}

if ($OutputDir) {
    $args += @("--output-dir", $OutputDir)
}

& $Python @args
