param(
    [string]$Config = "configs/resnet18.yaml",
    #[string]$Config = "onfigs/efficientnet_b0.yaml "

    [string]$PythonExe = "python",

    [switch]$SkipSplit,
    [switch]$SkipEval,
    [switch]$SkipViz,
    [switch]$SkipGradCam
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot

$config = $Config
$dataRoot = Join-Path $repoRoot "data\raw\PlantVillage"
$splitCsv = Join-Path $repoRoot "data\splits\split.csv"
$splitOutDir = Join-Path $repoRoot "data\splits"

Write-Host "=============================================="
Write-Host "Using config: $config"
Write-Host "Python: $PythonExe"
Write-Host "=============================================="

if (-not (Test-Path $dataRoot)) {
    throw "Raw dataset not found at $dataRoot. Download PlantVillage and place the class folders there before training."
}

if (-not $SkipSplit -and -not (Test-Path $splitCsv)) {
    Write-Host ""
    Write-Host "== [0/4] Building split.csv =="
    & $PythonExe -m src.data.split --data_root $dataRoot --out_dir $splitOutDir
}

Write-Host ""
Write-Host "== [1/4] Training =="
& $PythonExe -m src.train --config $config

if (-not $SkipEval) {
    Write-Host ""
    Write-Host "== [2/4] Evaluating on test split =="
    & $PythonExe -m src.eval --config $config
}

if (-not $SkipViz) {
    Write-Host ""
    Write-Host "== [3/4] Generating figures =="
    & $PythonExe -m src.visualize --normalize_cm --make_gallery --config $config
}

if (-not $SkipGradCam) {
    Write-Host ""
    Write-Host "== [4/4] Generating Grad-CAM overlays =="
    & $PythonExe -m src.explain.gradcam --config $config
}

Write-Host ""
Write-Host "Done."