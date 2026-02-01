# PowerShell script to start full rerun with logging
# Usage: .\scripts\start_full_rerun_fixed.ps1

$PlanPath = "outputs/reports/experiment_plan_fast.json"
$LogPath = "outputs/logs/full_rerun.log"

Write-Host "Starting full rerun..."
Write-Host "Plan: $PlanPath"
Write-Host "Log: $LogPath"
Write-Host "Fail-fast: ENABLED"
Write-Host "Force-rerun: ENABLED"
Write-Host ""

# Create log directory
New-Item -ItemType Directory -Force -Path "outputs/logs" | Out-Null

# Change to project directory
Set-Location "e:\m\DP FAIRNESS"

# Start run with unbuffered output and tee to log
python -u src/run_all.py `
    --only-plan $PlanPath `
    --fail-fast `
    --force-rerun `
    2>&1 | Tee-Object -FilePath $LogPath

$ExitCode = $LASTEXITCODE

if ($ExitCode -eq 0) {
    Write-Host ""
    Write-Host "=========================================="
    Write-Host "Full rerun completed successfully!"
    Write-Host "=========================================="
} else {
    Write-Host ""
    Write-Host "=========================================="
    Write-Host "Full rerun stopped due to failure!"
    Write-Host "Exit code: $ExitCode"
    Write-Host "=========================================="
}

exit $ExitCode
