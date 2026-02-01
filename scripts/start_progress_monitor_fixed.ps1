# PowerShell script for progress monitoring
# Usage: .\scripts\start_progress_monitor_fixed.ps1

Write-Host "Starting progress monitor..."
Write-Host "Will print progress every 30 seconds"
Write-Host "Press Ctrl+C to stop"
Write-Host ""

# Change to project directory
Set-Location "e:\m\DP FAIRNESS"

while ($true) {
    Get-Date
    python scripts/print_progress.py
    Write-Host ""
    Start-Sleep -Seconds 30
}
