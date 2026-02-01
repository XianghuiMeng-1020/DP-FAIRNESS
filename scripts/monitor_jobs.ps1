# Monitor background jobs and display output
# Usage: .\scripts\monitor_jobs.ps1

Set-Location "e:\m\DP FAIRNESS"

Write-Host "Monitoring background jobs..."
Write-Host "Press Ctrl+C to stop monitoring"
Write-Host ""

while ($true) {
    Clear-Host
    Write-Host "=" * 80
    Write-Host "Job Status: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "=" * 80
    Get-Job | Format-Table -AutoSize
    
    Write-Host ""
    Write-Host "Recent log output (last 10 lines):"
    Write-Host "-" * 80
    if (Test-Path "outputs/logs/full_rerun.log") {
        Get-Content "outputs/logs/full_rerun.log" -Tail 10
    } else {
        Write-Host "Log file not created yet..."
    }
    
    Write-Host ""
    Write-Host "Progress Update:"
    Write-Host "-" * 80
    python scripts/print_progress.py
    
    Write-Host ""
    Write-Host "Next update in 30 seconds... (Press Ctrl+C to stop)"
    Start-Sleep -Seconds 30
}
