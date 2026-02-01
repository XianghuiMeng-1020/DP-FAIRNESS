# Combined script: Run experiments and monitor progress
# Usage: .\scripts\run_with_monitor.ps1

Set-Location "e:\m\DP FAIRNESS"

$PlanPath = "outputs/reports/experiment_plan_fast.json"
$LogPath = "outputs/logs/full_rerun.log"

Write-Host "=" * 80
Write-Host "Starting TASK 3: Full Rerun"
Write-Host "=" * 80
Write-Host "Plan: $PlanPath"
Write-Host "Log: $LogPath"
Write-Host "Fail-fast: ENABLED"
Write-Host "Force-rerun: ENABLED"
Write-Host ""

# Create log directory
New-Item -ItemType Directory -Force -Path "outputs/logs" | Out-Null

# Start the main run process in background
$process = Start-Process python -ArgumentList "-u", "src/run_all.py", "--only-plan", $PlanPath, "--fail-fast", "--force-rerun" -PassThru -NoNewWindow -RedirectStandardOutput $LogPath -RedirectStandardError "$LogPath.error"

Write-Host "Main process started with PID: $($process.Id)"
Write-Host ""

# Monitor progress every 30 seconds
Write-Host "Starting progress monitor (updates every 30 seconds)..."
Write-Host "Press Ctrl+C to stop monitoring (process will continue running)"
Write-Host ""

try {
    while (-not $process.HasExited) {
        Clear-Host
        Write-Host "=" * 80
        Write-Host "Progress Update: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        Write-Host "Process Status: $($process.ProcessName) (PID: $($process.Id)) - $(if ($process.HasExited) { 'Exited' } else { 'Running' })"
        Write-Host "=" * 80
        Write-Host ""
        
        # Show progress
        python scripts/print_progress.py
        
        Write-Host ""
        Write-Host "Recent log (last 15 lines):"
        Write-Host "-" * 80
        if (Test-Path $LogPath) {
            Get-Content $LogPath -Tail 15 -ErrorAction SilentlyContinue
        } else {
            Write-Host "Log file not created yet..."
        }
        
        Write-Host ""
        Write-Host "Next update in 30 seconds... (Press Ctrl+C to stop monitoring)"
        
        Start-Sleep -Seconds 30
    }
    
    # Process finished
    Write-Host ""
    Write-Host "=" * 80
    Write-Host "Process completed with exit code: $($process.ExitCode)"
    Write-Host "=" * 80
    
    if ($process.ExitCode -eq 0) {
        Write-Host "SUCCESS: All runs completed!"
    } else {
        Write-Host "FAILED: Process exited with error code $($process.ExitCode)"
        Write-Host ""
        Write-Host "Last 50 lines of log:"
        Write-Host "-" * 80
        if (Test-Path $LogPath) {
            Get-Content $LogPath -Tail 50
        }
    }
    
} catch {
    Write-Host ""
    Write-Host "Monitoring stopped. Process is still running (PID: $($process.Id))"
    Write-Host "Check log file: $LogPath"
}
