@echo off
cd /d "e:\m\DP FAIRNESS"

echo ================================================================================
echo Starting TASK 3: Full Rerun
echo ================================================================================
echo Plan: outputs/reports/experiment_plan_fast.json
echo Log: outputs/logs/full_rerun.log
echo Fail-fast: ENABLED
echo Force-rerun: ENABLED
echo.

if not exist "outputs\logs" mkdir "outputs\logs"

python -u src/run_all.py --only-plan outputs/reports/experiment_plan_fast.json --fail-fast --force-rerun > outputs/logs/full_rerun.log 2>&1

set EXITCODE=%ERRORLEVEL%

echo.
echo ================================================================================
if %EXITCODE% EQU 0 (
    echo Full rerun completed successfully!
) else (
    echo Full rerun stopped due to failure!
    echo Exit code: %EXITCODE%
)
echo ================================================================================

exit /b %EXITCODE%
