@echo off
cd /d "e:\m\DP FAIRNESS"

echo Starting progress monitor...
echo Will print progress every 30 seconds
echo Press Ctrl+C to stop
echo.

:loop
echo ================================================================================
date /t
time /t
echo ================================================================================
python scripts/print_progress.py
echo.
timeout /t 30 /nobreak >nul
goto loop
