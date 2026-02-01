#!/bin/bash
# Start full rerun with fail-fast and logging

PLAN_PATH="outputs/reports/experiment_plan_fast.json"
LOG_PATH="outputs/logs/full_rerun.log"

echo "Starting full rerun..."
echo "Plan: $PLAN_PATH"
echo "Log: $LOG_PATH"
echo "Fail-fast: ENABLED"
echo ""

# Create log directory
mkdir -p outputs/logs

# Start run with unbuffered output and tee to log
python -u src/run_all.py \
    --only-plan "$PLAN_PATH" \
    --fail-fast \
    --force-rerun \
    2>&1 | tee "$LOG_PATH"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Full rerun completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Full rerun stopped due to failure!"
    echo "Exit code: $EXIT_CODE"
    echo "=========================================="
fi

exit $EXIT_CODE
