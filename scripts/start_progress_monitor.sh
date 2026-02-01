#!/bin/bash
# Progress monitor - prints progress every 30 seconds

echo "Starting progress monitor..."
echo "Will print progress every 30 seconds"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    date
    python scripts/print_progress.py
    echo ""
    sleep 30
done
