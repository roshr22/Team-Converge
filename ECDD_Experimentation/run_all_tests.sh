#!/bin/bash
# Bash wrapper for run_all_tests.py

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "🧪 ECDD Infrastructure Test Suite"
echo "=========================================="
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python not found. Please install Python 3.7+."
        exit 1
    fi
    PYTHON=python3
else
    PYTHON=python
fi

echo "Using Python: $PYTHON"
echo ""

# Run tests
$PYTHON run_all_tests.py --verbose --output test_report.json

# Show report location
if [ -f test_report.json ]; then
    echo ""
    echo "📊 Full report saved to: test_report.json"
fi

echo ""
echo "✨ Test run complete!"
