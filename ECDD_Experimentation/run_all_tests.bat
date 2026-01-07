@echo off
REM Windows wrapper for run_all_tests.py

echo ==========================================
echo 🧪 ECDD Infrastructure Test Suite
echo ==========================================
echo.

REM Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.7+.
    exit /b 1
)

echo Using Python: 
python --version
echo.

REM Run tests
python run_all_tests.py --verbose --output test_report.json

REM Check result
if exist test_report.json (
    echo.
    echo 📊 Full report saved to: test_report.json
)

echo.
echo ✨ Test run complete!
pause
