# 🧪 Master Test Runner

Comprehensive test suite for all ECDD Experimentation infrastructure.

---

## Quick Start

### Windows
```bash
run_all_tests.bat
```

### Linux/Mac
```bash
chmod +x run_all_tests.sh
./run_all_tests.sh
```

### Python Direct
```bash
python run_all_tests.py --verbose --output test_report.json
```

---

## What It Tests

### ✅ Phase 6 Experiments (4 tests)
- **E6.1**: Source-based split stress test (mock)
- **E6.2**: Time-based split drift probe (mock)
- **E6.3**: Transform suite conclusive test (real data)
- **E6.4**: Out-of-scope separation test (real data)

### ✅ Phase 5 Experiments (5 tests)
- **E5.1**: Float vs TFLite probability parity (mock)
- **E5.2**: Patch-logit map parity (mock)
- **E5.3**: Pooled logit parity (mock)
- **E5.4**: Post-quant calibration gate (mock)
- **E5.5**: Delegate and threading invariance (mock)

### ✅ Golden Hash System (1 test)
- **S0-S4**: Preprocessing stage hash generation

### ✅ Calibration Utilities (1 test)
- **Temperature Scaling**: Fit and verify calibration

### ✅ Pipeline Components (3 tests)
- **Face Detection**: Consistency test
- **Preprocessing**: Shape and dtype validation
- **Determinism**: Reproducibility check

### ✅ Export Infrastructure (1 test)
- **Parity Validation**: Model comparison utilities

**Total: 15 tests**

---

## Output

### Console Output
```
================================================================================
🧪 ECDD INFRASTRUCTURE TEST SUITE
================================================================================

=== Phase 6: Evaluation Battery ===
ℹ️ Running E6.1: Source-based split stress test
✅ PASS: E6.1 (123.4ms)
ℹ️ Running E6.2: Time-based split drift probe
✅ PASS: E6.2 (98.7ms)
...

================================================================================
TEST EXECUTION SUMMARY
================================================================================

Total Tests: 15
Passed: 15 ✅
Failed: 0 ❌
Success Rate: 100.0%
Total Time: 1234.5ms

By Category:
  Phase6: 4/4 passed
  Phase5: 5/5 passed
  GoldenHash: 1/1 passed
  Calibration: 1/1 passed
  Pipeline: 3/3 passed
  Export: 1/1 passed

================================================================================
```

### JSON Report
```json
{
  "timestamp": "2026-01-07T20:30:00",
  "summary": {
    "total": 15,
    "passed": 15,
    "failed": 0,
    "success_rate": 1.0,
    "total_time_ms": 1234.5
  },
  "categories": {
    "Phase6": {"passed": 4, "failed": 0},
    "Phase5": {"passed": 5, "failed": 0},
    ...
  },
  "results": [
    {
      "test_id": "E6.1",
      "name": "Source-based split stress test",
      "category": "Phase6",
      "passed": true,
      "duration_ms": 123.4,
      "details": {...}
    },
    ...
  ]
}
```

---

## Command Line Options

```bash
python run_all_tests.py [OPTIONS]

Options:
  --output, -o PATH    Save JSON report to file
  --verbose, -v        Show detailed test output
  --help, -h           Show help message
```

### Examples

**Basic run:**
```bash
python run_all_tests.py
```

**Verbose with report:**
```bash
python run_all_tests.py --verbose --output results.json
```

**Silent mode (summary only):**
```bash
python run_all_tests.py --output results.json
```

---

## Exit Codes

- **0**: All tests passed ✅
- **1**: One or more tests failed ❌

Use in CI/CD:
```bash
#!/bin/bash
python run_all_tests.py --output ci_report.json
if [ $? -eq 0 ]; then
    echo "✅ All tests passed"
else
    echo "❌ Tests failed"
    exit 1
fi
```

---

## Test Categories

### Real Data Tests (Available Now)
These use actual images from `ECDD_Experiment_Data/`:
- E6.3: Transform suite (18 real images)
- E6.4: OOD separation (20 OOD images)
- Golden hash generation (any images)

### Mock Tests (Infrastructure Validation)
These use synthetic data to test infrastructure:
- E6.1, E6.2: Split tests with generated metadata
- E5.1-E5.5: Parity tests with generated outputs
- All calibration and pipeline tests

---

## Expected Results

### First Run (No Models)
```
Total Tests: 15
Passed: 13-15 ✅
Failed: 0-2 ❌
```

**Potential failures:**
- Golden hash test if `ECDD_Experiment_Data/real/` not found (skip expected)
- E6.4 if OOD directory not found (skip expected)

### With Trained Models
Once models are integrated:
- Phase 4 tests (6 more)
- Phase 5 real mode (5 tests upgraded)
- CI gates (6 more)

**Total possible: 32+ tests**

---

## Troubleshooting

### Import Errors
**Problem**: `ModuleNotFoundError`

**Solution**: Run from `ECDD_Experimentation/` directory:
```bash
cd Team-Converge/ECDD_Experimentation
python run_all_tests.py
```

### Missing Dependencies
**Problem**: `ImportError: No module named 'numpy'`

**Solution**: Install dependencies:
```bash
pip install numpy scipy pillow
```

### Missing Test Data
**Problem**: "Golden directory not found"

**Solution**: This is expected if data isn't downloaded. Tests will skip gracefully.

### Mock Warnings
**Problem**: "Using mock predictions"

**Solution**: This is expected for E5.x and some E6.x tests. Not a failure.

---

## Integration with CI/CD

### GitHub Actions
```yaml
name: ECDD Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install numpy scipy pillow
      - name: Run tests
        run: |
          cd Team-Converge/ECDD_Experimentation
          python run_all_tests.py --verbose --output test_report.json
      - name: Upload report
        uses: actions/upload-artifact@v2
        if: always()
        with:
          name: test-report
          path: Team-Converge/ECDD_Experimentation/test_report.json
```

### GitLab CI
```yaml
test:
  stage: test
  script:
    - pip install numpy scipy pillow
    - cd Team-Converge/ECDD_Experimentation
    - python run_all_tests.py --verbose --output test_report.json
  artifacts:
    reports:
      junit: test_report.json
    when: always
```

---

## Extending the Test Suite

### Adding New Tests

```python
# In run_all_tests.py

def run_custom_tests(self):
    """Run custom tests."""
    self.log("\n=== Custom Tests ===", "SUMMARY")
    
    def test_my_feature():
        # Your test logic
        result = my_function()
        return {
            'passed': result.success,
            'details': result.data
        }
    
    self.run_test("Custom.1", "My feature test", "Custom", test_my_feature)

# Add to run_all():
def run_all(self):
    ...
    self.run_custom_tests()  # Add this
```

---

## Performance Baseline

Expected execution times (no models):
- Phase 6 tests: ~500ms
- Phase 5 tests: ~200ms
- Golden hash (5 images): ~100ms
- Calibration: ~50ms
- Pipeline: ~50ms
- Export: ~20ms

**Total: ~1-2 seconds**

---

## Summary

The master test runner provides:
- ✅ Comprehensive infrastructure validation
- ✅ Automated test execution
- ✅ JSON reports for CI/CD
- ✅ Detailed error reporting
- ✅ Mock tests for development
- ✅ Real data tests when available

Run it regularly to ensure infrastructure integrity! 🚀
