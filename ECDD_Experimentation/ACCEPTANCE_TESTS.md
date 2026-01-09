# ECDD Acceptance Tests

> "Done" checks with exact commands. All tests must pass before declaring a step complete.

---

## Quick Smoke Tests (< 2 min)

### 1. Master Test Runner
```bash
# Windows
python run_all_tests.py --verbose --output test_report.json

# Linux/Mac
./run_all_tests.sh
```

**Expected**: All tests pass, `test_report.json` created with `"overall_pass": true`.

---

### 2. Python Import Smoke Test
```bash
python -c "from ecdd_core.pipeline import decode_image_bytes, resize_rgb_uint8, normalize_rgb_uint8; print('Pipeline imports OK')"
python -c "from ecdd_core.calibration import fit_temperature, expected_calibration_error; print('Calibration imports OK')"
```

**Expected**: Both print success messages without errors.

---

### 3. Preprocessing Determinism Check
```bash
python -c "
import numpy as np
from ecdd_core.pipeline import PreprocessConfig, resize_rgb_uint8, sha256_ndarray

cfg = PreprocessConfig()
np.random.seed(42)
img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

h1 = sha256_ndarray(resize_rgb_uint8(img, cfg))
h2 = sha256_ndarray(resize_rgb_uint8(img, cfg))
assert h1 == h2, 'Preprocessing not deterministic!'
print(f'Determinism OK, hash={h1[:16]}...')
"
```

**Expected**: Prints hash confirmation without assertion error.

---

## Gate Tests (CI-level)

### G1: Pixel Equivalence Gate
```bash
python -c "from ci.gates.g1_pixel_equivalence import test_pixel_equivalence; test_pixel_equivalence()"
```

### G2: Guardrail Gate
```bash
python -c "from ci.gates.g2_guardrail import test_guardrail; test_guardrail()"
```

### G3: Model Semantics Gate
```bash
python -c "from ci.gates.g3_model_semantics import test_model_semantics; test_model_semantics()"
```

### G4: Calibration Gate
```bash
python -c "from ci.gates.g4_calibration import test_calibration; test_calibration()"
```

### G5: Quantization Parity Gate
```bash
python -c "from ci.gates.g5_quantization import test_quantization; test_quantization()"
```

### G6: Release Gate
```bash
python -c "from ci.gates.g6_release import test_release; test_release()"
```

---

## Preprocessing Equivalence Test (New — Step 4)

> Verifies training and inference preprocessing paths produce identical outputs.

```bash
python evaluation/preprocessing_equivalence_test.py --image-set test_images/ --tolerance 1e-5 --report results/preprocessing_equivalence_report.json
```

**Expected Outputs**:
- `results/preprocessing_equivalence_report.json` with `"all_passed": true`
- Per-image comparison: channel order, dtype/range, resize hash, normalization hash
- Any mismatch logs with exact pixel diff locations

---

## Evaluation Diagnostics (New — Step 2)

### Run Full Evaluation Suite
```bash
python evaluation/evaluate_model.py \
  --index-file <path-to-dataset-index.json> \
  --model-path <path-to-model> \
  --output-dir results/ \
  --threshold 0.5
```

**Expected Outputs**:
- `results/<timestamp>_metrics.json` — AUC, AP, F1, TPR@FPR, FPR@TPR per group
- `results/<timestamp>_metrics.csv` — Same in tabular format
- `results/<timestamp>_confusion.png` — Confusion matrix heatmap
- `results/<timestamp>_misclassified_confidence.png` — FP/FN confidence histogram

---

## Calibration Test (New — Step 3)

### Fit Temperature Scaling
```bash
python evaluation/fit_temp.py \
  --logits-file calibration_logits.npy \
  --labels-file calibration_labels.npy \
  --output-dir results/calibration/
```

**Expected Outputs**:
- `results/calibration/temperature_params.json` — `{"temperature": <float>}`
- `results/calibration/reliability_curve.png` — Before/after calibration
- `results/calibration/calibration_report.json` — ECE before/after, NLL before/after

---

## Adding New Tests

When adding a new test:
1. Add the exact command and expected output above
2. If it's a quick smoke test (< 30 sec), add to "Quick Smoke Tests"
3. If it requires model/data, add to appropriate section
4. Update `run_all_tests.py` to include the new test
5. Log the addition in `RUNLOG.md`
