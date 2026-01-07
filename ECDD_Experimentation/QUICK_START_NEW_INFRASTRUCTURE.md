# 🚀 Quick Start: New Infrastructure

This guide shows how to use the newly implemented Phase 4-6 experiments, CI gates, and golden hash system.

---

## 📦 What Was Built

### 1. Phase 4: Calibration & Thresholds (6 experiments)
- Validates calibration improves reliability (temperature/Platt scaling)
- Selects operating point thresholds at fixed error budget (e.g., FPR ≤ 5%)
- Designs abstain bands for selective classification
- Compares conditional threshold policies

### 2. Phase 5: Quantization & Parity (5 experiments)
- Tests float vs TFLite probability parity
- Validates patch-logit map parity
- Checks pooled logit parity
- Ensures post-quantization calibration improves reliability
- Tests delegate and threading invariance

### 3. Phase 6: Evaluation Battery (4 experiments)
- Source-based split stress test (cross-generator generalization)
- Time-based split drift probe (temporal distribution shift)
- Transform suite conclusive test (robustness to compression/blur)
- Out-of-scope separation test (OOD abstention)

### 4. CI Gates (6 gates)
- **G1**: Pixel equivalence (preprocessing determinism)
- **G2**: Guardrails (face detection, OOD handling)
- **G3**: Model semantics (sensible predictions)
- **G4**: Calibration (reliability improvement)
- **G5**: Quantization parity (float/TFLite agreement)
- **G6**: Release gate (final pre-deployment checklist)

### 5. Export Infrastructure
- TFLite converter with quantization support
- Model parity validation utilities
- Representative dataset handling

### 6. Golden Hash System
- Generate hashes for S0-S8 pipeline stages
- Verify reproducibility across environments

---

## 🏃 Running Phase 4-6 Experiments

### Phase 4: Calibration

**Requirements**: Calibration logits JSON file with format:
```json
[
  {"id": "sample_001", "logit": 1.23, "label": 1},
  {"id": "sample_002", "logit": -0.45, "label": 0},
  ...
]
```

**Run all Phase 4 experiments:**
```bash
cd PhaseWise_Experiments_ECDD
python phase4_experiments.py /path/to/calibration_logits.json
```

**Results written to**: `phase4_results/E4.1.json` through `E4.6.json`

**Individual experiments:**
```python
from phase4_experiments import e4_2_temperature_scaling_fit_and_verify
from pathlib import Path

result = e4_2_temperature_scaling_fit_and_verify(
    calibration_json=Path("calibration_logits.json")
)
print(result.passed)
print(result.details)
```

---

### Phase 5: Quantization Parity

**Current status**: Uses **mock data** until models are trained.

**Run all Phase 5 experiments:**
```bash
cd PhaseWise_Experiments_ECDD
python phase5_experiments.py
```

**With real models** (when available):
```python
from phase5_experiments import e5_1_float_vs_tflite_probability_parity
from pathlib import Path

result = e5_1_float_vs_tflite_probability_parity(
    epsilon_prob=0.05,
    float_model_path=Path("model.onnx"),
    tflite_model_path=Path("model.tflite")
)
```

**Results written to**: `phase5_results/E5.1.json` through `E5.5.json`

---

### Phase 6: Evaluation Battery

**Run all Phase 6 experiments:**
```bash
cd PhaseWise_Experiments_ECDD
python phase6_experiments.py
```

**E6.3 (Transform suite)** - Ready to use:
```python
from phase6_experiments import e6_3_transform_suite_conclusive_test
from pathlib import Path

result = e6_3_transform_suite_conclusive_test(
    golden_faces_dir=Path("../ECDD_Experiment_Data/real")
)
```

**E6.4 (OOD separation)** - Ready to use:
```python
from phase6_experiments import e6_4_out_of_scope_separation_test
from pathlib import Path

result = e6_4_out_of_scope_separation_test(
    ood_dir=Path("../ECDD_Experiment_Data/ood"),
    face_detector_backend="stub"  # or "mediapipe"
)
```

**Results written to**: `phase6_results/E6.1.json` through `E6.4.json`

---

## 🔐 CI Gates Usage

All gates are CLI tools that exit with code 0 (pass) or non-zero (fail).

### Gate G1: Pixel Equivalence

Tests that preprocessing is deterministic.

```bash
cd ci/gates
python g1_pixel_equivalence.py \
  --image-dir ../../ECDD_Experiment_Data/real \
  --tolerance 1e-6 \
  --max-images 10
```

**Exit code**: 0 if all images produce identical pixels on repeated runs.

---

### Gate G2: Guardrails

Tests face detection consistency and OOD abstention.

```bash
python g2_guardrail.py \
  --face-dir ../../ECDD_Experiment_Data/real \
  --ood-dir ../../ECDD_Experiment_Data/ood \
  --backend stub \
  --max-images 10
```

**Exit code**: 0 if face detection is consistent and OOD images abstain.

---

### Gate G3: Model Semantics

Tests that model produces sensible predictions.

```bash
python g3_model_semantics.py \
  --real-dir ../../ECDD_Experiment_Data/real \
  --fake-dir ../../ECDD_Experiment_Data/fake \
  --threshold 0.5 \
  --max-samples 10
```

**Current status**: Uses mock predictions. Integrate with trained model.

---

### Gate G4: Calibration

Tests that calibration parameters improve reliability.

```bash
python g4_calibration.py \
  --calibration-params /path/to/calibration_params.json \
  --calibration-data /path/to/calibration_logits.json
```

**Exit code**: 0 if calibration exists and improves ECE.

---

### Gate G5: Quantization Parity

Tests that TFLite model matches float model.

```bash
python g5_quantization.py \
  --float-model /path/to/model.onnx \
  --tflite-model /path/to/model.tflite \
  --tolerance 0.05
```

**Current status**: Uses mock data. Integrate with trained models.

---

### Gate G6: Release Gate

Final pre-deployment validation.

```bash
python g6_release.py \
  --gates-dir ./results \
  --metrics-file /path/to/metrics.json \
  --bundle-dir /path/to/bundle
```

**Exit code**: 0 if all checks pass (previous gates, metrics, versioning).

---

## 🔑 Golden Hash System

### Generate Golden Hashes

```bash
cd ecdd_core/golden
python generate_golden_hashes.py \
  --golden-dir ../../ECDD_Experiment_Data/real \
  --output golden_hashes.json \
  --max-images 30 \
  --face-backend stub
```

**Output**: JSON file with hashes for S0-S4 stages (preprocessing).

**Stages**:
- S0: Raw image bytes
- S1: Decoded RGB tensor
- S2: Face crop boxes
- S3: Resized 256×256 tensor
- S4: Normalized tensor
- S5-S8: Require trained model (marked as NOT_IMPLEMENTED)

---

### Verify Golden Hashes

```bash
python verify_golden_hashes.py \
  --golden-hashes golden_hashes.json \
  --golden-dir ../../ECDD_Experiment_Data/real \
  --face-backend stub
```

**Exit code**: 0 if current pipeline outputs match golden hashes.

**Use case**: Detect preprocessing regressions or environment differences.

---

## 🔧 Export Infrastructure

### TFLite Conversion (Scaffolding)

```python
from ecdd_core.export.tflite_converter import (
    convert_onnx_to_tflite,
    TFLiteConversionConfig
)
from pathlib import Path

config = TFLiteConversionConfig(
    input_shape=(1, 3, 256, 256),
    quantization="dynamic",  # or "int8" or "none"
)

convert_onnx_to_tflite(
    onnx_model_path=Path("model.onnx"),
    output_path=Path("model.tflite"),
    config=config
)
```

**Status**: Requires TensorFlow installation. Scaffolding ready.

---

### Model Parity Validation

```python
from ecdd_core.export.model_parity import compute_parity_metrics
import numpy as np

float_outputs = np.array([...])  # Model outputs
tflite_outputs = np.array([...])

result = compute_parity_metrics(
    float_outputs,
    tflite_outputs,
    tolerance=0.05
)

print(result.passed)
print(result.max_abs_diff)
print(result.rank_correlation)
```

---

## 📋 Integration with Existing Infrastructure

### Calibration Module

Phase 4 uses existing calibration utilities:
- `ecdd_core.calibration.temperature_scaling`
- `ecdd_core.calibration.platt_scaling`
- `ecdd_core.calibration.operating_point`
- `ecdd_core.calibration.calibration_set_contract`

### Evaluation Module

Phase 6 uses existing evaluation utilities:
- `ecdd_core.eval.transforms_suite`
- `ecdd_core.eval.splits`

### Pipeline Module

Golden hashes and gates use:
- `ecdd_core.pipeline.decode`
- `ecdd_core.pipeline.face`
- `ecdd_core.pipeline.preprocess`
- `ecdd_core.pipeline.guardrails`

---

## ⚠️ Mock vs Real Implementations

### Ready for Production Use ✅

- Phase 4: All experiments (needs calibration data)
- Phase 6 E6.3, E6.4: Transform suite and OOD tests
- CI Gates G1, G2: Preprocessing and guardrails
- Golden hashes S0-S4: Preprocessing stages

### Requires Model Integration ⚠️

- Phase 5: All experiments (mock data currently)
- Phase 6 E6.1, E6.2: Need model for evaluation
- CI Gates G3, G4, G5: Mock predictions currently
- Golden hashes S5-S8: Require model inference

**How to integrate**: Replace mock implementations with actual model loading and inference. Search for `is_mock = True` in code.

---

## 🎯 Example Workflow

### 1. Train Models (separate task)
```bash
# In deepfake-patch-audit/
python train_improved.py
python scripts/export_teacher.py
python scripts/export_student.py
```

### 2. Generate Calibration Data
```bash
# Run model on calibration set, save logits
python scripts/generate_calibration_logits.py \
  --model teacher.onnx \
  --calib-set /path/to/calib \
  --output calibration_logits.json
```

### 3. Run Phase 4 Experiments
```bash
cd ECDD_Experimentation/PhaseWise_Experiments_ECDD
python phase4_experiments.py ../calibration_logits.json
```

### 4. Export to TFLite
```bash
# Implement TFLite conversion
# Or use existing deepfake-patch-audit export scripts
```

### 5. Run Phase 5 Experiments
```bash
# Update phase5_experiments.py with model paths
python phase5_experiments.py
```

### 6. Run CI Gates
```bash
cd ../ci/gates
./run_all_gates.sh  # Create this script
```

### 7. Generate Golden Hashes
```bash
cd ../ecdd_core/golden
python generate_golden_hashes.py --golden-dir ../../ECDD_Experiment_Data/real
```

---

## 📊 Results Format

All experiments output JSON with this structure:

```json
{
  "experiment_id": "E4.2",
  "name": "Temperature scaling fit and verification",
  "passed": true,
  "details": {
    "temperature": 1.23,
    "pre_ece": 0.156,
    "post_ece": 0.089,
    ...
  }
}
```

Gates exit with codes:
- `0`: PASS
- `1`: FAIL

---

## 🐛 Troubleshooting

### Import Errors

If you get import errors, ensure you're running from the correct directory:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Team-Converge/ECDD_Experimentation"
```

### Missing Dependencies

Install required packages:
```bash
pip install numpy pillow scipy
pip install mediapipe  # Optional, for face detection
pip install tensorflow onnx-tf  # Optional, for TFLite conversion
```

### Mock Data Warnings

If you see "Using mock predictions" warnings, this is expected until models are trained. The infrastructure is ready but waiting for actual models.

---

## 📚 Further Reading

- `IMPLEMENTATION_STATUS_UPDATED.md`: Detailed implementation status
- `TODO_COMPREHENSIVE_AUDIT.md`: Original audit that identified gaps
- `ci/gates/README.md`: CI integration guide
- `ECDD_Paper_DR_3_Experimentation.md`: Full experiment specifications

---

**Created**: 2026-01-07  
**Status**: All infrastructure complete and ready for model integration
