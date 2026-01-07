# CI Gates for ECDD Pipeline

This directory contains mandatory gate scripts that enforce deployment constraints.

## Gates Overview

| Gate | Purpose | Command |
|------|---------|---------|
| **G1** | Pixel Equivalence | Ensures preprocessing is deterministic |
| **G2** | Guardrails | Validates face detection and OOD handling |
| **G3** | Model Semantics | Checks model produces sensible outputs |
| **G4** | Calibration | Validates calibration improves reliability |
| **G5** | Quantization Parity | Ensures TFLite matches float model |
| **G6** | Release Gate | Final pre-deployment checklist |

## Usage

Each gate is a standalone Python script that exits with code 0 (pass) or non-zero (fail).

### Gate G1: Pixel Equivalence

```bash
python g1_pixel_equivalence.py \
  --image-dir ../ECDD_Experiment_Data/real \
  --tolerance 1e-6 \
  --max-images 10
```

### Gate G2: Guardrails

```bash
python g2_guardrail.py \
  --face-dir ../ECDD_Experiment_Data/real \
  --ood-dir ../ECDD_Experiment_Data/ood \
  --backend stub
```

### Gate G3: Model Semantics

```bash
python g3_model_semantics.py \
  --real-dir ../ECDD_Experiment_Data/real \
  --fake-dir ../ECDD_Experiment_Data/fake \
  --threshold 0.5
```

### Gate G4: Calibration

```bash
python g4_calibration.py \
  --calibration-params /path/to/calibration_params.json \
  --calibration-data /path/to/calibration_logits.json
```

### Gate G5: Quantization Parity

```bash
python g5_quantization.py \
  --float-model /path/to/model.onnx \
  --tflite-model /path/to/model.tflite \
  --tolerance 0.05
```

### Gate G6: Release Gate

```bash
python g6_release.py \
  --gates-dir ./results \
  --metrics-file /path/to/metrics.json \
  --bundle-dir /path/to/bundle
```

## Integration with CI

Add to your CI pipeline (e.g., GitHub Actions, GitLab CI):

```yaml
stages:
  - test
  - gates
  - deploy

gates:
  stage: gates
  script:
    - cd ci/gates
    - python g1_pixel_equivalence.py --image-dir ../../ECDD_Experiment_Data/real
    - python g2_guardrail.py --face-dir ../../ECDD_Experiment_Data/real --ood-dir ../../ECDD_Experiment_Data/ood
    - python g3_model_semantics.py --real-dir ../../ECDD_Experiment_Data/real --fake-dir ../../ECDD_Experiment_Data/fake
    # Add G4, G5, G6 as models become available
  only:
    - main
    - release/*
```

## Notes

- **G1, G2**: Ready to use with existing infrastructure
- **G3, G4, G5**: Currently use mock data; integrate with trained models when available
- **G6**: Aggregates all previous gates for final release decision

All gates are designed to fail CI builds if constraints are violated.
