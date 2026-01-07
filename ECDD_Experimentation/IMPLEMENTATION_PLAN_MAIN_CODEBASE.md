# Implementation Plan: Main Training + Inference Codebase (ECDD_Experimentation)

**Constraint**: No changes to `Team-Converge/deepfake-patch-audit/`.

**Goal**: Implement the entire main training + inference codebase inside `Team-Converge/ECDD_Experimentation/` while a partner runs:
- the two CUDA finetunes
- Phase 4/5/6 experiments

This plan is contract-first and fully aligned to:
- `ECDD_Paper_DR_3_Architecture_v1.1.md`
- `ECDD_Paper_DR_3_Experimentation.md`
- `policy_contract.yaml`
- `manual_review_protocol.md`
- `TODO_COMPREHENSIVE_AUDIT.md`
- `ECDD_Collection1_Doc1.md`

---

## 0) Non-negotiable contracts (must freeze early)

### 0.1 Preprocessing Equivalence Contract (single source of truth)
Implement a canonical pipeline module used by BOTH training and inference:
- authoritative decode library
- EXIF orientation applied
- alpha handling policy
- RGB channel order invariants
- gamma policy invariants (sRGB-only)
- dtype/range policy
- deterministic resize kernel + rounding behavior
- fixed normalization constants

### 0.2 Guardrails Contract
Implement deterministic, parameterized guardrails with explicit reason codes:
- face detector version pinning + confidence threshold
- multi-face policy (max vs largest-face)
- minimum face size threshold
- blur metric + threshold
- compression proxy + threshold
- OOD/no-face policy (abstain + reason code)

### 0.3 Patch Grid + Pooling Contract
Freeze:
- patch-logit map shape (H×W)
- mapping from heatmap cell to image coordinates
- pooling choice (top-k vs attention) + parameters
- numerical determinism and quantization survivability

### 0.4 Calibration & Operating Point Contract
Freeze and version:
- calibration method (temperature vs platt)
- calibration set contract (sampling rules + disjointness)
- operating point constraints (e.g., FPR ≤ 5% on real faces)
- abstain band semantics and UI meaning

### 0.5 Artifact Bundle Contract
Every release bundle must include:
- preprocessing version + hash
- model weights hash
- pooling config
- calibration parameters
- threshold policy
- guardrail thresholds
- evaluation report used to approve release

---

## 1) Directory structure to build (inside ECDD_Experimentation)

Create a new top-level package:

```
Team-Converge/ECDD_Experimentation/ecdd_core/
  __init__.py

  pipeline/
    decode.py
    preprocess.py
    face.py
    guardrails.py
    reason_codes.py
    contracts.py

  models/
    __init__.py
    ladeda_student_wrapper.py
    ladeda_teacher_wrapper.py
    pooling.py
    patch_grid.py

  calibration/
    temperature_scaling.py
    platt_scaling.py
    operating_point.py
    abstain_policy.py
    calibration_set_contract.py

  export/
    onnx_export.py
    tflite_export.py
    post_quant_calibration.py
    artifact_manifest.py

  parity/
    float_vs_tflite.py
    patch_map_parity.py
    pooled_logit_parity.py

  monitoring/
    schema.py
    drift_triggers.py
    logger.py

  dataset_governance/
    lineage_tags.py
    leakage_scan.py
    validate_lineage.py

  golden/
    golden_sets.py
    golden_hashes.py
    generate_golden_hashes.py

  ci/
    gates/
      g1_pixel_equivalence.py
      g2_guardrails.py
      g3_model_semantics.py
      g4_calibration.py
      g5_quantization_parity.py
      g6_release_battery.py
```

---

## 2) PhaseWise experiments to implement (critical gaps)

Per `TODO_COMPREHENSIVE_AUDIT.md` and `ECDD_Paper_DR_3_Experimentation.md`, implement the missing scripts:

- `Team-Converge/ECDD_Experimentation/PhaseWise_Experiments_ECDD/phase4_experiments.py`
  - E4.1–E4.6 Calibration, thresholds, abstain semantics

- `Team-Converge/ECDD_Experimentation/PhaseWise_Experiments_ECDD/phase5_experiments.py`
  - E5.1–E5.5 Quantization and float-to-TFLite parity, post-quant calibration

- `Team-Converge/ECDD_Experimentation/PhaseWise_Experiments_ECDD/phase6_experiments.py`
  - E6.1–E6.4 Realistic evaluation battery: source/time split, transform suite, OOD separation

Each script must:
- load `policy_contract.yaml`
- use the canonical pipeline
- use golden sets and/or generated hashes
- output JSON results into `phase4_results/`, `phase5_results/`, `phase6_results/`

---

## 3) Parallel work plan (you vs partner)

### 3.1 You (codebase)
1. Implement `ecdd_core/pipeline/*` (decode + preprocess + invariants)
2. Implement `ecdd_core/guardrails/*` (face routing + quality gates + reason codes)
3. Implement `ecdd_core/calibration/*` (temperature/platt + operating point + abstain band)
4. Implement golden set harness + S0–S8 stage hashes
5. Implement export + quantization pipeline (TFLite) + artifact manifest
6. Implement parity harness (float vs TFLite at prob/patch/pool)
7. Implement CI gates G1–G6 scripts
8. Implement `phase4_experiments.py`, `phase5_experiments.py`, `phase6_experiments.py`

### 3.2 Partner (CUDA laptop)
1. Run the two finetunes and produce checkpoints + logs
2. Run Phase 4 experiments once `phase4_experiments.py` exists
3. Run Phase 5 parity once TFLite exports + parity harness exist
4. Run Phase 6 battery once `phase6_experiments.py` exists

---

## 4) Synchronization points

### Sync 1: Contract Freeze (Day 1–2)
Deliver:
- canonical pipeline
- guardrails stubs + reason codes
- golden sets definition + hash generator scaffold

### Sync 2: Phase 4 ready (Day 3–5)
Deliver:
- calibration module
- `phase4_experiments.py`

### Sync 3: Phase 5 ready (Day 5–7)
Deliver:
- TFLite export + manifest
- parity harness
- `phase5_experiments.py`

### Sync 4: Phase 6 ready (Week 2)
Deliver:
- split tooling + transform suite
- `phase6_experiments.py`

---

## 5) Definition of Done

Done when:
- canonical pipeline and guardrails are implemented and match `policy_contract.yaml`
- golden set S0–S8 hashes generated and validated
- Phase 4/5/6 scripts exist and run reproducibly
- quantization + parity gates pass or fail with explicit metrics and tolerances
- monitoring schema implemented (privacy-preserving)
- dataset governance scripts exist (lineage enforcement)

---

## 6) Immediate next coding steps (start now)

1. Create `ecdd_core/` package skeleton
2. Implement `pipeline/decode.py` and `pipeline/preprocess.py`
3. Implement `pipeline/reason_codes.py` and `pipeline/contracts.py`
4. Implement `golden/generate_golden_hashes.py` (S0–S4 first)
5. Implement a minimal `phase4_experiments.py` scaffold to unblock partner
