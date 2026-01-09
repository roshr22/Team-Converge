# ECDD Experimentation Run Log

> **Append-only log** — Do not modify historical entries. Each section records a date/time, plan, actions, diffs, test outputs, and next step.

---

## 2026-01-09T19:36:00+05:30 — Initial Codebase Scan & Step-1 Plan

### Objective
Bootstrap governance files and shared memory for finetuning workflow.

### Repository Scan Summary
- **Test runner**: `run_all_tests.py` (449 lines), with `run_all_tests.bat`/`.sh` wrappers
- **CI gates**: `ci/gates/g1_pixel_equivalence.py` through `g6_release.py`
- **Pipeline**: `ecdd_core/pipeline/` (decode, preprocess, face, guardrails, reason_codes, contracts)
- **Calibration**: `ecdd_core/calibration/temperature_scaling.py` (fit_temperature, ECE), `platt_scaling.py`, `operating_point.py`
- **Eval**: `ecdd_core/eval/` (splits.py, transforms_suite.py)
- **Policy**: `policy_contract.yaml` (391 lines freeze all policy knobs)
- **Training**: `Training/kaggle_training_notebook.py`
- **Missing**: `DECISIONS.md`, `RUNLOG.md`, `AGENT_PROTOCOL.md`, `ACCEPTANCE_TESTS.md`

### Step-1 Plan: Bootstrap Governance + Shared Memory
1. **Create RUNLOG.md** (this file) — append-only log with template section
2. **Create AGENT_PROTOCOL.md** — hard operating rules + preprocessing freeze policy
3. **Create ACCEPTANCE_TESTS.md** — define "done" checks with exact commands
4. **Create DECISIONS.md** — Deployment-Equivalence Contract freezing preprocessing, calibration, thresholds

### Actions Taken
- [x] Created `RUNLOG.md` (this file) — append-only log template
- [x] Created `AGENT_PROTOCOL.md` — hard rules + preprocessing freeze policy
- [x] Created `ACCEPTANCE_TESTS.md` — exact commands for quick smoke tests, gate tests, and new Step 2-4 tests
- [x] Created `DECISIONS.md` — Deployment-Equivalence Contract with 9 frozen policy sections

### Diffs Summary
| File | Lines | What | Why |
|------|-------|------|-----|
| `RUNLOG.md` | 44 | New file, append-only log template | Track all changes atomically |
| `AGENT_PROTOCOL.md` | 77 | New file, operating rules + freeze policy | Prevent silent/untracked changes |
| `ACCEPTANCE_TESTS.md` | 116 | New file, test commands + expected outputs | Define "done" checks |
| `DECISIONS.md` | 135 | New file, Deployment-Equivalence Contract | Freeze preprocessing/calibration/thresholds |

### Test Outputs
```
py -c "from ecdd_core.pipeline import decode_image_bytes, resize_rgb_uint8, normalize_rgb_uint8; print('Pipeline imports OK')"
> Pipeline imports OK

py -c "from ecdd_core.calibration import fit_temperature, expected_calibration_error; print('Calibration imports OK')"
> Calibration imports OK
```

**Status**: ✅ Step 1 COMPLETE

### Next Step
Proceed to Step 2: Implement evaluation diagnostics path.

---

## 2026-01-09T19:45:00+05:30 — Step 2: Implement Evaluation Diagnostics

### Objective
Add evaluation/diagnostics path for large online datasets with metrics grouped by source and compression.

### Plan
1. Create `evaluation/` directory under `ECDD_Experimentation/`
2. Create `dataset_index.py` — schema for dataset index (label, source family, compression bucket)
3. Create `metrics.py` — compute AUC, AP, F1, TPR@FPR, FPR@TPR, confusion matrix
4. Create `evaluate_model.py` — CLI to run evaluation on indexed dataset
5. Create `plot_diagnostics.py` — confidence histograms for misclassified samples
6. Write CSV + JSON summaries + plots to `results/`

### Actions Taken
- [x] Created `evaluation/` directory
- [x] Created `evaluation/__init__.py`
- [x] Created `evaluation/dataset_index.py` — SampleEntry + DatasetIndex classes + index persistence
- [x] Created `evaluation/metrics.py` — AUC, AP, F1, TPR@FPR, FPR@TPR, confusion matrix (numpy only)
- [x] Created `evaluation/plot_diagnostics.py` — confidence histogram, reliability curve, confusion matrix, ROC, grouped bar
- [x] Created `evaluation/evaluate_model.py` — CLI with demo mode, JSON/CSV output, all plots
- [x] Fixed numpy 2.0 compatibility (trapz → trapezoid)
- [x] Fixed Windows cp1252 encoding issue (removed Unicode emoji)

### Diffs Summary
| File | Lines | What | Why |
|------|-------|------|-----|
| `evaluation/__init__.py` | 31 | Package exports | Module organization |
| `evaluation/dataset_index.py` | 228 | Sample/index schema, JSON persistence | External dataset indexing |
| `evaluation/metrics.py` | 295 | Binary metrics without sklearn | Compute AUC/AP/F1/TPR@FPR/FPR@TPR |
| `evaluation/plot_diagnostics.py` | 308 | 6 plot functions | Diagnostic visualizations |
| `evaluation/evaluate_model.py` | 366 | CLI with demo mode | Run evaluations, output JSON/CSV/plots |

### Test Outputs
```
py evaluation/evaluate_model.py --demo --output-dir results
> Running demo evaluation with synthetic data...
> [OK] Evaluation complete!
>    JSON: results\20260109_194850_metrics.json
>    CSV:  results\20260109_194850_metrics.csv
>    Plots: results/20260109_194850_*.png
>
>    Overall AUC: 0.9688
>    Overall F1:  0.8970

Generated files:
- 20260109_194850_metrics.json (18 KB)
- 20260109_194850_metrics.csv (720 B)
- 20260109_194850_confusion.png (50 KB)
- 20260109_194850_roc.png (51 KB)
- 20260109_194850_misclassified_confidence.png (43 KB)
- 20260109_194850_auc_by_source.png (36 KB)
- 20260109_194850_auc_by_compression.png (30 KB)
```

**Status**: ✅ Step 2 COMPLETE

### Next Step
Proceed to Step 3: Calibration + deployment-relevant checkpointing.

---

