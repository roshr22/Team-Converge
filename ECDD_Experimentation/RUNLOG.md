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
Running smoke tests after commit...

### Next Step
Commit Step 1, run smoke tests, then proceed to Step 2 (evaluation diagnostics).

---
