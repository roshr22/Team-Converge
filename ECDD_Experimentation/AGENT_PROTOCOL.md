# ECDD Agent Protocol

> Hard operating rules for any agent or human working on this codebase. No exceptions.

---

## Core Workflow Rules

1. **Read repo first** — Scan and understand existing code before proposing changes.
2. **Propose a concrete plan** — Document the plan with expected outcomes.
3. **BEFORE any edits** — Append the plan into `ECDD_Experimentation/RUNLOG.md`.
4. **Execute in small, atomic chunks** — Each chunk should be testable independently.
5. **After EACH chunk**:
   - Commit with message: `"Plan Step X: <short description>"`
   - Immediately append to `RUNLOG.md`:
     - Files changed
     - Tight diff summary (what/why)
     - Commands run
     - Test/metric outputs
6. **No silent changes** — Every change must be logged.
7. **No squashing** — Preserve atomic commit history.
8. **RUNLOG.md is append-only** — Never delete or modify historical entries.

---

## Preprocessing & Calibration Freeze Policy

> **CRITICAL**: No changes to the following without explicit update to `DECISIONS.md`:

- Preprocessing pipeline (decode → resize → normalize)
- Threshold values (fake_threshold, real_threshold)
- Calibration method or parameters (temperature T, Platt a/b)
- Pooling method or configuration
- Face crop/alignment routing or parameters
- Normalization constants (mean, std)
- Resize kernel or target size

### Required Before Any Frozen-Field Change

1. **Update `DECISIONS.md`** with:
   - What changed
   - Why it changed
   - Expected metric impact (quantify if possible)
   - Link to supporting experiment/evidence

2. **Rerun Acceptance Tests** (see `ACCEPTANCE_TESTS.md`)

3. **Log results in `RUNLOG.md`** before proceeding

---

## File Modification Hierarchy

| File | Purpose | Modification Rules |
|------|---------|-------------------|
| `RUNLOG.md` | Append-only execution log | Append only, never delete |
| `DECISIONS.md` | Frozen policy decisions | Update only with rationale |
| `AGENT_PROTOCOL.md` | This file — operating rules | Update with team consensus |
| `ACCEPTANCE_TESTS.md` | Done-check definitions | Add tests, don't remove |
| `policy_contract.yaml` | Detailed policy knobs | Requires DECISIONS.md update first |

---

## Commit Message Format

```
Plan Step <N>: <Short description>

Files: <comma-separated list>
What: <1-line summary>
Why: <1-line rationale>
```

---

## Quick Checklist Before Each Commit

- [ ] Plan is in RUNLOG.md?
- [ ] Change is atomic and testable?
- [ ] Frozen fields unchanged OR DECISIONS.md updated?
- [ ] Acceptance tests pass?
- [ ] Commit message follows format?
