# ECDD Decisions Registry

> Authoritative record of frozen deployment-equivalent policies. Any drift requires explicit update here with rationale.

---

## Deployment-Equivalence Contract

> **Purpose**: Freeze all preprocessing, calibration, and decision-boundary policies to ensure training/inference parity.

### 1. Decode Library & Path

| Field | Frozen Value | Source |
|-------|-------------|--------|
| Library | `pillow` | `policy_contract.yaml` line 22 |
| Version Pin | `>=10.0.0` | `policy_contract.yaml` line 23 |
| Supported Formats | `jpeg, jpg, png, webp` | `policy_contract.yaml` line 27 |

**Rationale**: Pillow provides deterministic decode across platforms. OpenCV has BGR ordering issues.

---

### 2. sRGB & Color Space

| Field | Frozen Value | Source |
|-------|-------------|--------|
| Channel Order | `RGB` | `policy_contract.yaml` line 39 |
| Color Space | `sRGB` | `policy_contract.yaml` line 40 |
| ICC Profile Handling | `ignore` | `policy_contract.yaml` line 41 |

**Rationale**: sRGB is the web default. ICC profile application introduces platform variance.

---

### 3. EXIF Handling

| Field | Frozen Value | Source |
|-------|-------------|--------|
| Apply Orientation | `true` | `policy_contract.yaml` line 34 |
| Strip After Decode | `true` | `policy_contract.yaml` line 35 |

**Rationale**: EXIF orientation must be applied to get intended pixel layout. Stripping prevents metadata leakage.

---

### 4. Face Crop/Alignment Policy

| Field | Frozen Value | Source |
|-------|-------------|--------|
| Detector Model | `mediapipe_face_detection` | `policy_contract.yaml` line 82 |
| Confidence Threshold | `0.8` | `policy_contract.yaml` line 88 |
| Use Landmarks | `true` | `policy_contract.yaml` line 92 |
| Crop Margin | `0.3` | `policy_contract.yaml` line 93 |
| Deterministic | `true` | `policy_contract.yaml` line 94 |
| Multi-Face Policy | `max_p_fake` | `policy_contract.yaml` line 98 |
| Min Face Size | `64x64` | `policy_contract.yaml` lines 103-104 |

**Rationale**: MediaPipe provides consistent cross-platform detection. Margin of 0.3 captures context without noise.

---

### 5. Resize Kernel

| Field | Frozen Value | Source |
|-------|-------------|--------|
| Target Size | `256x256` | `policy_contract.yaml` line 56, `preprocess.py` line 23 |
| Interpolation | `bilinear` | `policy_contract.yaml` line 57, `preprocess.py` line 24 |
| Antialias | `true` | `policy_contract.yaml` line 58 |

**Rationale**: Bilinear is faster and sufficient for detection. 256x256 matches training resolution.

---

### 6. Normalization Constants

| Field | Frozen Value | Source |
|-------|-------------|--------|
| Mean | `[0.485, 0.456, 0.406]` | `policy_contract.yaml` line 63, `preprocess.py` line 27 |
| Std | `[0.229, 0.224, 0.225]` | `policy_contract.yaml` line 64, `preprocess.py` line 28 |
| Output Dtype | `float32` | `preprocess.py` line 31 |
| Layout | `CHW` | `preprocess.py` line 57 |

**Rationale**: ImageNet normalization is the training baseline. Must match exactly for inference parity.

---

### 7. Pooling Rule

| Field | Frozen Value | Source |
|-------|-------------|--------|
| Method | `attention` | `policy_contract.yaml` line 152 |
| Compute Dtype | `float32` | `policy_contract.yaml` line 161 |
| Softmax Temperature | `1.0` | `policy_contract.yaml` line 162 |
| Deterministic Ops Required | `true` | `policy_contract.yaml` line 166 |

**Rationale**: Attention pooling improves needle detection. Float32 ensures determinism.

---

### 8. Calibration Method

| Field | Frozen Value | Source |
|-------|-------------|--------|
| Method | `temperature_scaling` | `policy_contract.yaml` line 178 |
| Temperature T | `1.0` (TBD - fit on calib set) | `policy_contract.yaml` line 182 |
| Min Calib Set Size | `500` | `policy_contract.yaml` line 191 |
| Sampling | `stratified_by_source` | `policy_contract.yaml` line 192 |

**Rationale**: Temperature scaling is simple, effective, and preserves ranking. Single parameter minimizes overfit.

---

### 9. Thresholding & Abstain Policy

| Field | Frozen Value | Source |
|-------|-------------|--------|
| Primary Metric | `FPR @ 0.05` | `policy_contract.yaml` lines 206-207 |
| Secondary Metric | `FNR @ 0.10` | `policy_contract.yaml` lines 211-212 |
| Fake Threshold | `0.7` (TBD) | `policy_contract.yaml` line 216 |
| Real Threshold | `0.3` (TBD) | `policy_contract.yaml` line 217 |
| Abstain Enabled | `true` | `policy_contract.yaml` line 222 |
| Max Abstain Rate | `0.15` | `policy_contract.yaml` line 223 |

**Rationale**: FPR-constrained threshold protects real users. Abstain band handles uncertainty.

---

## Change Log

| Date | Field | Old Value | New Value | Rationale | Metric Impact | Author |
|------|-------|-----------|-----------|-----------|---------------|--------|
| 2026-01-09 | (initial) | — | — | Initial freeze from `policy_contract.yaml` | — | Agent |

---

## How to Update This Document

1. **Never delete rows** from the Change Log
2. **Add a new row** for each change with:
   - Date (ISO format)
   - Field being changed
   - Old and new values
   - Rationale (link to experiment if available)
   - Expected metric impact
   - Author
3. **Rerun acceptance tests** after any change
4. **Log results in RUNLOG.md**
