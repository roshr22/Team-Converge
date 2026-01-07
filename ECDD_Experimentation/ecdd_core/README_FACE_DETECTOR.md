# Face Detector Backend (Phase 2)

We standardize on **MediaPipe Face Detection** for Phase 2 guardrails and for golden hash stage **S2** (face boxes + crop hashes).

## Why MediaPipe
- Easy to install
- Runs on CPU
- Stable API
- Good enough for determinism + routing experiments

## Install
```bash
pip install mediapipe
```

## Configure
In code, use:
```python
from ecdd_core.pipeline.face import FaceDetectorConfig
cfg = FaceDetectorConfig(backend="mediapipe", min_confidence=0.5, multi_face_policy="max")
```

## Generate Golden Hashes with S2
This will compute S0–S4 and also S2 face box + crop hashes:
```bash
python ecdd_core/golden/generate_golden_hashes.py --face-backend mediapipe
```

Output:
- `ecdd_core/golden_hashes_s0_s4_s2_mediapipe.json`

## Note on reproducibility
If `backend="mediapipe"` is selected and mediapipe is not installed, the pipeline **fails loudly**.
Silent fallback would invalidate audits.
