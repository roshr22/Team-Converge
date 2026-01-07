"""CI gate scripts for enforcing deployment constraints.

Gates (from ECDD_Paper_DR_3_Experimentation.md):
- G1: Pixel Equivalence
- G2: Guardrail Gate
- G3: Model Semantics
- G4: Calibration
- G5: Quantization Parity
- G6: Release Gate

Each gate should fail CI if constraints are violated.
"""
