"""Top-K pooling for aggregating patch-level predictions."""

import torch
import torch.nn as nn
import numpy as np


class TopKLogitPooling(nn.Module):
    """
    Top-K pooling for selecting high-confidence patches and aggregating their logits.

    Key features:
    - Selects patches with HIGHEST fake-class logits (most fake-like regions)
    - Dynamic K: K = max(min_k, ceil(r × num_patches))
    - Works with both spatial (B, 1, H, W) and flattened (B, num_patches) inputs
    - Aggregates via mean or max over top-k logits

    Example:
        - 31×31 patch map (961 patches) with r=0.1, min_k=5 → Select top 97 patches
        - 126×126 patch map (15,876 patches) with r=0.1, min_k=5 → Select top 1,588 patches
    """

    def __init__(self, r=0.1, min_k=5, aggregation="mean"):
        """
        Args:
            r: Ratio of patches to select (K = ceil(r × num_patches))
            min_k: Minimum number of patches to select (floor for K)
            aggregation: How to aggregate top-k logits ('mean' or 'max')
        """
        super().__init__()
        self.r = r
        self.min_k = min_k
        self.aggregation = aggregation

    def forward(self, patch_logits, return_indices=False):
        """
        Select and aggregate top-k patches based on logit values.

        Args:
            patch_logits: Patch-wise logits
                - Shape (B, 1, H, W): Spatial patch-logit map
                - Shape (B, num_patches): Flattened patch logits
            return_indices: Return indices of selected patches

        Returns:
            image_logit: (B, 1) - aggregated image-level logit
            top_indices (optional): (B, k) - indices of selected patches
        """
        # Handle spatial input: (B, 1, H, W) → (B, H*W)
        if patch_logits.dim() == 4:
            B, C, H, W = patch_logits.shape
            assert C == 1, "Expected single channel for patch logits"
            patch_logits = patch_logits.view(B, -1)  # (B, H*W)

        B, num_patches = patch_logits.shape

        # Calculate dynamic K
        k = max(self.min_k, int(np.ceil(self.r * num_patches)))
        k = min(k, num_patches)  # Clamp to actual number of patches

        # Select top-k patches with highest logits (most fake-like)
        top_values, top_indices = torch.topk(patch_logits, k, dim=1)  # (B, k)

        # Aggregate top-k logits
        if self.aggregation == "mean":
            image_logit = top_values.mean(dim=1, keepdim=True)  # (B, 1)
        elif self.aggregation == "max":
            image_logit = top_values.max(dim=1, keepdim=True)[0]  # (B, 1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        if return_indices:
            return image_logit, top_indices
        return image_logit


class MeanPooling(nn.Module):
    """Simple mean pooling across all patches (baseline without selection)."""

    def __init__(self):
        super().__init__()

    def forward(self, patch_logits):
        """
        Average all patches (no selection).

        Args:
            patch_logits: (B, 1, H, W) or (B, num_patches)

        Returns:
            image_logit: (B, 1) - averaged logit
        """
        # Handle spatial input: (B, 1, H, W) → (B, H*W)
        if patch_logits.dim() == 4:
            B, C, H, W = patch_logits.shape
            patch_logits = patch_logits.view(B, -1)

        # Average all patches
        return patch_logits.mean(dim=1, keepdim=True)  # (B, 1)
