"""Patch adapter for enforcing patch grid alignment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchAdapter(nn.Module):
    """
    Enforces patch grid alignment for the teacher model.
    Extracts patches in a grid pattern for consistent spatial analysis.
    """

    def __init__(self, patch_size=224, stride=16, enable_padding=True):
        """
        Args:
            patch_size: Size of each patch (height = width)
            stride: Stride between patches
            enable_padding: Add padding to maintain grid structure
        """
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.enable_padding = enable_padding

    def forward(self, x):
        """
        Extract patches from input in grid pattern.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Patches as (B, num_patches, C, patch_size, patch_size)
            and patch positions as (num_patches, 2)
        """
        B, C, H, W = x.shape

        if self.enable_padding:
            # Calculate padding needed for grid alignment
            pad_h = (self.stride - H % self.stride) % self.stride
            pad_w = (self.stride - W % self.stride) % self.stride
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = x.shape[2:]

        # Extract patches using unfold
        # unfold(dimension, size, step)
        patches = x.unfold(2, self.patch_size, self.stride).unfold(
            3, self.patch_size, self.stride
        )
        B, C, num_h, num_w, ph, pw = patches.shape

        # Reshape to (B, num_patches, C, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, num_h * num_w, C, ph, pw)

        # Compute patch positions
        positions = []
        for i in range(num_h):
            for j in range(num_w):
                positions.append([i * self.stride, j * self.stride])
        positions = torch.tensor(positions, dtype=torch.float32)

        return patches, positions

    def get_grid_shape(self, h, w):
        """Get output grid shape for given input size."""
        if self.enable_padding:
            h = h + (self.stride - h % self.stride) % self.stride
            w = w + (self.stride - w % self.stride) % self.stride

        num_h = (h - self.patch_size) // self.stride + 1
        num_w = (w - self.patch_size) // self.stride + 1
        return num_h, num_w
