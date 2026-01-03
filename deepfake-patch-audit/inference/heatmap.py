"""Heatmap visualization for patch-level predictions."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


class HeatmapVisualizer:
    """
    Visualize patch-level predictions as heatmaps.
    Shows which patches contribute most to the deepfake classification.
    """

    def __init__(self, patch_size=224, stride=16):
        """
        Args:
            patch_size: Size of patches used in model
            stride: Stride between patches
        """
        self.patch_size = patch_size
        self.stride = stride

    def generate_heatmap(self, image_path, model, device="cuda", threshold=0.5):
        """
        Generate patch-wise heatmap for an image.

        Args:
            image_path: Path to image file
            model: Trained model with patch output
            device: Device for computation
            threshold: Decision threshold

        Returns:
            dict with heatmap and visualization
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        h, w = image.size

        # Preprocess for patches
        # This is a placeholder - actual implementation would extract patches
        patches = self._extract_patches(image, self.patch_size, self.stride)
        B, num_patches, C, ph, pw = patches.shape

        # Get patch predictions
        with torch.no_grad():
            patches = patches.to(device)
            logits = model(patches.view(-1, C, ph, pw))
            probs = F.softmax(logits, dim=1)

        # Get fake probability for each patch
        fake_probs = probs[:, 1].view(B, num_patches).cpu().numpy()

        # Reshape to grid
        grid_h, grid_w = self._get_grid_shape(h, w)
        heatmap = fake_probs.reshape(grid_h, grid_w)

        return {
            "heatmap": heatmap,
            "grid_shape": (grid_h, grid_w),
            "patch_positions": self._get_patch_positions(grid_h, grid_w),
        }

    def visualize_heatmap(
        self, image_path, heatmap, output_path=None, colormap="RdYlBu_r"
    ):
        """
        Create visualization of heatmap overlaid on image.

        Args:
            image_path: Path to original image
            heatmap: Heatmap array (grid_h, grid_w)
            output_path: Save visualization to this path
            colormap: Matplotlib colormap

        Returns:
            PIL Image of visualization
        """
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Heatmap
        im = axes[1].imshow(heatmap, cmap=colormap, alpha=0.8)
        axes[1].imshow(image, alpha=0.3)
        axes[1].set_title("Deepfake Probability Heatmap")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], label="Fake Probability")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        # Convert to PIL Image
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        result = Image.fromarray(image_array)

        plt.close(fig)
        return result

    def _extract_patches(self, image, patch_size, stride):
        """Extract patches from image in grid pattern."""
        # Placeholder implementation
        pass

    def _get_grid_shape(self, h, w):
        """Calculate grid shape for given image size."""
        num_h = (h - self.patch_size) // self.stride + 1
        num_w = (w - self.patch_size) // self.stride + 1
        return num_h, num_w

    def _get_patch_positions(self, grid_h, grid_w):
        """Get (top-left) positions of all patches."""
        positions = []
        for i in range(grid_h):
            for j in range(grid_w):
                positions.append((i * self.stride, j * self.stride))
        return positions
