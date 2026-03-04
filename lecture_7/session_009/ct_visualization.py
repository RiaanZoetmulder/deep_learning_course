"""
CT Scan Visualization Utilities for Medical Image Segmentation Tutorial.

This module provides functions for loading DICOM series, applying different
normalization methods, and visualizing the comparison between per-slice
and per-volume normalization approaches.
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pydicom as pdm
import torch
import ipywidgets as widgets


def load_dicom_series(
    dicom_series_path: Path,
    hu_min: float = -1024,
    hu_max: float = 3071,
) -> torch.Tensor:
    """
    Load a DICOM series and convert to Hounsfield Units.
    
    Args:
        dicom_series_path: Path to folder containing .dcm files
        hu_min: Minimum HU value for clipping (default: -1024, air)
        hu_max: Maximum HU value for clipping (default: 3071, dense materials)
    
    Returns:
        torch.Tensor of shape (num_slices, H, W) with HU values clipped to [hu_min, hu_max]
    
    Raises:
        FileNotFoundError: If the DICOM series path does not exist
    """
    if not dicom_series_path.exists():
        print("=" * 60)
        print("CT DICOM Data Not Found!")
        print("=" * 60)
        print(f"\nExpected path: {dicom_series_path.absolute()}")
        print("\nTo run this demonstration, you need CT DICOM data.")
        print("Get an open source dicom image, store it in data, and update the path above.")
        raise FileNotFoundError("Please download CT DICOM data first")
    
    # Load the DICOM series
    dicom_files = sorted(dicom_series_path.glob("*.dcm"))
    print(f"Found {len(dicom_files)} DICOM files in {dicom_series_path}")
    
    # Load all slices and stack them
    slices = []
    for dcm_file in dicom_files:
        ds = pdm.dcmread(dcm_file)
        # Convert to Hounsfield Units (HU) if possible
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            hu_image = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        else:
            hu_image = ds.pixel_array.astype(np.float32)
        slices.append(hu_image)
    
    # Stack into a 3D volume
    dicom_stack = torch.tensor(np.stack(slices, axis=0), dtype=torch.float32)
    
    # Clip to valid HU range (values below -1024 are padding)
    dicom_stack = torch.clamp(dicom_stack, min=hu_min, max=hu_max)
    
    return dicom_stack


def normalize_zscore(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Z-score normalization: (x - mean) / std
    
    Args:
        data: Input array
        mean: Mean value for normalization
        std: Standard deviation for normalization
    
    Returns:
        Normalized array
    """
    return (data - mean) / (std + 1e-8)


def normalize_minmax(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Min-max normalization: (x - min) / (max - min) -> [0, 1]
    
    Args:
        data: Input array
        min_val: Minimum value for normalization
        max_val: Maximum value for normalization
    
    Returns:
        Normalized array scaled to [0, 1]
    """
    return (data - min_val) / (max_val - min_val + 1e-8)


def show_normalization_comparison(
    slice_idx: int,
    ct_volume: np.ndarray,
    volume_mean: float,
    volume_std: float,
    volume_min: float,
    volume_max: float,
) -> None:
    """
    Display comparison of per-slice vs per-volume normalization for a CT slice.
    
    Creates a figure with:
    - Row 1: Raw CT, Z-score (per-slice), Z-score (per-volume)
    - Row 2: Stacked histograms, Min-max (per-slice), Min-max (per-volume)
    
    Args:
        slice_idx: Index of the slice to display
        ct_volume: 3D numpy array of shape (num_slices, H, W)
        volume_mean: Mean of the entire volume
        volume_std: Standard deviation of the entire volume
        volume_min: Minimum value of the entire volume
        volume_max: Maximum value of the entire volume
    """
    slice_data = ct_volume[slice_idx]
    
    # Per-slice statistics
    slice_mean = slice_data.mean()
    slice_std = slice_data.std()
    slice_min = slice_data.min()
    slice_max = slice_data.max()
    
    # Apply normalizations
    zscore_per_slice = normalize_zscore(slice_data, slice_mean, slice_std)
    zscore_per_volume = normalize_zscore(slice_data, volume_mean, volume_std)
    minmax_per_slice = normalize_minmax(slice_data, slice_min, slice_max)
    minmax_per_volume = normalize_minmax(slice_data, volume_min, volume_max)
    
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(12, 8))
    
    # GridSpec: 4 rows (2 for each image row), 3 columns
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 1],
                  hspace=0.3, wspace=0.25)
    
    # Row 1: Raw and Z-score comparisons (span rows 0-1)
    ax_raw = fig.add_subplot(gs[0:2, 0])
    ax_zscore_slice = fig.add_subplot(gs[0:2, 1])
    ax_zscore_vol = fig.add_subplot(gs[0:2, 2])
    
    # Row 2: Stacked histograms (left), Min-max images (middle, right)
    ax_hist_zscore = fig.add_subplot(gs[2, 0])
    ax_hist_minmax = fig.add_subplot(gs[3, 0])
    ax_minmax_slice = fig.add_subplot(gs[2:4, 1])
    ax_minmax_vol = fig.add_subplot(gs[2:4, 2])
    
    # === Row 1: Raw and Z-score images ===
    im0 = ax_raw.imshow(slice_data, cmap='gray', aspect='equal')
    ax_raw.set_title(f'Raw CT Slice {slice_idx}\n(HU: {slice_min:.0f} to {slice_max:.0f})', fontsize=10)
    plt.colorbar(im0, ax=ax_raw, fraction=0.046, pad=0.02)
    ax_raw.axis('off')
    
    im1 = ax_zscore_slice.imshow(zscore_per_slice, cmap='gray', aspect='equal')
    ax_zscore_slice.set_title(f'Z-score (Per-Slice)\nμ={slice_mean:.1f}, σ={slice_std:.1f}', fontsize=10)
    plt.colorbar(im1, ax=ax_zscore_slice, fraction=0.046, pad=0.02)
    ax_zscore_slice.axis('off')
    
    im2 = ax_zscore_vol.imshow(zscore_per_volume, cmap='gray', aspect='equal')
    ax_zscore_vol.set_title(f'Z-score (Per-Volume)\nμ={volume_mean:.1f}, σ={volume_std:.1f}', fontsize=10)
    plt.colorbar(im2, ax=ax_zscore_vol, fraction=0.046, pad=0.02)
    ax_zscore_vol.axis('off')
    
    # === Row 2 Left: Stacked histograms comparing normalization methods ===
    ax_hist_zscore.hist(zscore_per_slice.flatten(), bins=80, alpha=0.7, 
                        label='Per-Slice', color='tab:blue', density=True)
    ax_hist_zscore.hist(zscore_per_volume.flatten(), bins=80, alpha=0.6, 
                        label='Per-Volume', color='tab:orange', density=True)
    ax_hist_zscore.set_title('Z-score Normalized', fontsize=9)
    ax_hist_zscore.legend(fontsize=7, loc='upper right')
    ax_hist_zscore.set_ylabel('Density', fontsize=8)
    ax_hist_zscore.tick_params(labelsize=7)
    
    ax_hist_minmax.hist(minmax_per_slice.flatten(), bins=80, alpha=0.7, 
                        label='Per-Slice', color='tab:blue', density=True)
    ax_hist_minmax.hist(minmax_per_volume.flatten(), bins=80, alpha=0.6, 
                        label='Per-Volume', color='tab:orange', density=True)
    ax_hist_minmax.set_title('Min-Max Normalized', fontsize=9)
    ax_hist_minmax.legend(fontsize=7, loc='upper right')
    ax_hist_minmax.set_xlabel('Normalized Value', fontsize=8)
    ax_hist_minmax.set_ylabel('Density', fontsize=8)
    ax_hist_minmax.tick_params(labelsize=7)
    
    # === Row 2: Min-max images ===
    im3 = ax_minmax_slice.imshow(minmax_per_slice, cmap='gray', vmin=0, vmax=1, aspect='equal')
    ax_minmax_slice.set_title(f'Min-Max (Per-Slice)\nrange=[{slice_min:.0f}, {slice_max:.0f}]', fontsize=10)
    plt.colorbar(im3, ax=ax_minmax_slice, fraction=0.046, pad=0.02)
    ax_minmax_slice.axis('off')
    
    im4 = ax_minmax_vol.imshow(minmax_per_volume, cmap='gray', vmin=0, vmax=1, aspect='equal')
    ax_minmax_vol.set_title(f'Min-Max (Per-Volume)\nrange=[{volume_min:.0f}, {volume_max:.0f}]', fontsize=10)
    plt.colorbar(im4, ax=ax_minmax_vol, fraction=0.046, pad=0.02)
    ax_minmax_vol.axis('off')
    
    plt.show()
    
    # Print statistics comparison
    print(f"\nSlice {slice_idx} Statistics Comparison:")
    print(f"{'Metric':<20} {'Per-Slice':<15} {'Per-Volume':<15}")
    print("-" * 50)
    print(f"{'Z-score range':<20} [{zscore_per_slice.min():.2f}, {zscore_per_slice.max():.2f}]   [{zscore_per_volume.min():.2f}, {zscore_per_volume.max():.2f}]")
    print(f"{'Min-max range':<20} [0.00, 1.00]        [{minmax_per_volume.min():.2f}, {minmax_per_volume.max():.2f}]")


def create_normalization_slider(ct_volume: np.ndarray) -> widgets.interactive:
    """
    Create an interactive slider widget for exploring normalization across CT slices.
    
    Args:
        ct_volume: 3D numpy array of shape (num_slices, H, W)
    
    Returns:
        ipywidgets interactive widget
    """
    # Compute volume-level statistics
    volume_mean = ct_volume.mean()
    volume_std = ct_volume.std()
    volume_min = ct_volume.min()
    volume_max = ct_volume.max()
    
    def _show_slice(slice_idx):
        show_normalization_comparison(
            slice_idx, ct_volume, volume_mean, volume_std, volume_min, volume_max
        )
    
    return widgets.interact(
        _show_slice,
        slice_idx=widgets.IntSlider(
            value=ct_volume.shape[0] // 2,
            min=0,
            max=ct_volume.shape[0] - 1,
            step=1,
            description='Slice:',
            continuous_update=False
        )
    )
