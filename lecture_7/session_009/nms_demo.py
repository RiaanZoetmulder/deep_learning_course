"""Visualize Non-Maximum Suppression: before vs. after side-by-side plot."""

from typing import Callable, List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def _draw_boxes(
    ax: plt.Axes,
    boxes: np.ndarray,
    scores: np.ndarray,
    indices: List[int],
    colors: tuple,
    filled: bool = False,
) -> None:
    """Draw bounding boxes on *ax* for the given *indices*."""
    for idx in indices:
        x1, y1, x2, y2 = boxes[idx]
        w, h = x2 - x1, y2 - y1
        colour = colors[idx % len(colors)]

        if filled:
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1), w, h,
                    linewidth=3, edgecolor=colour,
                    facecolor=colour, alpha=0.3,
                )
            )
            # Solid border on top of the fill
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1), w, h,
                    linewidth=3, edgecolor=colour,
                    facecolor="none",
                )
            )
        else:
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1), w, h,
                    linewidth=2, edgecolor=colour,
                    facecolor="none", linestyle="-",
                )
            )

        ax.text(
            x1 + 5, y1 + 15,
            f"{scores[idx]:.2f}",
            fontsize=10, color=colour, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )


def _setup_axis(ax: plt.Axes, title: str, xlim: int = 400, ylim: int = 400) -> None:
    """Configure a detection-style axis (inverted y, equal aspect, grey bg)."""
    ax.set_xlim(0, xlim)
    ax.set_ylim(ylim, 0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)
    ax.set_facecolor("lightgray")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def visualize_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    nms_fn: Callable[[np.ndarray, np.ndarray, float], List[int]],
) -> None:
    """Show a before / after NMS comparison.

    Parameters
    ----------
    boxes : (N, 4) array of [x_min, y_min, x_max, y_max]
    scores : (N,) confidence scores
    iou_threshold : IoU threshold used for suppression
    nms_fn : the student's NMS implementation (same signature as
             ``non_maximum_suppression(boxes, scores, iou_threshold)``)
    """
    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10.colors

    # --- Left: all boxes ---
    _setup_axis(ax_before, "Before NMS (All predicted boxes)")
    _draw_boxes(ax_before, boxes, scores, list(range(len(boxes))), colors, filled=False)

    # --- Right: kept boxes ---
    kept_indices = nms_fn(boxes, scores, iou_threshold)
    _setup_axis(ax_after, f"After NMS (IoU threshold = {iou_threshold})")
    _draw_boxes(ax_after, boxes, scores, kept_indices, colors, filled=True)

    plt.suptitle(
        "Non-Maximum Suppression (NMS) Demonstration",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    print(f"Original boxes: {len(boxes)}")
    print(f"Boxes after NMS: {len(kept_indices)}")
    print(f"Kept indices: {kept_indices}")
