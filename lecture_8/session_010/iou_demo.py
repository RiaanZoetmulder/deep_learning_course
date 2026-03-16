"""Interactive IoU demo with a draggable predicted box and detection threshold.

Uses ipywidgets sliders so it works in VS Code, JupyterLab, and Colab
without requiring the ipympl (widget) matplotlib backend.

Copied from session_009/iou_demo.py for reuse in the evaluation tutorial.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import display, clear_output
import ipywidgets as widgets


def _compute_iou(box_a, box_b):
    """Compute IoU between two (x_min, y_min, x_max, y_max) boxes."""
    xi1 = max(box_a[0], box_b[0])
    yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2])
    yi2 = min(box_a[3], box_b[3])
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def _draw_iou_frame(pred_x, pred_y, iou_threshold):
    """Draw a single frame of the IoU demo for the given slider values."""

    # --- Ground truth box (fixed) ---
    gt_box = (20, 20, 70, 70)

    # --- Predicted box ---
    pred_w, pred_h = 50, 50
    pred_box = (pred_x, pred_y, pred_x + pred_w, pred_y + pred_h)

    iou = _compute_iou(gt_box, pred_box)
    detected = iou >= iou_threshold
    colour = '#2ca02c' if detected else '#d62728'

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.set_title('IoU Detection Demo', fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Ground-truth box (blue, solid)
    gt_rect = mpatches.FancyBboxPatch(
        (gt_box[0], gt_box[1]),
        gt_box[2] - gt_box[0],
        gt_box[3] - gt_box[1],
        boxstyle="square,pad=0",
        linewidth=2.5,
        edgecolor='#1f77b4',
        facecolor='none',
        label='Ground Truth',
        zorder=3,
    )
    ax.add_patch(gt_rect)

    # Predicted box (green/red, dashed)
    pred_rect = mpatches.FancyBboxPatch(
        (pred_x, pred_y),
        pred_w,
        pred_h,
        boxstyle="square,pad=0",
        linewidth=2.5,
        edgecolor=colour,
        facecolor='none',
        linestyle='--',
        label='Predicted',
        zorder=3,
    )
    ax.add_patch(pred_rect)

    # Overlap shading
    ox1 = max(gt_box[0], pred_box[0])
    oy1 = max(gt_box[1], pred_box[1])
    ox2 = min(gt_box[2], pred_box[2])
    oy2 = min(gt_box[3], pred_box[3])
    ow = max(0, ox2 - ox1)
    oh = max(0, oy2 - oy1)
    if ow > 0 and oh > 0:
        overlap_rect = mpatches.Rectangle(
            (ox1, oy1), ow, oh,
            linewidth=0, facecolor=colour, alpha=0.35, zorder=2,
        )
        ax.add_patch(overlap_rect)

    # IoU / threshold text
    ax.text(
        2, 97, f'IoU = {iou:.3f}   (threshold = {iou_threshold:.2f})',
        fontsize=12, fontfamily='monospace', verticalalignment='top', zorder=5,
    )

    # Detected / NOT Detected status
    if detected:
        ax.text(2, 91, 'Detected \u2713', fontsize=14, fontweight='bold',
                color='#2ca02c', verticalalignment='top', zorder=5)
    else:
        ax.text(2, 91, 'NOT Detected \u2717', fontsize=14, fontweight='bold',
                color='#d62728', verticalalignment='top', zorder=5)

    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.show()


def iou_detection_demo():
    """
    Launch an interactive IoU demo using ipywidgets sliders.

    * Blue box  = ground truth (fixed).
    * Red/Green box = predicted (moveable via sliders).
    * Green/Red shaded region = overlap area.
    * IoU value, threshold, and Detected / NOT Detected status shown in the
      top-left corner.
    """

    slider_x = widgets.IntSlider(
        value=40, min=0, max=50, step=1,
        description='Pred X:', continuous_update=True,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='450px'),
    )
    slider_y = widgets.IntSlider(
        value=40, min=0, max=50, step=1,
        description='Pred Y:', continuous_update=True,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='450px'),
    )
    slider_t = widgets.FloatSlider(
        value=0.5, min=0.0, max=1.0, step=0.05,
        description='IoU Threshold:', continuous_update=True,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='450px'),
        readout_format='.2f',
    )

    output = widgets.Output()

    def _on_change(_=None):
        with output:
            clear_output(wait=True)
            _draw_iou_frame(slider_x.value, slider_y.value, slider_t.value)

    slider_x.observe(_on_change, names='value')
    slider_y.observe(_on_change, names='value')
    slider_t.observe(_on_change, names='value')

    # Initial render
    _on_change()

    display(widgets.VBox([slider_x, slider_y, slider_t, output]))
    plt.show()
