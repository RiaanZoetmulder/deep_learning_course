"""
Step-through demo of Anchor Boxes → Classification → Regression → NMS.

Uses ipywidgets so it works in VS Code, JupyterLab, and Colab without
requiring the ipympl backend.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import display, clear_output
import ipywidgets as widgets


# ──────────────────────────── helpers ────────────────────────────

def _compute_iou(box_a, box_b):
    """IoU between two (x1, y1, x2, y2) boxes."""
    xi1 = max(box_a[0], box_b[0])
    yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2])
    yi2 = min(box_a[3], box_b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _gt_box_from_mask(voc_demo_dir, image_id):
    """
    Compute the ground-truth bounding box from the VOC segmentation mask.

    Uses the same VOC-colormap approach as the notebook's main cell
    so the result is guaranteed to match.
    """
    mask_path = os.path.join(voc_demo_dir, f"{image_id}_mask.png")
    mask_rgb = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

    VOC_COLORMAP = np.array([
        [0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],
        [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
        [64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],
        [192,128,128],[0,64,0],[128,64,0],[0,192,0],[128,192,0],[0,64,128]
    ], dtype=np.uint8)

    class_map = np.full(mask_rgb.shape[:2], fill_value=255, dtype=np.uint8)
    for idx, colour in enumerate(VOC_COLORMAP):
        match = np.all(mask_rgb == colour, axis=-1)
        class_map[match] = idx

    # Foreground = any valid class except background (0) and void (255)
    fg_mask = ((class_map > 0) & (class_map < 255)).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(fg_mask)
    return (x, y, x + w, y + h)


# FPN level colours used in step-2 rendering
_LEVEL_COLORS = ['#ffd700', '#ff8c00', '#dc143c']   # gold, dark-orange, crimson
_LEVEL_LABELS = ['Small (fine grid)', 'Medium', 'Large (coarse grid)']


def _generate_fpn_anchors(img_h, img_w):
    """
    Generate anchor boxes mimicking a Feature Pyramid Network (FPN).

    Each pyramid level uses a *different* grid stride and anchor size so
    that small anchors are placed densely and large anchors sparsely.
    This avoids the "cluster-of-boxes repeated on a grid" pattern that a
    single-level configuration produces.

    Returns
    -------
    anchors : list of (x1, y1, x2, y2)
    levels  : list of int  (pyramid level index for each anchor)
    """
    # (grid_rows, grid_cols, base_size_px, aspect_ratios)
    level_configs = [
        (6, 8,  30, [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]),   # fine:   small anchors
        (3, 4,  65, [(1.0, 1.0), (1.5, 0.65), (0.65, 1.5)]),  # medium
        (2, 3, 120, [(1.0, 1.0), (1.7, 0.6), (0.6, 1.7)]),    # coarse: large anchors
    ]

    anchors = []
    levels = []
    for lvl, (grid_rows, grid_cols, base, ratios) in enumerate(level_configs):
        cell_h = img_h / grid_rows
        cell_w = img_w / grid_cols
        for r in range(grid_rows):
            cy = cell_h * (r + 0.5)
            for c in range(grid_cols):
                cx = cell_w * (c + 0.5)
                for (ar_w, ar_h) in ratios:
                    w = base * ar_w
                    h = base * ar_h
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    anchors.append((x1, y1, x2, y2))
                    levels.append(lvl)
    return anchors, levels


# ──────────────────────────── drawing ────────────────────────────

def _draw_boxes(ax, boxes, color, linewidth=1.5, linestyle='-', label=None, alpha=1.0):
    """Draw a list of (x1, y1, x2, y2) boxes on *ax*."""
    first = True
    for (x1, y1, x2, y2) in boxes:
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=linewidth, edgecolor=color,
            facecolor='none', linestyle=linestyle,
            label=label if first else None,
            alpha=alpha,
        )
        ax.add_patch(rect)
        first = False


def _draw_arrow(ax, from_box, to_box, color='#ff7f0e'):
    """Draw a straight arrow from the centre of *from_box* to *to_box*."""
    fx = (from_box[0] + from_box[2]) / 2
    fy = (from_box[1] + from_box[3]) / 2
    tx = (to_box[0] + to_box[2]) / 2
    ty = (to_box[1] + to_box[3]) / 2
    ax.annotate(
        '', xy=(tx, ty), xytext=(fx, fy),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
    )


# ──────────────────────────── frame renderers ────────────────────

STEP_TITLES = [
    "Step 1 — Original Image + Ground Truth",
    "Step 2 — Anchor Boxes ",
    "Step 3 — Classify: keep anchors that overlap the object \n(we only keep a subset now for visual clarity)",
    "Step 4 — Regress: refine anchors toward ground truth",
    "Step 5 — Non-Maximum Suppression: \nkeep the box with the highest confidence",
]


def _render_frame(step, image, gt_box, anchors, anchor_levels,
                  pos_anchors, regressed, kept_idx):
    """Render a single step of the demo. Returns the matplotlib figure."""

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.imshow(image)
    ax.set_title(STEP_TITLES[step], fontsize=13, fontweight='bold')
    ax.axis('off')

    if step == 0:
        # Just the image with GT box
        _draw_boxes(ax, [gt_box], color='#1f77b4', linewidth=2.5, label='Ground Truth')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.85)

    elif step == 1:
        # Anchors colour-coded by FPN pyramid level
        n_levels = max(anchor_levels) + 1 if anchor_levels else 1
        for lvl in range(n_levels):
            lvl_anchors = [a for a, l in zip(anchors, anchor_levels) if l == lvl]
            if lvl_anchors:
                col = _LEVEL_COLORS[lvl % len(_LEVEL_COLORS)]
                lab = _LEVEL_LABELS[lvl % len(_LEVEL_LABELS)]
                _draw_boxes(ax, lvl_anchors, color=col, linewidth=0.6,
                            linestyle='--',
                            label=f'{lab} ({len(lvl_anchors)})',
                            alpha=0.55)
        _draw_boxes(ax, [gt_box], color='#1f77b4', linewidth=2.5, label='Ground Truth')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.85)

    elif step == 2:
        # Positive anchors in green, rest removed
        _draw_boxes(ax, pos_anchors, color='#2ca02c', linewidth=2,
                    label=f'Positive Anchors ({len(pos_anchors)})')
        _draw_boxes(ax, [gt_box], color='#1f77b4', linewidth=2.5, label='Ground Truth')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.85)

        # Annotate IoU of each positive anchor
        for anc in pos_anchors:
            iou = _compute_iou(anc, gt_box)
            cx = (anc[0] + anc[2]) / 2
            cy = anc[1] - 4
            ax.text(cx, cy, f'IoU={iou:.2f}', fontsize=7, color='lime',
                    ha='center', va='bottom', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.6, pad=1))

    elif step == 3:
        # Show regression: anchor (green dashed) → refined (orange solid)
        # Arrows point from the refined box back to the anchor it came from,
        # visually linking each refined prediction to its origin.
        _draw_boxes(ax, pos_anchors, color='#2ca02c', linewidth=1.2, linestyle='--',
                    label='Positive Anchors (before)', alpha=0.5)
        _draw_boxes(ax, regressed, color='#ff7f0e', linewidth=2,
                    label='Refined Boxes (after)')
        _draw_boxes(ax, [gt_box], color='#1f77b4', linewidth=2.5, label='Ground Truth')

        for anc, reg in zip(pos_anchors, regressed):
            _draw_arrow(ax, from_box=reg, to_box=anc, color='#ff7f0e')

        # Annotate dx, dy, dw, dh for the first anchor as an example
        if pos_anchors:
            a = pos_anchors[0]
            r = regressed[0]
            aw = a[2] - a[0]; ah = a[3] - a[1]
            acx = (a[0] + a[2]) / 2; acy = (a[1] + a[3]) / 2
            rcx = (r[0] + r[2]) / 2; rcy = (r[1] + r[3]) / 2
            rw = r[2] - r[0]; rh = r[3] - r[1]
            dx = (rcx - acx) / aw if aw > 0 else 0
            dy = (rcy - acy) / ah if ah > 0 else 0
            dw = np.log(rw / aw) if aw > 0 and rw > 0 else 0
            dh = np.log(rh / ah) if ah > 0 and rh > 0 else 0
            info = (f'Example offsets (first anchor):\n'
                    f'dx={dx:+.2f}  dy={dy:+.2f}\n'
                    f'dw={dw:+.2f}  dh={dh:+.2f}')
            ax.text(0.99, 0.02, info, transform=ax.transAxes, fontsize=9,
                    fontfamily='monospace', va='bottom', ha='right',
                    bbox=dict(facecolor='white', alpha=0.85, pad=4))

        ax.legend(loc='upper left', fontsize=10, framealpha=0.85)

    elif step == 4:
        # NMS result: show all regressed dim, winning box bright
        eliminated = [b for i, b in enumerate(regressed) if i != kept_idx]
        winner = regressed[kept_idx]

        _draw_boxes(ax, eliminated, color='#d62728', linewidth=1.5,
                    linestyle='--', label='Suppressed', alpha=0.45)
        _draw_boxes(ax, [winner], color='#2ca02c', linewidth=3,
                    label='Kept (highest IoU)')
        _draw_boxes(ax, [gt_box], color='#1f77b4', linewidth=2.5, label='Ground Truth')

        iou_winner = _compute_iou(winner, gt_box)
        ax.text(
            (winner[0] + winner[2]) / 2, winner[1] - 6,
            f'IoU = {iou_winner:.3f}', fontsize=10, color='#2ca02c',
            ha='center', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.85, pad=2))

        ax.legend(loc='upper left', fontsize=10, framealpha=0.85)

    plt.tight_layout()
    return fig


# ──────────────────────────── public API ─────────────────────────

def anchor_box_demo(voc_demo_dir: str = None, gt_box=None):
    """
    Launch the interactive step-through demo.

    Parameters
    ----------
    voc_demo_dir : str, optional
        Path to the VOC demo directory containing the train image.
        Defaults to ``009_object_detection/voc_demo``.
    gt_box : tuple of int, optional
        Ground-truth box as (x1, y1, x2, y2).  When *None* (default),
        the box is computed from the segmentation mask.
    """

    if voc_demo_dir is None:
        voc_demo_dir = os.path.join("009_object_detection", "voc_demo")

    image_id = "2007_007168"
    img_path = os.path.join(voc_demo_dir, f"{image_id}.jpg")
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]

    # Ground-truth bounding box — compute from mask if not provided
    if gt_box is None:
        gt_box = _gt_box_from_mask(voc_demo_dir, image_id)

    # ---- Generate FPN-style anchors (different grid per scale) ----
    anchors, anchor_levels = _generate_fpn_anchors(img_h, img_w)

    # ---- Step 2: classify — keep anchors with IoU > threshold to the GT ----
    iou_pos_threshold = 0.35
    pos_anchors = [a for a in anchors if _compute_iou(a, gt_box) > iou_pos_threshold]
    # If somehow none, take top-5 by IoU
    if len(pos_anchors) == 0:
        scored = sorted(anchors, key=lambda a: _compute_iou(a, gt_box), reverse=True)
        pos_anchors = scored[:5]

    # ---- Step 3: regress each positive anchor toward the GT ----
    # We simulate the "learned" regression by interpolating 80 % toward
    # the ground truth, which is realistic for a trained detector.
    mix = 0.80
    regressed = []
    for a in pos_anchors:
        rx1 = a[0] + mix * (gt_box[0] - a[0])
        ry1 = a[1] + mix * (gt_box[1] - a[1])
        rx2 = a[2] + mix * (gt_box[2] - a[2])
        ry2 = a[3] + mix * (gt_box[3] - a[3])
        regressed.append((rx1, ry1, rx2, ry2))

    # ---- Step 4: NMS — pick box with highest IoU to GT ----
    ious = [_compute_iou(r, gt_box) for r in regressed]
    kept_idx = int(np.argmax(ious))

    # ---- Widget ----
    step_slider = widgets.IntSlider(
        value=0, min=0, max=4, step=1,
        description='Step:',
        continuous_update=True,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px'),
    )

    step_label = widgets.HTML(value=f'<b>{STEP_TITLES[0]}</b>')

    output = widgets.Output()

    def _on_change(_=None):
        s = step_slider.value
        step_label.value = f'<b>{STEP_TITLES[s]}</b>'
        with output:
            clear_output(wait=True)
            fig = _render_frame(s, image, gt_box, anchors, anchor_levels,
                                pos_anchors, regressed, kept_idx)
            plt.show()

    step_slider.observe(_on_change, names='value')

    # Initial render
    _on_change()

    display(widgets.VBox([step_slider, step_label, output]))
