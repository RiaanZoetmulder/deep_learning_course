"""Interactive binary segmentation demo with circular masks.

Provides five visualizations:
1. ``create_binary_seg_demo``  — TP / FP / FN / TN overlay with sliders
   for ground-truth and prediction circle position and diameter.
2. ``create_metrics_dashboard`` — same slider layout, but the right panel
   shows all CV segmentation metrics computed in real time.
3. ``create_boundary_demo``    — boundary extraction with sampled
   point-to-boundary distance arrows and a distance table.
4. ``create_boundary_metrics_dashboard`` — overlay + boundaries with
   both overlap (DSC, IoU, Precision, Recall) and boundary
   (HD, HD95, ASSD, NSD) metrics, plus protrusion control and presets.
5. ``create_volume_metrics_dashboard`` — overlay + metric table showing
   volume metrics (AVD, RVD) alongside overlap (DSC, IoU) and boundary
   (HD95, ASSD) metrics, with failure-case presets including a
   *fragmented prediction* scenario.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ipywidgets as widgets
from IPython.display import clear_output, display
from matplotlib.patheffects import withStroke
from scipy.ndimage import binary_erosion, distance_transform_edt


# ──────────────────────────────────────────────────────────────
#  Colour constants
# ──────────────────────────────────────────────────────────────
TP_COLOR = np.array([0.18, 0.80, 0.25])   # green
FP_COLOR = np.array([0.90, 0.15, 0.15])   # red
FN_COLOR = np.array([1.00, 0.85, 0.10])   # yellow
TN_COLOR = np.array([0.25, 0.45, 0.85])   # blue


# ──────────────────────────────────────────────────────────────
#  Helper: build circular binary mask on a square grid
# ──────────────────────────────────────────────────────────────
def _circular_mask(grid_size: int, cx: int, cy: int, diameter: int) -> np.ndarray:
    """Return a boolean mask with a filled circle."""
    yy, xx = np.mgrid[0:grid_size, 0:grid_size]
    r = diameter / 2.0
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2


# ──────────────────────────────────────────────────────────────
#  Helper: paint TP / FP / FN / TN into an RGB image
# ──────────────────────────────────────────────────────────────
def _build_overlay(gt_mask: np.ndarray, pred_mask: np.ndarray) -> tuple:
    """Return (rgb_image, tp_count, fp_count, fn_count, tn_count)."""
    tp_map = pred_mask & gt_mask
    fp_map = pred_mask & ~gt_mask
    fn_map = ~pred_mask & gt_mask
    tn_map = ~pred_mask & ~gt_mask

    h, w = gt_mask.shape
    img = np.zeros((h, w, 3), dtype=np.float64)
    img[tp_map] = TP_COLOR
    img[fp_map] = FP_COLOR
    img[fn_map] = FN_COLOR
    img[tn_map] = TN_COLOR

    return img, int(tp_map.sum()), int(fp_map.sum()), int(fn_map.sum()), int(tn_map.sum())


# ──────────────────────────────────────────────────────────────
#  Helper: extract 1-pixel-wide boundary from a binary mask
# ──────────────────────────────────────────────────────────────
def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    """Return the 1-pixel-wide boundary of *mask*.

    boundary = mask AND NOT erode(mask).  A foreground pixel is on
    the boundary if at least one of its 4-connected neighbours is
    background.
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask)
    return mask & np.logical_not(eroded)


# ──────────────────────────────────────────────────────────────
#  Helper: standard legend patches
# ──────────────────────────────────────────────────────────────
def _legend_patches(tp: int, fp: int, fn: int, tn: int) -> list:
    return [
        mpatches.Patch(color=TP_COLOR, label=f"TP = {tp:,}"),
        mpatches.Patch(color=FP_COLOR, label=f"FP = {fp:,}"),
        mpatches.Patch(color=FN_COLOR, label=f"FN = {fn:,}"),
        mpatches.Patch(color=TN_COLOR, label=f"TN = {tn:,}"),
    ]


# ──────────────────────────────────────────────────────────────
#  Helper: build the six sliders shared by both demos
# ──────────────────────────────────────────────────────────────
def _make_sliders(grid_size: int) -> dict:
    half = grid_size // 2
    style = {"description_width": "110px"}
    layout = widgets.Layout(width="380px")

    return dict(
        gt_cx=widgets.IntSlider(
            value=half, min=0, max=grid_size - 1, step=1,
            description="GT centre X:", style=style, layout=layout,
        ),
        gt_cy=widgets.IntSlider(
            value=half, min=0, max=grid_size - 1, step=1,
            description="GT centre Y:", style=style, layout=layout,
        ),
        gt_diameter=widgets.IntSlider(
            value=60, min=2, max=grid_size, step=1,
            description="GT diameter:", style=style, layout=layout,
        ),
        pred_cx=widgets.IntSlider(
            value=half + 15, min=0, max=grid_size - 1, step=1,
            description="Pred centre X:", style=style, layout=layout,
        ),
        pred_cy=widgets.IntSlider(
            value=half + 15, min=0, max=grid_size - 1, step=1,
            description="Pred centre Y:", style=style, layout=layout,
        ),
        pred_diameter=widgets.IntSlider(
            value=55, min=2, max=grid_size, step=1,
            description="Pred diameter:", style=style, layout=layout,
        ),
    )


# ══════════════════════════════════════════════════════════════
#  VISUALISATION 1 — Binary Segmentation Confusion Matrix Demo
# ══════════════════════════════════════════════════════════════
def create_binary_seg_demo(grid_size: int = 128) -> None:
    """Show an interactive binary segmentation overlay (TP/FP/FN/TN).

    Two circular masks (ground truth and prediction) are controlled by
    sliders for centre position and diameter. The overlay is coloured:
    green = TP, red = FP, yellow = FN, blue = TN.
    """
    sliders = _make_sliders(grid_size)
    out = widgets.Output()

    def _draw(**kwargs):
        gt_mask = _circular_mask(grid_size, kwargs["gt_cx"], kwargs["gt_cy"], kwargs["gt_diameter"])
        pred_mask = _circular_mask(grid_size, kwargs["pred_cx"], kwargs["pred_cy"], kwargs["pred_diameter"])
        img, tp, fp, fn, tn = _build_overlay(gt_mask, pred_mask)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, interpolation="nearest")
        ax.set_title("Binary Segmentation: TP / FP / FN / TN", fontsize=14, fontweight="bold")
        ax.axis("off")
        ax.legend(handles=_legend_patches(tp, fp, fn, tn),
                  loc="lower right", fontsize=11, framealpha=0.9)
        plt.tight_layout()
        plt.show()

    def _on_change(_=None):
        with out:
            clear_output(wait=True)
            _draw(**{k: v.value for k, v in sliders.items()})

    for s in sliders.values():
        s.observe(_on_change, names="value")

    # Initial render
    with out:
        _draw(**{k: v.value for k, v in sliders.items()})

    ui_gt = widgets.VBox([sliders["gt_cx"], sliders["gt_cy"], sliders["gt_diameter"]],
                         layout=widgets.Layout(margin="0 20px 0 0"))
    ui_pred = widgets.VBox([sliders["pred_cx"], sliders["pred_cy"], sliders["pred_diameter"]])
    ui = widgets.HBox([ui_gt, ui_pred])
    display(widgets.VBox([ui, out]))


# ══════════════════════════════════════════════════════════════
#  Failure-case presets (grid_size = 128)
# ══════════════════════════════════════════════════════════════
_FAILURE_CASES = {
    "Case 1: Accuracy Paradox": dict(
        gt_cx=64, gt_cy=64, gt_diameter=20,
        pred_cx=0, pred_cy=0, pred_diameter=2,
    ),
    "Case 2: Over-Segmentation": dict(
        gt_cx=64, gt_cy=64, gt_diameter=30,
        pred_cx=64, pred_cy=64, pred_diameter=100,
    ),
    "Case 3: Under-Segmentation": dict(
        gt_cx=64, gt_cy=64, gt_diameter=100,
        pred_cx=64, pred_cy=64, pred_diameter=30,
    ),
    "Case 4: Complete Miss": dict(
        gt_cx=30, gt_cy=30, gt_diameter=40,
        pred_cx=100, pred_cy=100, pred_diameter=40,
    ),
    "Case 5: Small Shift, Big IoU Drop": dict(
        gt_cx=64, gt_cy=64, gt_diameter=30,
        pred_cx=80, pred_cy=80, pred_diameter=30,
    ),
    "Case 6: mIoU vs FWIoU": dict(
        gt_cx=64, gt_cy=64, gt_diameter=25,
        pred_cx=75, pred_cy=75, pred_diameter=25,
    ),
}


# ══════════════════════════════════════════════════════════════
#  VISUALISATION 2 — Comprehensive Metrics Dashboard
# ══════════════════════════════════════════════════════════════
def create_metrics_dashboard(grid_size: int = 128) -> None:
    """Interactive dashboard: binary segmentation overlay + all CV metrics.

    Left panel — same TP/FP/FN/TN overlay as ``create_binary_seg_demo``.
    Right panel — bar chart of all segmentation metrics from the CV section,
    computed on the current slider settings.

    Includes a dropdown for preset failure cases and a Reset button.
    """
    sliders = _make_sliders(grid_size)
    out = widgets.Output()

    # ── Remember default slider values for Reset ─────────────
    _defaults = {k: s.value for k, s in sliders.items()}

    # ── Preset dropdown + Reset button ───────────────────────
    preset_dropdown = widgets.Dropdown(
        options=["-- select a failure case --"] + list(_FAILURE_CASES.keys()),
        value="-- select a failure case --",
        description="Presets:",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="340px"),
    )
    reset_button = widgets.Button(
        description="Reset",
        button_style="warning",
        icon="refresh",
        layout=widgets.Layout(width="100px"),
    )

    def _apply_values(values: dict) -> None:
        """Set slider values without triggering one redraw per slider."""
        for s in sliders.values():
            s.unobserve(_on_change, names="value")
        for key, val in values.items():
            sliders[key].value = val
        for s in sliders.values():
            s.observe(_on_change, names="value")
        _on_change()  # single redraw

    def _on_preset_change(change):
        if change["new"] == "-- select a failure case --":
            return
        preset = _FAILURE_CASES.get(change["new"])
        if preset is not None:
            _apply_values(preset)

    def _on_reset_click(_btn):
        preset_dropdown.value = "-- select a failure case --"
        _apply_values(_defaults)

    preset_dropdown.observe(_on_preset_change, names="value")
    reset_button.on_click(_on_reset_click)

    def _draw(**kwargs):
        gt_mask = _circular_mask(grid_size, kwargs["gt_cx"], kwargs["gt_cy"], kwargs["gt_diameter"])
        pred_mask = _circular_mask(grid_size, kwargs["pred_cx"], kwargs["pred_cy"], kwargs["pred_diameter"])
        img, tp, fp, fn, tn = _build_overlay(gt_mask, pred_mask)
        total = grid_size * grid_size

        # ── Compute all metrics ──────────────────────────────
        pixel_acc = (tp + tn) / total if total > 0 else 0.0

        # Per-class recall (fg = foreground, bg = background)
        fg_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        bg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        mean_pa = (fg_recall + bg_recall) / 2.0

        # IoU per class (one-vs-all)
        iou_fg = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        iou_bg = tn / (tn + fn + fp) if (tn + fn + fp) > 0 else 0.0
        miou = (iou_fg + iou_bg) / 2.0

        # Frequency-weighted IoU
        freq_fg = (tp + fn) / total if total > 0 else 0.0
        freq_bg = (tn + fp) / total if total > 0 else 0.0
        fwiou = freq_fg * iou_fg + freq_bg * iou_bg

        # DSC / F1
        dsc = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        # Precision & Recall (foreground)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = fg_recall

        # ── Draw ─────────────────────────────────────────────
        fig, (ax_img, ax_met) = plt.subplots(
            1, 2, figsize=(16, 7),
            gridspec_kw={"width_ratios": [1, 1.1]},
        )

        # Left: overlay
        ax_img.imshow(img, interpolation="nearest")
        ax_img.set_title("Binary Segmentation", fontsize=14, fontweight="bold")
        ax_img.axis("off")
        ax_img.legend(handles=_legend_patches(tp, fp, fn, tn),
                      loc="lower right", fontsize=10, framealpha=0.9)

        # Right: horizontal bar chart of metrics
        metric_items = [
            ("Pixel Accuracy",       pixel_acc),
            ("Mean Pixel Accuracy",  mean_pa),
            ("IoU (foreground)",     iou_fg),
            ("IoU (background)",     iou_bg),
            ("mIoU",                 miou),
            ("FWIoU",                fwiou),
            ("DSC / F1",             dsc),
            ("Precision",            precision),
            ("Recall",               recall),
        ]
        names = [m[0] for m in metric_items][::-1]
        values = [m[1] for m in metric_items][::-1]

        y_pos = np.arange(len(names))
        bars = ax_met.barh(y_pos, values, color="steelblue",
                           edgecolor="black", linewidth=0.6, height=0.6)
        ax_met.set_yticks(y_pos)
        ax_met.set_yticklabels(names, fontsize=11)
        ax_met.set_xlim(0, 1.15)
        ax_met.set_xlabel("Score", fontsize=12)
        ax_met.set_title("Segmentation Metrics", fontsize=14, fontweight="bold")

        for bar_rect, val in zip(bars, values):
            ax_met.text(bar_rect.get_width() + 0.02, bar_rect.get_y() + bar_rect.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=10)

        ax_met.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _on_change(_=None):
        with out:
            clear_output(wait=True)
            _draw(**{k: v.value for k, v in sliders.items()})

    for s in sliders.values():
        s.observe(_on_change, names="value")

    # Initial render
    with out:
        _draw(**{k: v.value for k, v in sliders.items()})

    ui_gt = widgets.VBox([sliders["gt_cx"], sliders["gt_cy"], sliders["gt_diameter"]],
                         layout=widgets.Layout(margin="0 20px 0 0"))
    ui_pred = widgets.VBox([sliders["pred_cx"], sliders["pred_cy"], sliders["pred_diameter"]])
    slider_row = widgets.HBox([ui_gt, ui_pred])
    preset_row = widgets.HBox(
        [preset_dropdown, reset_button],
        layout=widgets.Layout(margin="4px 0 8px 0"),
    )
    ui = widgets.VBox([preset_row, slider_row])
    display(widgets.VBox([ui, out]))


# ══════════════════════════════════════════════════════════════
#  Volume-metric presets (grid_size = 128)
# ══════════════════════════════════════════════════════════════
_VOLUME_PRESETS = {
    "Shifted (same volume)": dict(
        gt_cx=64, gt_cy=64, gt_diameter=60,
        pred_cx=90, pred_cy=90, pred_diameter=60,
    ),
    "Fragmented (same volume)": dict(
        gt_cx=64, gt_cy=64, gt_diameter=60,
        pred_cx=64, pred_cy=64, pred_diameter=60,
        _fragmented=True,
    ),
    "Over-segmentation": dict(
        gt_cx=64, gt_cy=64, gt_diameter=50,
        pred_cx=64, pred_cy=64, pred_diameter=80,
    ),
    "Under-segmentation": dict(
        gt_cx=64, gt_cy=64, gt_diameter=70,
        pred_cx=64, pred_cy=64, pred_diameter=44,
    ),
    "Near-perfect": dict(
        gt_cx=64, gt_cy=64, gt_diameter=60,
        pred_cx=65, pred_cy=65, pred_diameter=59,
    ),
}


def _fragmented_mask(grid_size: int, gt_cx: int, gt_cy: int,
                     gt_diameter: int, n_fragments: int = 4) -> np.ndarray:
    """Create *n_fragments* small circles whose total area ≈ GT area.

    Fragments are placed at fixed angular positions (corners of a
    regular polygon) at a distance that keeps them well separated from
    the GT region, making the spatial mismatch obvious.
    """
    gt_area = np.pi * (gt_diameter / 2.0) ** 2
    frag_area = gt_area / n_fragments
    frag_r = np.sqrt(frag_area / np.pi)
    frag_d = max(int(round(2 * frag_r)), 2)

    # Place fragments on a ring around the grid centre, offset from GT
    ring_r = grid_size * 0.35
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    for i in range(n_fragments):
        angle = 2 * np.pi * i / n_fragments + np.pi / 4  # start at 45°
        cx = int(round(grid_size / 2 + ring_r * np.cos(angle)))
        cy = int(round(grid_size / 2 + ring_r * np.sin(angle)))
        cx = np.clip(cx, frag_d, grid_size - 1 - frag_d)
        cy = np.clip(cy, frag_d, grid_size - 1 - frag_d)
        mask |= _circular_mask(grid_size, cx, cy, frag_d)
    return mask


# ══════════════════════════════════════════════════════════════
#  VISUALISATION 5 — Volume Metrics Dashboard
# ══════════════════════════════════════════════════════════════
def create_volume_metrics_dashboard(grid_size: int = 128) -> None:
    """Interactive dashboard: volume metrics vs overlap and boundary metrics.

    Left panel  — mask overlay with extracted boundaries drawn on top.
    Right panel — metric table with colour-coded values for DSC, IoU,
    HD95, ASSD, AVD, and RVD.

    Presets highlight failure cases where volume metrics are misleading:
    * **Shifted (same volume)**: AVD and RVD ≈ 0, but DSC and boundary
      metrics reveal a large spatial mismatch.
    * **Fragmented (same volume)**: four small disconnected circles
      whose total area equals the GT area; AVD ≈ 0 but everything
      else is poor.

    Parameters
    ----------
    grid_size : int
        Side length of the square canvas (pixels).
    """

    style = {"description_width": "110px"}
    layout = widgets.Layout(width="380px")
    half = grid_size // 2

    # ── Sliders ──────────────────────────────────────────────
    sliders = dict(
        gt_cx=widgets.IntSlider(
            value=half, min=0, max=grid_size - 1,
            description="GT centre X:", style=style, layout=layout),
        gt_cy=widgets.IntSlider(
            value=half, min=0, max=grid_size - 1,
            description="GT centre Y:", style=style, layout=layout),
        gt_diameter=widgets.IntSlider(
            value=60, min=2, max=grid_size,
            description="GT diameter:", style=style, layout=layout),
        pred_cx=widgets.IntSlider(
            value=half + 15, min=0, max=grid_size - 1,
            description="Pred centre X:", style=style, layout=layout),
        pred_cy=widgets.IntSlider(
            value=half + 15, min=0, max=grid_size - 1,
            description="Pred centre Y:", style=style, layout=layout),
        pred_diameter=widgets.IntSlider(
            value=55, min=2, max=grid_size,
            description="Pred diameter:", style=style, layout=layout),
    )

    out = widgets.Output()
    _defaults = {k: s.value for k, s in sliders.items()}

    # Track whether the fragmented preset is active
    _state = {"fragmented": False}

    # ── Preset dropdown + Reset ──────────────────────────────
    preset_dropdown = widgets.Dropdown(
        options=["-- select a preset --"] + list(_VOLUME_PRESETS.keys()),
        value="-- select a preset --",
        description="Presets:",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="340px"),
    )
    reset_button = widgets.Button(
        description="Reset", button_style="warning", icon="refresh",
        layout=widgets.Layout(width="100px"),
    )

    def _set_sliders_enabled(enabled: bool) -> None:
        """Enable or grey-out all sliders."""
        for s in sliders.values():
            s.disabled = not enabled

    def _apply(values: dict) -> None:
        """Batch-update slider values without triggering intermediate redraws."""
        _state["fragmented"] = values.pop("_fragmented", False)
        for s in sliders.values():
            s.unobserve(_on_change, names="value")
        for key, val in values.items():
            if key in sliders:
                sliders[key].value = val
        for s in sliders.values():
            s.observe(_on_change, names="value")
        _set_sliders_enabled(not _state["fragmented"])
        _redraw()

    def _on_preset(change):
        if change["new"] == "-- select a preset --":
            return
        preset = _VOLUME_PRESETS.get(change["new"])
        if preset is not None:
            _apply(dict(preset))  # copy so pop doesn't mutate original

    def _on_reset(_btn):
        preset_dropdown.value = "-- select a preset --"
        _state["fragmented"] = False
        _apply(dict(_defaults))

    preset_dropdown.observe(_on_preset, names="value")
    reset_button.on_click(_on_reset)

    # ── Metric computations ──────────────────────────────────
    def _compute_metrics(gt_mask, pred_mask):
        """Return dict with DSC, IoU, HD95, ASSD, AVD (px), RVD."""
        tp = int((pred_mask & gt_mask).sum())
        fp = int((pred_mask & ~gt_mask).sum())
        fn = int((~pred_mask & gt_mask).sum())

        dsc = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        # Boundaries
        gt_bd = _extract_boundary(gt_mask)
        pred_bd = _extract_boundary(pred_mask)

        if not gt_bd.any() or not pred_bd.any():
            hd95 = float("nan")
            assd = float("nan")
        else:
            dt_gt = distance_transform_edt(~gt_bd)
            dt_pred = distance_transform_edt(~pred_bd)
            d_pred_to_gt = dt_gt[pred_bd]
            d_gt_to_pred = dt_pred[gt_bd]
            hd95 = float(max(np.percentile(d_pred_to_gt, 95),
                             np.percentile(d_gt_to_pred, 95)))
            assd = float((d_pred_to_gt.sum() + d_gt_to_pred.sum())
                         / (len(d_pred_to_gt) + len(d_gt_to_pred)))

        # Volume metrics (pixel counts)
        vol_gt = int(gt_mask.sum())
        vol_pred = int(pred_mask.sum())
        avd = abs(vol_pred - vol_gt)
        rvd = (vol_pred - vol_gt) / vol_gt if vol_gt > 0 else float("nan")

        return dict(DSC=dsc, IoU=iou, HD95=hd95, ASSD=assd,
                    AVD=avd, RVD=rvd,
                    vol_gt=vol_gt, vol_pred=vol_pred)

    # ── Clipping detection ────────────────────────────────
    def _is_clipped(cx, cy, diameter):
        """True when any part of the circle extends beyond the canvas."""
        r = diameter / 2.0
        return cx - r < 0 or cx + r >= grid_size or cy - r < 0 or cy + r >= grid_size

    # ── Draw ─────────────────────────────────────────────────
    def _draw(**kw):
        gt_mask = _circular_mask(grid_size, kw["gt_cx"], kw["gt_cy"],
                                 kw["gt_diameter"])

        if _state["fragmented"]:
            pred_mask = _fragmented_mask(
                grid_size, kw["gt_cx"], kw["gt_cy"], kw["gt_diameter"])
        else:
            pred_mask = _circular_mask(
                grid_size, kw["pred_cx"], kw["pred_cy"], kw["pred_diameter"])

        gt_clipped = _is_clipped(kw["gt_cx"], kw["gt_cy"], kw["gt_diameter"])
        pred_clipped = (not _state["fragmented"]
                        and _is_clipped(kw["pred_cx"], kw["pred_cy"],
                                        kw["pred_diameter"]))

        metrics = _compute_metrics(gt_mask, pred_mask)

        fig, (ax_img, ax_tbl) = plt.subplots(
            1, 2, figsize=(16, 7),
            gridspec_kw={"width_ratios": [1, 1]})

        # ── Left panel: overlay with boundaries ──────────────
        gt_bd = _extract_boundary(gt_mask)
        pred_bd = _extract_boundary(pred_mask)

        canvas = np.ones((grid_size, grid_size, 3)) * 0.95
        gt_only = gt_mask & ~pred_mask
        pred_only = pred_mask & ~gt_mask
        overlap = gt_mask & pred_mask
        canvas[gt_only]         = [0.78, 0.88, 1.0]
        canvas[pred_only]       = [1.0, 0.84, 0.84]
        canvas[overlap]         = [0.85, 0.80, 0.95]
        canvas[gt_bd]           = [0.0, 0.35, 0.85]
        canvas[pred_bd]         = [0.85, 0.15, 0.10]
        canvas[gt_bd & pred_bd] = [0.5, 0.0, 0.5]

        ax_img.imshow(canvas, interpolation="nearest")
        title_extra = " [fragmented]" if _state["fragmented"] else ""
        ax_img.set_title(f"Mask overlay + boundaries{title_extra}",
                         fontsize=13, fontweight="bold")
        ax_img.axis("off")
        ax_img.legend(handles=[
            mpatches.Patch(color=[0.0, 0.35, 0.85], label="GT boundary"),
            mpatches.Patch(color=[0.85, 0.15, 0.10], label="Pred boundary"),
            mpatches.Patch(color=[0.78, 0.88, 1.0],  label="FN region"),
            mpatches.Patch(color=[1.0, 0.84, 0.84],  label="FP region"),
            mpatches.Patch(color=[0.85, 0.80, 0.95], label="Overlap"),
        ], loc="lower right", fontsize=8, framealpha=0.9)

        # ── Right panel: metric table ────────────────────────
        ax_tbl.axis("off")
        ax_tbl.set_title("Volume vs Overlap vs Boundary Metrics",
                         fontsize=13, fontweight="bold")

        # Build table data
        m = metrics
        vol_gt = m["vol_gt"]
        vol_pred = m["vol_pred"]

        # Colour-coding helpers
        def _overlap_color(v):
            """Green if close to 1, red if close to 0."""
            if np.isnan(v):
                return "#cccccc"
            return "#2ca02c" if v >= 0.8 else "#e89b0c" if v >= 0.5 else "#d62728"

        def _dist_color(v, good_thresh=3.0, warn_thresh=10.0):
            """Green if small, red if large."""
            if np.isnan(v):
                return "#cccccc"
            return "#2ca02c" if v <= good_thresh else "#e89b0c" if v <= warn_thresh else "#d62728"

        def _vol_color_avd(v, total):
            """Green if AVD is small fraction of GT volume."""
            if np.isnan(v) or total == 0:
                return "#cccccc"
            frac = v / total
            return "#2ca02c" if frac < 0.05 else "#e89b0c" if frac < 0.15 else "#d62728"

        def _vol_color_rvd(v):
            if np.isnan(v):
                return "#cccccc"
            a = abs(v)
            return "#2ca02c" if a < 0.05 else "#e89b0c" if a < 0.15 else "#d62728"

        # Metric rows: (group, name, value_str, colour, direction_hint)
        rows = [
            ("Overlap",  "DSC",  f"{m['DSC']:.3f}",
             _overlap_color(m["DSC"]),  "higher is better"),
            ("",         "IoU",  f"{m['IoU']:.3f}",
             _overlap_color(m["IoU"]),  "higher is better"),
            ("Boundary", "HD95", f"{m['HD95']:.1f} px" if np.isfinite(m["HD95"]) else "N/A",
             _dist_color(m["HD95"]),    "lower is better"),
            ("",         "ASSD", f"{m['ASSD']:.1f} px" if np.isfinite(m["ASSD"]) else "N/A",
             _dist_color(m["ASSD"]),    "lower is better"),
            ("Volume",   "AVD",  f"{m['AVD']:,} px",
             _vol_color_avd(m["AVD"], vol_gt), "lower is better"),
            ("",         "RVD",  f"{m['RVD']:+.3f}" if np.isfinite(m["RVD"]) else "N/A",
             _vol_color_rvd(m["RVD"]), "closer to 0"),
        ]

        # Column headers
        col_labels = ["Group", "Metric", "Value", "Interpretation"]
        n_rows = len(rows)

        # Create a table using matplotlib
        cell_text = []
        cell_colours = []
        for group, name, val_str, colour, hint in rows:
            cell_text.append([group, name, val_str, hint])
            cell_colours.append(["#f5f5f5", "#f5f5f5", colour + "28", "#f5f5f5"])

        table = ax_tbl.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            colWidths=[0.18, 0.14, 0.28, 0.32],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.0, 2.2)

        # Style header
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_facecolor("#404040")
            cell.set_edgecolor("white")

        # Style data cells
        for i in range(n_rows):
            for j in range(len(col_labels)):
                cell = table[i + 1, j]
                cell.set_edgecolor("#cccccc")
                if j == 2:
                    # value column: tint with metric colour
                    base = cell_colours[i][j]
                    cell.set_facecolor(base)
                    cell.set_text_props(fontweight="bold", fontsize=13)
                else:
                    cell.set_facecolor(cell_colours[i][j])

        # Volume info below table
        gt_tag = " (clipped!)" if gt_clipped else ""
        pred_tag = " (clipped!)" if pred_clipped else ""
        vol_line = (f"GT volume = {vol_gt:,} px{gt_tag}    |    "
                    f"Pred volume = {vol_pred:,} px{pred_tag}")

        any_clipped = gt_clipped or pred_clipped
        box_face = "#fff3cd" if any_clipped else "#e8e8e8"
        box_edge = "#d4a017" if any_clipped else "#999999"

        ax_tbl.text(
            0.5, 0.08, vol_line,
            ha="center", va="center", fontsize=11,
            transform=ax_tbl.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=box_face,
                      edgecolor=box_edge, alpha=0.9),
        )
        if any_clipped:
            ax_tbl.text(
                0.5, 0.01,
                "Circle extends beyond canvas; pixel count is lower "
                "than the full circle area.",
                ha="center", va="center", fontsize=9,
                color="#996600", style="italic",
                transform=ax_tbl.transAxes,
            )

        plt.tight_layout()
        plt.show()

    def _redraw():
        """Re-render without touching _state['fragmented']."""
        with out:
            clear_output(wait=True)
            _draw(**{k: v.value for k, v in sliders.items()})

    def _on_change(_=None):
        _redraw()

    for s in sliders.values():
        s.observe(_on_change, names="value")

    # Initial render
    _redraw()

    # ── Layout ───────────────────────────────────────────────
    ui_gt = widgets.VBox(
        [sliders["gt_cx"], sliders["gt_cy"], sliders["gt_diameter"]],
        layout=widgets.Layout(margin="0 20px 0 0"))
    ui_pred = widgets.VBox(
        [sliders["pred_cx"], sliders["pred_cy"], sliders["pred_diameter"]])
    slider_row = widgets.HBox([ui_gt, ui_pred])
    preset_row = widgets.HBox(
        [preset_dropdown, reset_button],
        layout=widgets.Layout(margin="4px 0 8px 0"))
    ui = widgets.VBox([preset_row, slider_row])
    display(widgets.VBox([ui, out]))

# ──────────────────────────────────────────────────────────────
#  Constants for boundary-demo sample points
# ──────────────────────────────────────────────────────────────
_SAMPLE_ANGLES = np.linspace(0, 2 * np.pi, 5, endpoint=False)

_POINT_COLORS = [
    (0.90, 0.10, 0.29),   # red
    (0.24, 0.71, 0.29),   # green
    (0.26, 0.39, 0.85),   # blue
    (0.96, 0.51, 0.19),   # orange
    (0.57, 0.12, 0.71),   # purple
]


# ══════════════════════════════════════════════════════════════
#  VISUALISATION 3 — Interactive Boundary Distance Demo
# ══════════════════════════════════════════════════════════════
def create_boundary_demo(grid_size: int = 128) -> None:
    """Interactive demo: boundary extraction + sampled boundary distances.

    Two panels:
      Left  — GT and Pred masks with their extracted boundaries and five
              evenly-spaced sample points on the Pred boundary, each
              connected by an arrow to the nearest GT-boundary point.
      Right — small table listing the five point-to-boundary distances.

    The five sample points are placed at fixed angular positions relative
    to the prediction circle centre so that they remain stable when only
    the diameter changes.
    """
    sliders = _make_sliders(grid_size)
    out = widgets.Output()

    def _sample_boundary_points(pred_bd, gt_bd, pred_cx, pred_cy, pred_r):
        """Return (sample_pts, closest_gt_pts, distances) for 5 samples."""
        pred_bd_yx = np.argwhere(pred_bd)    # (N, 2): row, col
        gt_bd_yx = np.argwhere(gt_bd)        # (M, 2)
        dt_gt = distance_transform_edt(np.logical_not(gt_bd))

        sample_pts, closest_gt_pts, dists = [], [], []
        for angle in _SAMPLE_ANGLES:
            # ideal position on the circle boundary
            target_col = pred_cx + pred_r * np.cos(angle)
            target_row = pred_cy + pred_r * np.sin(angle)

            # snap to nearest actual pred-boundary pixel
            d2 = ((pred_bd_yx[:, 0] - target_row) ** 2
                  + (pred_bd_yx[:, 1] - target_col) ** 2)
            idx = int(np.argmin(d2))
            pr, pc = int(pred_bd_yx[idx, 0]), int(pred_bd_yx[idx, 1])
            sample_pts.append((pr, pc))

            # nearest GT-boundary pixel (for arrow head)
            d2g = ((gt_bd_yx[:, 0] - pr) ** 2
                   + (gt_bd_yx[:, 1] - pc) ** 2)
            gidx = int(np.argmin(d2g))
            gr, gc = int(gt_bd_yx[gidx, 0]), int(gt_bd_yx[gidx, 1])
            closest_gt_pts.append((gr, gc))

            dists.append(float(dt_gt[pr, pc]))

        return sample_pts, closest_gt_pts, dists

    def _draw(**kwargs):
        gt_mask = _circular_mask(
            grid_size, kwargs["gt_cx"], kwargs["gt_cy"], kwargs["gt_diameter"])
        pred_mask = _circular_mask(
            grid_size, kwargs["pred_cx"], kwargs["pred_cy"],
            kwargs["pred_diameter"])

        # ---- boundary extraction ----
        gt_bd = _extract_boundary(gt_mask)
        pred_bd = _extract_boundary(pred_mask)
        has_both = gt_bd.any() and pred_bd.any()

        # ---- sample 5 fixed-angle points on pred boundary ----
        if has_both:
            sample_pts, gt_pts, dists = _sample_boundary_points(
                pred_bd, gt_bd,
                kwargs["pred_cx"], kwargs["pred_cy"],
                kwargs["pred_diameter"] / 2.0)
        else:
            sample_pts, gt_pts, dists = [], [], []

        # ---- figure: boundary panel + distance table ----
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 6),
            gridspec_kw={"width_ratios": [2.2, 1]})

        # ---- Panel 1: masks + boundaries + arrows ----
        canvas = np.ones((grid_size, grid_size, 3)) * 0.95
        gt_only = gt_mask & np.logical_not(pred_mask)
        pred_only = pred_mask & np.logical_not(gt_mask)
        overlap = gt_mask & pred_mask
        canvas[gt_only]            = [0.78, 0.88, 1.0]   # light blue
        canvas[pred_only]          = [1.0, 0.84, 0.84]   # light red
        canvas[overlap]            = [0.85, 0.80, 0.95]  # light purple
        canvas[gt_bd]              = [0.0, 0.35, 0.85]   # blue boundary
        canvas[pred_bd]            = [0.85, 0.15, 0.10]  # red boundary
        canvas[gt_bd & pred_bd]    = [0.5, 0.0, 0.5]     # purple overlap

        ax1.imshow(canvas, interpolation="nearest")
        ax1.set_title("Boundaries  +  sampled distances",
                      fontsize=13, fontweight="bold")
        ax1.axis("off")

        # Draw arrows: pred sample point → nearest GT boundary point
        for i, ((pr, pc), (gr, gc), d) in enumerate(
                zip(sample_pts, gt_pts, dists)):
            colour = _POINT_COLORS[i]
            if d > 0.5:     # skip zero-length arrows
                ax1.annotate(
                    "", xy=(gc, gr), xytext=(pc, pr),
                    arrowprops=dict(arrowstyle="-|>", color=colour,
                                    lw=2.0, mutation_scale=14))
            # coloured dot on the pred boundary
            ax1.plot(pc, pr, "o", color=colour, markersize=8,
                     markeredgecolor="k", markeredgewidth=0.8, zorder=5)
            # label next to the dot
            ax1.annotate(
                f"P{i + 1}", xy=(pc, pr),
                xytext=(6, -6), textcoords="offset points",
                fontsize=8, fontweight="bold", color=colour,
                path_effects=[
                    withStroke(linewidth=2, foreground="white")])

        ax1.legend(
            handles=[
                mpatches.Patch(color=[0.0, 0.35, 0.85],
                               label="GT boundary (\u2202R)"),
                mpatches.Patch(color=[0.85, 0.15, 0.10],
                               label="Pred boundary (\u2202P)"),
            ],
            loc="lower right", fontsize=9, framealpha=0.9)

        # ---- Panel 2: distance table ----
        ax2.axis("off")
        if has_both:
            ax2.set_title("Point \u2192 nearest GT boundary",
                          fontsize=12, fontweight="bold", pad=15)

            cell_text = [[f"{d:.1f} px"] for d in dists]
            row_labels = [f"  P{i + 1}  " for i in range(len(dists))]

            table = ax2.table(
                cellText=cell_text,
                rowLabels=row_labels,
                colLabels=["Distance"],
                cellLoc="center",
                rowLoc="center",
                loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.0, 1.8)

            # header style
            table[0, 0].set_facecolor("#e0e0e0")
            table[0, 0].set_text_props(fontweight="bold")

            # colour-code row labels to match arrows
            for i in range(len(dists)):
                table[i + 1, -1].set_facecolor(
                    (*_POINT_COLORS[i], 0.25))
                table[i + 1, -1].set_text_props(fontweight="bold")
                table[i + 1, 0].set_facecolor((0.97, 0.97, 0.97, 1.0))
        else:
            ax2.text(0.5, 0.5, "No boundary\n(empty mask)",
                     ha="center", va="center", fontsize=14,
                     transform=ax2.transAxes)

        plt.tight_layout()
        plt.show()

    def _on_change(_=None):
        with out:
            clear_output(wait=True)
            _draw(**{k: v.value for k, v in sliders.items()})

    for s in sliders.values():
        s.observe(_on_change, names="value")

    with out:
        _draw(**{k: v.value for k, v in sliders.items()})

    ui_gt = widgets.VBox(
        [sliders["gt_cx"], sliders["gt_cy"], sliders["gt_diameter"]],
        layout=widgets.Layout(margin="0 20px 0 0"))
    ui_pred = widgets.VBox(
        [sliders["pred_cx"], sliders["pred_cy"], sliders["pred_diameter"]])
    ui = widgets.HBox([ui_gt, ui_pred])
    display(widgets.VBox([ui, out]))


# ══════════════════════════════════════════════════════════════
#  Boundary-metric presets (grid_size = 128)
# ══════════════════════════════════════════════════════════════
_BOUNDARY_PRESETS = {
    "Protrusion": dict(
        gt_cx=64, gt_cy=64, gt_diameter=60,
        pred_cx=64, pred_cy=64, pred_diameter=60,
        prot_radius=14, prot_angle=45,
    ),
    "Erosion (under-seg)": dict(
        gt_cx=64, gt_cy=64, gt_diameter=70,
        pred_cx=64, pred_cy=64, pred_diameter=44,
        prot_radius=0, prot_angle=0,
    ),
    "Shift": dict(
        gt_cx=64, gt_cy=64, gt_diameter=60,
        pred_cx=78, pred_cy=78, pred_diameter=60,
        prot_radius=0, prot_angle=0,
    ),
    "Over-segmentation": dict(
        gt_cx=64, gt_cy=64, gt_diameter=40,
        pred_cx=64, pred_cy=64, pred_diameter=80,
        prot_radius=0, prot_angle=0,
    ),
    "Near-perfect": dict(
        gt_cx=64, gt_cy=64, gt_diameter=60,
        pred_cx=65, pred_cy=65, pred_diameter=59,
        prot_radius=0, prot_angle=0,
    ),
}


# ══════════════════════════════════════════════════════════════
#  VISUALISATION 4 — Boundary Metrics Dashboard
# ══════════════════════════════════════════════════════════════
def create_boundary_metrics_dashboard(
    grid_size: int = 128,
    nsd_tau: float = 2.0,
) -> None:
    """Interactive dashboard: overlap + boundary metrics side by side.

    Left panel  — mask overlay with extracted boundaries drawn on top.
    Right panel — bar chart of overlap metrics (DSC, IoU, Precision,
    Recall) and boundary metrics (HD, HD95, ASSD, NSD).

    Controls include:
    * GT and Pred position / diameter sliders.
    * A **protrusion radius** slider that adds a circular bump to the
      prediction boundary at a controllable angle.
    * A preset dropdown for common failure modes.

    Parameters
    ----------
    grid_size : int
        Side length of the square canvas (pixels).
    nsd_tau : float
        Tolerance parameter for NSD (pixels).
    """

    style = {"description_width": "110px"}
    layout = widgets.Layout(width="380px")
    half = grid_size // 2

    # ── Sliders ──────────────────────────────────────────────
    sliders = dict(
        gt_cx=widgets.IntSlider(
            value=half, min=0, max=grid_size - 1,
            description="GT centre X:", style=style, layout=layout),
        gt_cy=widgets.IntSlider(
            value=half, min=0, max=grid_size - 1,
            description="GT centre Y:", style=style, layout=layout),
        gt_diameter=widgets.IntSlider(
            value=60, min=2, max=grid_size,
            description="GT diameter:", style=style, layout=layout),
        pred_cx=widgets.IntSlider(
            value=half + 15, min=0, max=grid_size - 1,
            description="Pred centre X:", style=style, layout=layout),
        pred_cy=widgets.IntSlider(
            value=half + 15, min=0, max=grid_size - 1,
            description="Pred centre Y:", style=style, layout=layout),
        pred_diameter=widgets.IntSlider(
            value=55, min=2, max=grid_size,
            description="Pred diameter:", style=style, layout=layout),
        prot_radius=widgets.IntSlider(
            value=0, min=0, max=grid_size // 3,
            description="Protrusion r:", style=style, layout=layout),
        prot_angle=widgets.IntSlider(
            value=0, min=0, max=359, step=5,
            description="Prot. angle °:", style=style, layout=layout),
    )

    out = widgets.Output()
    _defaults = {k: s.value for k, s in sliders.items()}

    # ── Preset dropdown + Reset ──────────────────────────────
    preset_dropdown = widgets.Dropdown(
        options=["-- select a preset --"] + list(_BOUNDARY_PRESETS.keys()),
        value="-- select a preset --",
        description="Presets:",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="340px"),
    )
    reset_button = widgets.Button(
        description="Reset", button_style="warning", icon="refresh",
        layout=widgets.Layout(width="100px"),
    )

    def _apply(values: dict) -> None:
        for s in sliders.values():
            s.unobserve(_on_change, names="value")
        for key, val in values.items():
            sliders[key].value = val
        for s in sliders.values():
            s.observe(_on_change, names="value")
        _on_change()

    def _on_preset(change):
        if change["new"] == "-- select a preset --":
            return
        preset = _BOUNDARY_PRESETS.get(change["new"])
        if preset is not None:
            _apply(preset)

    def _on_reset(_btn):
        preset_dropdown.value = "-- select a preset --"
        _apply(_defaults)

    preset_dropdown.observe(_on_preset, names="value")
    reset_button.on_click(_on_reset)

    # ── Build prediction mask (circle + optional protrusion) ─
    def _pred_mask(kw):
        base = _circular_mask(grid_size, kw["pred_cx"], kw["pred_cy"],
                              kw["pred_diameter"])
        pr = kw["prot_radius"]
        if pr <= 0:
            return base
        # place protrusion on the edge of the prediction circle
        angle_rad = np.deg2rad(kw["prot_angle"])
        pred_r = kw["pred_diameter"] / 2.0
        bump_cx = kw["pred_cx"] + pred_r * np.cos(angle_rad)
        bump_cy = kw["pred_cy"] + pred_r * np.sin(angle_rad)
        bump = _circular_mask(grid_size, int(round(bump_cx)),
                              int(round(bump_cy)), 2 * pr)
        return base | bump

    # ── Metric computations ──────────────────────────────────
    def _compute_metrics(gt_mask, pred_mask):
        """Return dict of overlap + boundary metrics."""
        tp = int((pred_mask & gt_mask).sum())
        fp = int((pred_mask & ~gt_mask).sum())
        fn = int((~pred_mask & gt_mask).sum())

        dsc = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Boundaries
        gt_bd = _extract_boundary(gt_mask)
        pred_bd = _extract_boundary(pred_mask)

        if not gt_bd.any() or not pred_bd.any():
            return dict(DSC=dsc, IoU=iou, Precision=prec, Recall=rec,
                        HD=float("nan"), HD95=float("nan"),
                        ASSD=float("nan"), NSD=float("nan"))

        # Distance transforms
        dt_gt = distance_transform_edt(~gt_bd)
        dt_pred = distance_transform_edt(~pred_bd)

        d_pred_to_gt = dt_gt[pred_bd]   # distance from each pred bd pt to nearest gt bd pt
        d_gt_to_pred = dt_pred[gt_bd]   # distance from each gt bd pt to nearest pred bd pt

        # HD
        hd = float(max(d_pred_to_gt.max(), d_gt_to_pred.max()))

        # HD95
        hd95 = float(max(np.percentile(d_pred_to_gt, 95),
                         np.percentile(d_gt_to_pred, 95)))

        # ASSD
        assd = float((d_pred_to_gt.sum() + d_gt_to_pred.sum())
                      / (len(d_pred_to_gt) + len(d_gt_to_pred)))

        # NSD
        n_p = int((d_pred_to_gt <= nsd_tau).sum())
        n_r = int((d_gt_to_pred <= nsd_tau).sum())
        nsd = (n_p + n_r) / (len(d_pred_to_gt) + len(d_gt_to_pred))

        return dict(DSC=dsc, IoU=iou, Precision=prec, Recall=rec,
                    HD=hd, HD95=hd95, ASSD=assd, NSD=nsd)

    # ── Draw ─────────────────────────────────────────────────
    def _draw(**kw):
        gt_mask = _circular_mask(grid_size, kw["gt_cx"], kw["gt_cy"],
                                 kw["gt_diameter"])
        pred_mask = _pred_mask(kw)
        metrics = _compute_metrics(gt_mask, pred_mask)

        fig, (ax_img, ax_met) = plt.subplots(
            1, 2, figsize=(16, 7),
            gridspec_kw={"width_ratios": [1, 1.1]})

        # ── Left panel: overlay with boundaries ──────────────
        gt_bd = _extract_boundary(gt_mask)
        pred_bd = _extract_boundary(pred_mask)

        canvas = np.ones((grid_size, grid_size, 3)) * 0.95
        gt_only = gt_mask & ~pred_mask
        pred_only = pred_mask & ~gt_mask
        overlap = gt_mask & pred_mask
        canvas[gt_only]        = [0.78, 0.88, 1.0]   # light blue
        canvas[pred_only]      = [1.0, 0.84, 0.84]   # light red
        canvas[overlap]        = [0.85, 0.80, 0.95]   # light purple
        canvas[gt_bd]          = [0.0, 0.35, 0.85]    # blue boundary
        canvas[pred_bd]        = [0.85, 0.15, 0.10]   # red boundary
        canvas[gt_bd & pred_bd] = [0.5, 0.0, 0.5]     # purple overlap

        ax_img.imshow(canvas, interpolation="nearest")
        ax_img.set_title("Mask overlay + boundaries",
                         fontsize=13, fontweight="bold")
        ax_img.axis("off")
        ax_img.legend(handles=[
            mpatches.Patch(color=[0.0, 0.35, 0.85], label="GT boundary"),
            mpatches.Patch(color=[0.85, 0.15, 0.10], label="Pred boundary"),
            mpatches.Patch(color=[0.78, 0.88, 1.0],  label="FN region"),
            mpatches.Patch(color=[1.0, 0.84, 0.84],  label="FP region"),
            mpatches.Patch(color=[0.85, 0.80, 0.95], label="Overlap"),
        ], loc="lower right", fontsize=8, framealpha=0.9)

        # ── Right panel: grouped bar chart ──────────────────
        # Group 1: overlap (higher is better, 0-1)
        # Group 2: boundary distances (lower is better, variable scale)
        # Group 3: NSD (higher is better, 0-1)
        overlap_names = ["DSC", "IoU", "Precision", "Recall"]
        overlap_vals = [metrics[k] for k in overlap_names]

        dist_names = ["HD", "HD95", "ASSD"]
        dist_vals = [metrics[k] for k in dist_names]

        nsd_val = metrics["NSD"]

        # Use two y-axes: left for 0-1 scores, right for distance
        ax_met.set_xlim(-0.5, 8.5)
        ax_met.set_ylim(0, 1.15)

        # Overlap bars (0-1)
        x_overlap = np.arange(len(overlap_names))
        bars1 = ax_met.bar(x_overlap, overlap_vals, width=0.7,
                           color="steelblue", edgecolor="black",
                           linewidth=0.6, label="Overlap (0-1, ↑)")
        for b, v in zip(bars1, overlap_vals):
            ax_met.text(b.get_x() + b.get_width() / 2, v + 0.02,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        # NSD bar (0-1, higher is better)
        x_nsd = len(overlap_names)
        bar_nsd = ax_met.bar(x_nsd, nsd_val, width=0.7,
                             color="#2ca02c", edgecolor="black",
                             linewidth=0.6, label=f"NSD τ={nsd_tau} (0-1, ↑)")
        ax_met.text(x_nsd, nsd_val + 0.02,
                    f"{nsd_val:.3f}", ha="center", va="bottom", fontsize=9)

        ax_met.set_ylabel("Score (0-1, higher is better)", fontsize=11)

        # Distance bars on secondary axis
        ax2 = ax_met.twinx()
        x_dist = np.arange(len(dist_names)) + len(overlap_names) + 1
        max_dist = max(dist_vals) if all(np.isfinite(dist_vals)) else 1.0
        bars2 = ax2.bar(x_dist, dist_vals, width=0.7,
                        color="#d62728", edgecolor="black",
                        linewidth=0.6, alpha=0.85, label="Boundary dist (px, ↓)")
        for b, v in zip(bars2, dist_vals):
            label = f"{v:.1f}" if np.isfinite(v) else "N/A"
            ax2.text(b.get_x() + b.get_width() / 2, v + max_dist * 0.03,
                     label, ha="center", va="bottom", fontsize=9)

        ax2.set_ylim(0, max_dist * 1.25 if max_dist > 0 else 1.0)
        ax2.set_ylabel("Distance (px, lower is better)", fontsize=11,
                       color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")

        # X ticks
        all_names = overlap_names + [f"NSD\nτ={nsd_tau}"] + dist_names
        all_x = list(x_overlap) + [x_nsd] + list(x_dist)
        ax_met.set_xticks(all_x)
        ax_met.set_xticklabels(all_names, fontsize=10)
        ax_met.set_title("Overlap  vs  Boundary Metrics",
                         fontsize=13, fontweight="bold")

        # Separator line between groups
        sep_x = len(overlap_names) + 0.5
        ax_met.axvline(sep_x, color="grey", ls="--", lw=0.8, alpha=0.5)

        # Combined legend
        handles1, labels1 = ax_met.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax_met.legend(handles1 + handles2, labels1 + labels2,
                      loc="upper right", fontsize=8, framealpha=0.9)

        ax_met.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _on_change(_=None):
        with out:
            clear_output(wait=True)
            _draw(**{k: v.value for k, v in sliders.items()})

    for s in sliders.values():
        s.observe(_on_change, names="value")

    # Initial render
    with out:
        _draw(**{k: v.value for k, v in sliders.items()})

    # ── Layout ───────────────────────────────────────────────
    ui_gt = widgets.VBox(
        [sliders["gt_cx"], sliders["gt_cy"], sliders["gt_diameter"]],
        layout=widgets.Layout(margin="0 20px 0 0"))
    ui_pred = widgets.VBox(
        [sliders["pred_cx"], sliders["pred_cy"], sliders["pred_diameter"],
         sliders["prot_radius"], sliders["prot_angle"]])
    slider_row = widgets.HBox([ui_gt, ui_pred])
    preset_row = widgets.HBox(
        [preset_dropdown, reset_button],
        layout=widgets.Layout(margin="4px 0 8px 0"))
    ui = widgets.VBox([preset_row, slider_row])
    display(widgets.VBox([ui, out]))
