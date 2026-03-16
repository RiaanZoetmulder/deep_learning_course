"""Interactive FROC curve demo: sweep the confidence threshold and watch
Sensitivity and FPI change on the curve.

Functions
---------
create_froc_curve_demo    Widget that sweeps confidence and shows the FROC curve
                          with the current operating point highlighted.
"""

import io
import threading

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _bbox_iou(box_a, box_b):
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


def _fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ------------------------------------------------------------------
# Synthetic data for the demo
# ------------------------------------------------------------------

def _build_demo_data(seed=2026):
    """Generate a lung-nodule screening dataset that produces a realistic
    FROC curve spanning FPI 0 to ~10+.

    Key design choices for a clear demo:
    - 50 images, 15 positive → 35 healthy (small N so FPI grows fast)
    - ~25 GT nodules total
    - TP detections spread across a wide confidence range (0.15–0.95)
    - Many FPs at low confidence so the curve reaches high FPI values
    - Some hard nodules only detected at very low confidence

    Returns
    -------
    pred_boxes, pred_scores, pred_image_ids : prediction arrays
    gt_boxes, gt_image_ids                  : ground-truth arrays
    image_labels : (N,) 0/1 per image
    num_images   : int
    """
    rng = np.random.RandomState(seed)

    num_images = 50
    n_positive = 15
    n_negative = num_images - n_positive

    image_labels = np.zeros(num_images, dtype=int)
    image_labels[:n_positive] = 1

    # Ground-truth: 1-2 nodules per positive image (~25 total)
    gt_records = []
    for img_id in range(n_positive):
        n_nodules = rng.choice([1, 1, 2, 2, 2])
        for _ in range(n_nodules):
            cx, cy = rng.randint(40, 220), rng.randint(40, 220)
            w, h = rng.randint(15, 35), rng.randint(15, 35)
            gt_records.append((img_id, cx - w, cy - h, cx + w, cy + h))

    gt_image_ids = np.array([r[0] for r in gt_records])
    gt_boxes = np.array([r[1:] for r in gt_records], dtype=float)
    n_gt = len(gt_boxes)

    pred_records = []  # (image_id, confidence, x1, y1, x2, y2)

    # --- True positives: spread confidence across a wide range ---
    # ~75% detected (some are hard and only found at low conf)
    for j in range(n_gt):
        if rng.rand() < 0.76:
            gt = gt_boxes[j]
            dx, dy = rng.randint(-5, 6, size=2)
            # Spread confidences: some easy (high), some hard (low)
            if rng.rand() < 0.45:
                conf = np.clip(rng.beta(8, 3), 0.55, 0.95)   # easy
            elif rng.rand() < 0.6:
                conf = np.clip(rng.beta(3, 3), 0.25, 0.60)   # medium
            else:
                conf = np.clip(rng.beta(2, 5), 0.08, 0.35)   # hard
            pred_records.append((
                gt_image_ids[j], float(conf),
                gt[0] + dx, gt[1] + dy, gt[2] + dx, gt[3] + dy,
            ))

    # --- FPs on positive images (moderate-to-low confidence) ---
    for img_id in range(n_positive):
        n_fp = rng.choice([0, 1, 1, 2, 2, 3])
        for _ in range(n_fp):
            cx, cy = rng.randint(20, 240), rng.randint(20, 240)
            w, h = rng.randint(10, 30), rng.randint(10, 30)
            conf = float(np.clip(rng.beta(2, 4), 0.03, 0.50))
            pred_records.append((img_id, conf, cx - w, cy - h, cx + w, cy + h))

    # --- FPs on healthy images (many, low confidence) ---
    # Need enough FPs so that total_FP / num_images > 8
    # Target: ~500 FP total → 500/50 = 10 FPI at lowest threshold
    for img_id in range(n_positive, num_images):
        n_fp = rng.choice([5, 8, 10, 12, 15, 18])
        for _ in range(n_fp):
            cx, cy = rng.randint(20, 240), rng.randint(20, 240)
            w, h = rng.randint(8, 28), rng.randint(8, 28)
            conf = float(np.clip(rng.beta(1.5, 8), 0.01, 0.45))
            pred_records.append((
                img_id, conf, cx - w, cy - h, cx + w, cy + h,
            ))

    pred_image_ids = np.array([r[0] for r in pred_records])
    pred_scores = np.array([r[1] for r in pred_records])
    pred_boxes = np.array([r[2:] for r in pred_records], dtype=float)

    return (pred_boxes, pred_scores, pred_image_ids,
            gt_boxes, gt_image_ids,
            image_labels, num_images, n_gt)


# ------------------------------------------------------------------
# FROC curve computation  (same logic as the notebook cell that
# follows, but self-contained for the demo)
# ------------------------------------------------------------------

def _compute_froc(pred_image_ids, pred_boxes, pred_scores,
                  gt_image_ids, gt_boxes, num_images, iou_thresh=0.5):
    """Compute the FROC curve (FPI, Sensitivity) as threshold sweeps."""
    order = np.argsort(-pred_scores)
    sorted_scores = pred_scores[order]
    sorted_boxes = pred_boxes[order]
    sorted_img_ids = pred_image_ids[order]

    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)

    # For each prediction determine TP or FP via per-image greedy matching
    # We track which GT boxes are matched globally per-threshold sweep
    gt_matched = np.zeros(n_gt, dtype=bool)
    tp_cum = np.zeros(n_pred, dtype=int)
    fp_cum = np.zeros(n_pred, dtype=int)

    for i in range(n_pred):
        img_id = sorted_img_ids[i]
        best_iou = 0.0
        best_gt = -1

        for j in range(n_gt):
            if gt_image_ids[j] != img_id:
                continue
            if gt_matched[j]:
                continue
            iou = _bbox_iou(sorted_boxes[i], gt_boxes[j])
            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if best_iou >= iou_thresh and best_gt >= 0:
            tp_cum[i] = 1
            gt_matched[best_gt] = True
        else:
            fp_cum[i] = 1

    cum_tp = np.cumsum(tp_cum)
    cum_fp = np.cumsum(fp_cum)
    sensitivity = cum_tp / n_gt
    fpi = cum_fp / num_images

    return fpi, sensitivity, sorted_scores


# ------------------------------------------------------------------
# Public API: interactive widget
# ------------------------------------------------------------------

_CSS = ("font-family:monospace;font-size:13px;padding:10px;"
        "background:#f5f5f5;border-radius:6px;margin-top:4px;"
        "line-height:1.55")

LUNA_FP_LEVELS = [0.125, 0.25, 0.5, 1, 2, 4, 8]


def create_froc_curve_demo():
    """Interactive demo: sweep the confidence threshold and watch the
    operating point move along the FROC curve.

    A slider controls the confidence threshold.  At each setting the
    plot shows:
    - The full FROC curve (light blue)
    - The current operating point (large red dot) with Sensitivity and FPI
    - LUNA16 reference levels (green dots)
    - An inset bar showing the TP / FP / FN breakdown
    """

    # ---- build data & precompute full FROC curve ----
    (pred_boxes, pred_scores, pred_image_ids,
     gt_boxes, gt_image_ids,
     image_labels, num_images, n_gt) = _build_demo_data()

    fpi_full, sens_full, scores_sorted = _compute_froc(
        pred_image_ids, pred_boxes, pred_scores,
        gt_image_ids, gt_boxes, num_images,
    )

    # Unique threshold values for the slider (quantised to 2 decimals)
    all_thresholds = np.unique(np.round(
        np.concatenate([[0.01], scores_sorted, [0.99]]), 2))[::-1]

    # ---- widgets ----
    img_widget = widgets.Image(format="png")
    info_html = widgets.HTML()

    def render(threshold):
        old_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        try:
            # Find the operating-point index for this threshold
            # All predictions with score >= threshold are kept
            mask = scores_sorted >= threshold
            n_kept = int(mask.sum())

            if n_kept == 0:
                cur_sens = 0.0
                cur_fpi = 0.0
                cur_tp = 0
                cur_fp = 0
            else:
                cur_sens = float(sens_full[n_kept - 1])
                cur_fpi = float(fpi_full[n_kept - 1])
                cur_tp = int(round(cur_sens * n_gt))
                cur_fp = int(round(cur_fpi * num_images))
            cur_fn = n_gt - cur_tp

            # ---- figure: two panels ----
            fig = plt.figure(figsize=(14, 5.5))
            gs = fig.add_gridspec(1, 5, wspace=0.35)
            ax_froc = fig.add_subplot(gs[0, :3])
            ax_bar = fig.add_subplot(gs[0, 3:])

            # -- Left: FROC curve --
            ax_froc.plot(fpi_full, sens_full, color="#90CAF9",
                         linewidth=2.5, zorder=2, label="FROC curve")
            # shade the "accepted" region
            if n_kept > 0:
                ax_froc.fill_between(
                    fpi_full[:n_kept], sens_full[:n_kept],
                    alpha=0.12, color="#1976D2")
                ax_froc.plot(fpi_full[:n_kept], sens_full[:n_kept],
                             color="#1976D2", linewidth=3, zorder=3)

            # LUNA16 reference dots
            colors_luna = plt.cm.RdYlGn(
                np.linspace(0.25, 0.85, len(LUNA_FP_LEVELS)))
            for fp_level, clr in zip(LUNA_FP_LEVELS, colors_luna):
                if fp_level <= fpi_full[-1]:
                    idx = np.searchsorted(fpi_full, fp_level)
                    s = sens_full[min(idx, len(sens_full) - 1)]
                else:
                    s = sens_full[-1]
                ax_froc.plot(fp_level, s, "o", color=clr, markersize=7,
                             zorder=4, alpha=0.6)

            # Current operating point
            ax_froc.plot(cur_fpi, cur_sens, "o", color="#D32F2F",
                         markersize=14, zorder=6, markeredgecolor="k",
                         markeredgewidth=1.5)
            ax_froc.annotate(
                f"  Sens = {cur_sens:.2f}\n  FPI  = {cur_fpi:.2f}",
                xy=(cur_fpi, cur_sens),
                xytext=(18, -8), textcoords="offset points",
                fontsize=10, fontweight="bold", color="#D32F2F",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#D32F2F", alpha=0.9),
            )

            # Threshold line annotation
            ax_froc.set_xlabel("False Positives per Image (FPI)", fontsize=12)
            ax_froc.set_ylabel("Sensitivity", fontsize=12)
            ax_froc.set_title(
                f"FROC Curve  —  conf ≥ {threshold:.2f}", fontsize=13,
                fontweight="bold")
            ax_froc.set_xlim(-0.15, max(10, fpi_full[-1] + 0.5))
            ax_froc.set_ylim(-0.03, 1.08)
            ax_froc.axhline(1.0, color="gray", ls="--", alpha=0.4)
            ax_froc.grid(True, alpha=0.25)
            ax_froc.legend(loc="lower right", fontsize=10)

            # -- Right: TP / FP / FN bar breakdown --
            cats = ["TP\n(detected)", "FP\n(false alarms)", "FN\n(missed)"]
            vals = [cur_tp, cur_fp, cur_fn]
            colors = ["#43A047", "#E53935", "#FFA726"]
            bars = ax_bar.bar(cats, vals, color=colors, edgecolor="k",
                              linewidth=0.8, width=0.6)
            for bar, val in zip(bars, vals):
                ax_bar.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            str(val), ha="center", va="bottom",
                            fontsize=13, fontweight="bold")
            ax_bar.set_ylim(0, max(n_gt, max(vals) + 5) * 1.15)
            ax_bar.set_ylabel("Count", fontsize=11)
            ax_bar.set_title("Detection Breakdown", fontsize=12,
                             fontweight="bold")
            ax_bar.grid(True, alpha=0.2, axis="y")

            fig.suptitle(
                f"Confidence threshold = {threshold:.2f}   |   "
                f"{n_kept} / {len(scores_sorted)} predictions kept",
                fontsize=11, y=1.01, color="#555")
            fig.tight_layout()
            img_widget.value = _fig_to_png(fig)
        finally:
            matplotlib.use(old_backend)

        # ---- info HTML ----
        info_html.value = (
            f'<div style="{_CSS}">'
            f"<b>Threshold</b> = {threshold:.2f} &nbsp;→&nbsp; "
            f"keep predictions with confidence ≥ {threshold:.2f}<br>"
            f"<b>Kept:</b> {n_kept} predictions &nbsp;|&nbsp; "
            f"<b>TP:</b> {cur_tp} &nbsp;|&nbsp; "
            f"<b>FP:</b> {cur_fp} &nbsp;|&nbsp; "
            f"<b>FN:</b> {cur_fn}<br>"
            f"<b>Sensitivity</b> = TP / (TP + FN) = {cur_tp}/{n_gt} "
            f"= <b>{cur_sens:.3f}</b> &nbsp;|&nbsp; "
            f"<b>FPI</b> = FP / N<sub>images</sub> = {cur_fp}/{num_images} "
            f"= <b>{cur_fpi:.3f}</b><br><br>"
            f"<span style='color:#555'>↑ Raise threshold → fewer detections → "
            f"lower Sensitivity, lower FPI (fewer false alarms)<br>"
            f"↓ Lower threshold → more detections → higher Sensitivity, "
            f"higher FPI (more false alarms)</span></div>"
        )

    # ---- controls ----
    slider = widgets.FloatSlider(
        min=0.05, max=0.95, step=0.05, value=0.50,
        description="Conf ≥",
        continuous_update=False,
        style={"description_width": "50px"},
        layout=widgets.Layout(width="50%"),
        readout_format=".2f",
    )
    bprev = widgets.Button(description="◀ Stricter",
                           layout=widgets.Layout(width="100px"))
    bnext = widgets.Button(description="Looser ▶",
                           layout=widgets.Layout(width="100px"))
    breset = widgets.Button(description="Reset",
                            layout=widgets.Layout(width="80px"))
    bplay = widgets.Button(description="▶ Play",
                           layout=widgets.Layout(width="80px"),
                           button_style="success")
    speed = widgets.Dropdown(
        options=[("0.15 s", 0.15), ("0.3 s", 0.3), ("0.5 s", 0.5)],
        value=0.3, description="Delay:",
        layout=widgets.Layout(width="140px"),
    )

    _play_event = threading.Event()

    def _play_loop():
        """Animate: sweep threshold from high to low."""
        while not _play_event.is_set():
            if slider.value <= slider.min + slider.step:
                break
            slider.value = round(slider.value - slider.step, 2)
            if _play_event.wait(speed.value):
                break
        bplay.description = "▶ Play"
        bplay.button_style = "success"

    def _toggle_play(_):
        if bplay.description.endswith("Play"):
            bplay.description = "⏸ Pause"
            bplay.button_style = "warning"
            _play_event.clear()
            t = threading.Thread(target=_play_loop, daemon=True)
            t.start()
        else:
            _play_event.set()

    slider.observe(lambda c: render(c["new"]), names="value")
    bprev.on_click(lambda _: setattr(
        slider, "value", min(0.95, round(slider.value + 0.05, 2))))
    bnext.on_click(lambda _: setattr(
        slider, "value", max(0.05, round(slider.value - 0.05, 2))))
    breset.on_click(lambda _: setattr(slider, "value", 0.50))
    bplay.on_click(_toggle_play)

    display(widgets.VBox([
        widgets.HBox([bprev, bnext, breset, bplay, speed, slider]),
        img_widget,
        info_html,
    ]))
    render(slider.value)
