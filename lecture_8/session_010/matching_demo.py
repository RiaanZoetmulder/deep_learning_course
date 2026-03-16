"""Interactive step-through demo of greedy VOC-style matching.

Shows two synchronised panels:
- Left:  bounding boxes on a canvas, with the current prediction highlighted
         and IoU lines drawn to all GT boxes.
- Right: IoU matrix heatmap (Predictions x GT). Matched cells are highlighted;
         claimed GT columns get a strikethrough overlay.

A slider (or Prev/Next buttons) advances through the predictions in
confidence-sorted order so students can watch the greedy algorithm unfold.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display, clear_output
import ipywidgets as widgets


# ------------------------------------------------------------------
# IoU helper
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Full IoU matrix
# ------------------------------------------------------------------

def _build_iou_matrix(pred_boxes, gt_boxes):
    """Return (N_pred, N_gt) IoU matrix."""
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    mat = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            mat[i, j] = _compute_iou(pred_boxes[i], gt_boxes[j])
    return mat


# ------------------------------------------------------------------
# Run the VOC-style greedy matching up to *step* predictions
# ------------------------------------------------------------------

def _run_matching(iou_matrix, sorted_indices, iou_threshold, up_to_step):
    """Return per-prediction results and GT matched state after *up_to_step* steps.

    VOC protocol: for each prediction, find the GT with highest IoU among ALL
    GT boxes. If that GT is already matched, the prediction is FP even if
    another unmatched GT has sufficient IoU.

    Returns
    -------
    results : list[str|None]  – 'TP', 'FP', or None (not yet processed), length = n_pred
    gt_matched_by : list[int|None] – pred_idx that matched each GT, or None
    match_info : list[dict|None] – per-step info (best_gt, best_iou, reason)
    """
    n_pred = iou_matrix.shape[0]
    n_gt = iou_matrix.shape[1]
    results = [None] * n_pred
    gt_matched_by = [None] * n_gt
    match_info = [None] * n_pred

    for step in range(min(up_to_step, len(sorted_indices))):
        pred_idx = sorted_indices[step]
        row = iou_matrix[pred_idx]

        best_gt = int(np.argmax(row))
        best_iou = row[best_gt]

        if best_iou < iou_threshold:
            results[pred_idx] = "FP"
            match_info[pred_idx] = dict(
                best_gt=best_gt, best_iou=best_iou,
                reason="IoU below threshold",
            )
        elif gt_matched_by[best_gt] is not None:
            results[pred_idx] = "FP"
            match_info[pred_idx] = dict(
                best_gt=best_gt, best_iou=best_iou,
                reason=f"GT {best_gt} already matched to P{gt_matched_by[best_gt]}",
            )
        else:
            results[pred_idx] = "TP"
            gt_matched_by[best_gt] = pred_idx
            match_info[pred_idx] = dict(
                best_gt=best_gt, best_iou=best_iou,
                reason="Matched!",
            )

    return results, gt_matched_by, match_info


# ------------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------------

_BOX_CANVAS = (-10, 270)  # x/y limits


def _draw_box_panel(ax, gt_boxes, pred_boxes, pred_scores,
                    sorted_indices, step, results, gt_matched_by,
                    match_info, iou_matrix, iou_threshold):
    """Left panel: boxes on a canvas with the current step highlighted."""
    ax.set_xlim(*_BOX_CANVAS)
    ax.set_ylim(*_BOX_CANVAS)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("Bounding Boxes", fontsize=13, fontweight="bold")

    # Draw GT boxes
    for j, box in enumerate(gt_boxes):
        matched = gt_matched_by[j] is not None
        ec = "green" if matched else "#888888"
        lw = 2.5 if matched else 1.5
        alpha_fill = 0.15 if matched else 0.05
        rect = plt.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=lw, edgecolor=ec, facecolor=ec,
            alpha=alpha_fill, linestyle="--",
        )
        ax.add_patch(rect)
        status = f"(matched P{gt_matched_by[j]})" if matched else ""
        ax.text(box[0] + 2, box[1] - 4, f"GT {j} {status}",
                color=ec, fontsize=9, fontweight="bold")

    # Draw prediction boxes
    color_map = {"TP": "blue", "FP": "red", None: "#cccccc"}
    for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
        res = results[i]
        ec = color_map[res]
        lw = 1.5
        alpha_fill = 0.08
        rect = plt.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=lw, edgecolor=ec, facecolor=ec, alpha=alpha_fill,
        )
        ax.add_patch(rect)
        label = f"P{i} ({score:.2f})"
        if res is not None:
            label += f" {res}"
        ax.text(box[2] - 2, box[3] + 12, label,
                color=ec, fontsize=8, ha="right", fontweight="bold")

    # Highlight current prediction
    if step > 0:
        cur_pred = sorted_indices[step - 1]
        cur_box = pred_boxes[cur_pred]
        highlight = plt.Rectangle(
            (cur_box[0] - 2, cur_box[1] - 2),
            cur_box[2] - cur_box[0] + 4, cur_box[3] - cur_box[1] + 4,
            linewidth=3, edgecolor="gold", facecolor="none",
            linestyle="-", zorder=10,
        )
        ax.add_patch(highlight)

        # Draw IoU lines from current prediction to all GT boxes
        info = match_info[cur_pred]
        pc = _box_center(cur_box)
        for j, gt_box in enumerate(gt_boxes):
            iou_val = iou_matrix[cur_pred, j]
            if iou_val < 0.001:
                continue
            gc = _box_center(gt_box)
            is_best = (j == info["best_gt"])
            line_color = "blue" if is_best else "#bbbbbb"
            line_lw = 2.0 if is_best else 0.8
            ax.plot([pc[0], gc[0]], [pc[1], gc[1]],
                    color=line_color, lw=line_lw, linestyle=":", zorder=8)
            mx, my = (pc[0] + gc[0]) / 2, (pc[1] + gc[1]) / 2
            ax.text(mx, my - 3, f"{iou_val:.2f}",
                    fontsize=8, color=line_color, ha="center",
                    fontweight="bold" if is_best else "normal",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.85),
                    zorder=9)

    # TP arrows
    for j, matched_pred in enumerate(gt_matched_by):
        if matched_pred is not None:
            pc = _box_center(pred_boxes[matched_pred])
            gc = _box_center(gt_boxes[j])
            ax.annotate("", xy=gc, xytext=pc,
                        arrowprops=dict(arrowstyle="->", color="blue",
                                        lw=1.5, linestyle="--"),
                        zorder=7)

    # Legend
    ax.legend(
        [
            mpatches.Patch(edgecolor="green", facecolor="green",
                           alpha=0.15, linestyle="--"),
            mpatches.Patch(edgecolor="blue", facecolor="blue", alpha=0.15),
            mpatches.Patch(edgecolor="red", facecolor="red", alpha=0.15),
            mpatches.Patch(edgecolor="#cccccc", facecolor="#cccccc", alpha=0.15),
            mpatches.Patch(edgecolor="gold", facecolor="none", linewidth=2),
        ],
        ["GT (matched)", "TP", "FP", "Not yet processed", "Current step"],
        loc="lower right", fontsize=8, framealpha=0.9,
    )


def _draw_matrix_panel(ax, iou_matrix, pred_scores, sorted_indices, step,
                       results, gt_matched_by, match_info, iou_threshold):
    """Right panel: IoU heatmap with matching overlays."""
    n_pred, n_gt = iou_matrix.shape

    # Reorder rows by confidence (descending)
    display_order = list(sorted_indices)
    ordered_matrix = iou_matrix[display_order]

    # Light colormap for IoU values
    cmap = LinearSegmentedColormap.from_list(
        "iou_cmap", ["#f7f7f7", "#fee08b", "#d9ef8b", "#66bd63"], N=256,
    )
    ax.imshow(ordered_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Cell text
    for row in range(n_pred):
        for col in range(n_gt):
            val = ordered_matrix[row, col]
            ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if val < 0.7 else "white")

    # Overlay: processed cells
    for s in range(min(step, len(sorted_indices))):
        pred_idx = sorted_indices[s]
        info = match_info[pred_idx]
        if info is None:
            continue
        display_row = s  # row in the display (sorted by confidence)
        best_gt = info["best_gt"]

        if results[pred_idx] == "TP":
            # Green border on the matched cell
            rect = plt.Rectangle(
                (best_gt - 0.48, display_row - 0.48), 0.96, 0.96,
                linewidth=3, edgecolor="blue", facecolor="blue",
                alpha=0.25, zorder=5,
            )
            ax.add_patch(rect)
        elif results[pred_idx] == "FP":
            # Red X on the best cell
            rect = plt.Rectangle(
                (best_gt - 0.48, display_row - 0.48), 0.96, 0.96,
                linewidth=2. , edgecolor="red", facecolor="red",
                alpha=0.15, zorder=5,
            )
            ax.add_patch(rect)
            ax.plot([best_gt - 0.35, best_gt + 0.35],
                    [display_row - 0.35, display_row + 0.35],
                    color="red", lw=2, zorder=6)
            ax.plot([best_gt - 0.35, best_gt + 0.35],
                    [display_row + 0.35, display_row - 0.35],
                    color="red", lw=2, zorder=6)

    # Column overlay for claimed GT
    for j, matched_pred in enumerate(gt_matched_by):
        if matched_pred is not None:
            rect = plt.Rectangle(
                (j - 0.5, -0.5), 1, n_pred,
                linewidth=0, facecolor="blue", alpha=0.06, zorder=1,
            )
            ax.add_patch(rect)

    # Highlight current row
    if step > 0:
        cur_row = step - 1
        rect = plt.Rectangle(
            (-0.5, cur_row - 0.5), n_gt, 1,
            linewidth=2.5, edgecolor="gold", facecolor="gold",
            alpha=0.12, zorder=4,
        )
        ax.add_patch(rect)

    # Threshold line annotation
    ax.set_title(f"IoU Matrix  (threshold = {iou_threshold:.2f})",
                 fontsize=13, fontweight="bold")

    # Axis labels
    ax.set_xticks(range(n_gt))
    ax.set_xticklabels([f"GT {j}" for j in range(n_gt)], fontsize=10)
    ax.set_yticks(range(n_pred))
    row_labels = []
    for s, pred_idx in enumerate(sorted_indices):
        lbl = f"P{pred_idx} ({pred_scores[pred_idx]:.2f})"
        res = results[pred_idx]
        if res is not None:
            lbl += f"  {res}"
        row_labels.append(lbl)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")


def _box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


# ------------------------------------------------------------------
# Scoreboard text
# ------------------------------------------------------------------

def _scoreboard_text(step, sorted_indices, results, gt_matched_by,
                     match_info, pred_scores, n_gt):
    """Build a multi-line status string."""
    lines = []
    tp = sum(1 for r in results if r == "TP")
    fp = sum(1 for r in results if r == "FP")
    fn = sum(1 for m in gt_matched_by if m is None)
    processed = sum(1 for r in results if r is not None)

    lines.append(f"Processed: {processed}/{len(results)}   |   "
                 f"TP: {tp}   FP: {fp}   FN: {fn}")

    if step > 0:
        cur_pred = sorted_indices[step - 1]
        info = match_info[cur_pred]
        if info is not None:
            lines.append(
                f"Step {step}: P{cur_pred} (conf {pred_scores[cur_pred]:.2f})  "
                f"→  best overlap GT {info['best_gt']} "
                f"(IoU {info['best_iou']:.2f})  →  {info['reason']}"
            )
    else:
        lines.append("Press Next or move the slider to start matching.")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Public widget
# ------------------------------------------------------------------

def greedy_matching_demo(
    gt_boxes=None,
    pred_boxes=None,
    pred_scores=None,
    iou_threshold=0.5,
):
    """Launch an interactive step-through of VOC-style greedy matching.

    Parameters
    ----------
    gt_boxes : (M, 4) array, default 3 demo boxes
    pred_boxes : (N, 4) array, default 5 demo boxes
    pred_scores : (N,) array, default demo scores
    iou_threshold : float
    """
    # Default demo data
    if gt_boxes is None:
        gt_boxes = np.array([
            [20,  20,  80,  80],    # GT 0
            [120, 30,  200, 90],    # GT 1
            [50,  130, 110, 190],   # GT 2
        ])
    if pred_boxes is None:
        pred_boxes = np.array([
            [18,  22,  78,  82],    # P0 – good match for GT 0
            [115, 28,  195, 88],    # P1 – good match for GT 1
            [25,  25,  70,  70],    # P2 – duplicate of GT 0
            [200, 200, 250, 250],   # P3 – false alarm
            [55,  135, 108, 185],   # P4 – good match for GT 2
        ])
    if pred_scores is None:
        pred_scores = np.array([0.95, 0.87, 0.82, 0.70, 0.60])

    gt_boxes = np.asarray(gt_boxes)
    pred_boxes = np.asarray(pred_boxes)
    pred_scores = np.asarray(pred_scores)

    n_pred = len(pred_boxes)
    sorted_indices = np.argsort(-pred_scores)
    iou_matrix = _build_iou_matrix(pred_boxes, gt_boxes)

    # --- Widgets ---
    step_slider = widgets.IntSlider(
        value=0, min=0, max=n_pred, step=1,
        description="Step:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="500px"),
    )
    btn_prev = widgets.Button(description="◀ Prev", layout=widgets.Layout(width="80px"))
    btn_next = widgets.Button(description="Next ▶", layout=widgets.Layout(width="80px"))
    btn_reset = widgets.Button(description="Reset", layout=widgets.Layout(width="80px"))
    status_label = widgets.HTML(value="", layout=widgets.Layout(width="700px"))
    output = widgets.Output()

    def _render(step):
        results, gt_matched_by, match_info = _run_matching(
            iou_matrix, sorted_indices, iou_threshold, step,
        )
        with output:
            clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            _draw_box_panel(
                axes[0], gt_boxes, pred_boxes, pred_scores,
                sorted_indices, step, results, gt_matched_by,
                match_info, iou_matrix, iou_threshold,
            )
            _draw_matrix_panel(
                axes[1], iou_matrix, pred_scores, sorted_indices, step,
                results, gt_matched_by, match_info, iou_threshold,
            )
            fig.suptitle(
                f"Greedy Matching Step-Through  (IoU ≥ {iou_threshold})",
                fontsize=15, fontweight="bold", y=1.01,
            )
            plt.tight_layout()
            plt.show()

        status_label.value = (
            "<pre style='font-family:monospace; font-size:13px; "
            "line-height:1.5; margin:4px 0 0 0;'>"
            + _scoreboard_text(
                step, sorted_indices, results, gt_matched_by,
                match_info, pred_scores, len(gt_boxes),
            )
            + "</pre>"
        )

    def _on_slider(_=None):
        _render(step_slider.value)

    def _on_next(_=None):
        if step_slider.value < n_pred:
            step_slider.value += 1

    def _on_prev(_=None):
        if step_slider.value > 0:
            step_slider.value -= 1

    def _on_reset(_=None):
        step_slider.value = 0

    step_slider.observe(_on_slider, names="value")
    btn_next.on_click(_on_next)
    btn_prev.on_click(_on_prev)
    btn_reset.on_click(_on_reset)

    # Initial render
    _render(0)

    controls = widgets.HBox(
        [btn_prev, btn_next, btn_reset, step_slider],
        layout=widgets.Layout(align_items="center", gap="8px"),
    )
    display(widgets.VBox([controls, status_label, output]))
