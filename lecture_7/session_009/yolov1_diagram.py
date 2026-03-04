"""
Visualise the YOLOv1 grid + prediction tensor for a teaching notebook.

The main diagram (``plot_yolov1_grid_and_tensor``) tells a three-panel story:

    Panel 1 – "Where?"    Image with the S×S grid.  One ground-truth object is
                           shown; its centre falls in a highlighted cell.
    Panel 2 – "What?"     That cell's 30-value prediction vector, with concrete
                           example numbers for each group of values.
    Panel 3 – "Result"    The decoded bounding box drawn back on the image.

This makes the full flow visible: image → grid cell → numbers → box.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec


# ─── Diagram: three-panel YOLOv1 story ───────────────────────────────────────

def plot_yolov1_grid_and_tensor(
    image: np.ndarray | None = None,
    S: int = 7, B: int = 2, C: int = 20,
    gt_box: tuple[int, int, int, int] | None = None,
    gt_class: str | None = None,
):
    """Three-panel figure explaining YOLOv1's grid → tensor → box flow.

    Parameters
    ----------
    image : np.ndarray or None
        HxWx3 RGB image.  If *None* a solid-colour placeholder is used.
    S, B, C : int
        Grid size, boxes per cell, number of classes (defaults 7, 2, 20).
    gt_box : tuple (x_min, y_min, x_max, y_max) or None
        Ground-truth bounding box in **pixel** coordinates.  When provided
        the diagram uses real coordinates derived from this box instead of
        made-up numbers.
    gt_class : str or None
        Class name for the ground-truth box (e.g. ``"train"``).
    """
    if image is None:
        image = np.full((448, 448, 3), 200, dtype=np.uint8)

    h, w = image.shape[:2]
    cell_w, cell_h = w / S, h / S

    # ── Derive YOLO-normalised values from the GT box (or use defaults) ──
    if gt_box is not None:
        bx1, by1, bx2, by2 = gt_box
        gt_cx = (bx1 + bx2) / 2
        gt_cy = (by1 + by2) / 2
        gt_w  = bx2 - bx1
        gt_h  = by2 - by1

        hi_c = min(int(gt_cx / cell_w), S - 1)
        hi_r = min(int(gt_cy / cell_h), S - 1)

        obj_cx_frac = (gt_cx / cell_w) - hi_c   # offset within cell
        obj_cy_frac = (gt_cy / cell_h) - hi_r
        obj_w_frac  = gt_w / w                   # fraction of image
        obj_h_frac  = gt_h / h
        obj_class   = gt_class or "object"
    else:
        # Fallback: hardcoded demo values (dog in cell 3,3)
        hi_r, hi_c = 3, 3
        obj_cx_frac, obj_cy_frac = 0.55, 0.45
        obj_w_frac, obj_h_frac   = 0.32, 0.44
        obj_class = "dog"

    obj_conf       = 0.92          # simulated confidence (always fake)
    obj_class_prob = 0.95          # simulated P(class|obj)

    # Absolute pixel coords of the predicted box (= GT when gt_box given)
    abs_cx = (hi_c + obj_cx_frac) * cell_w
    abs_cy = (hi_r + obj_cy_frac) * cell_h
    abs_w  = obj_w_frac * w
    abs_h  = obj_h_frac * h
    x1, y1 = abs_cx - abs_w / 2, abs_cy - abs_h / 2

    # Second (weaker) box — perturbed version for illustration
    box2_cx_frac = max(0.0, obj_cx_frac - 0.25)
    box2_cy_frac = min(1.0, obj_cy_frac + 0.15)
    box2_w_frac  = obj_w_frac * 0.5
    box2_h_frac  = obj_h_frac * 0.5
    box2_conf    = 0.12

    # ── Layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 6.5))
    gs = GridSpec(1, 3, width_ratios=[1, 1.4, 1], wspace=0.28, figure=fig)

    # Colours
    COL_CELL  = "#e91e63"
    COL_BOX1  = "#2196f3"
    COL_BOX2  = "#42a5f5"
    COL_CLASS = "#ff9800"
    COL_PRED  = "#4caf50"
    COL_DOT   = "#ffeb3b"

    # =====================================================================
    # Panel 1  –  "Where?" : image + grid + object centre → responsible cell
    # =====================================================================
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(image)
    ax1.set_title("① Grid & cell assignment", fontsize=12, fontweight="bold",
                  pad=8)

    for i in range(1, S):
        ax1.axhline(i * cell_h, color="white", lw=0.5, alpha=0.55)
        ax1.axvline(i * cell_w, color="white", lw=0.5, alpha=0.55)

    ax1.add_patch(patches.Rectangle(
        (hi_c * cell_w, hi_r * cell_h), cell_w, cell_h,
        lw=2.5, edgecolor=COL_CELL,
        facecolor=to_rgba(COL_CELL, 0.22)))

    ax1.plot(abs_cx, abs_cy, marker="o", ms=10, color=COL_DOT,
             markeredgecolor="black", markeredgewidth=1.2, zorder=5)

    # Place annotation so it doesn't fall off-screen
    txt_x = abs_cx + cell_w * 1.6
    txt_y = abs_cy - cell_h * 1.8
    # Clamp within image bounds
    txt_x = min(txt_x, w - cell_w * 2)
    txt_y = max(txt_y, cell_h * 1.5)
    ax1.annotate(
        "object centre\n→ cell responsible",
        xy=(abs_cx, abs_cy),
        xytext=(txt_x, txt_y),
        fontsize=8, color="white", ha="left",
        arrowprops=dict(arrowstyle="->", color="white", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.3", fc=COL_CELL, alpha=0.88))

    ax1.set_xlim(0, w); ax1.set_ylim(h, 0)
    ax1.set_xticks([]); ax1.set_yticks([])

    # =====================================================================
    # Panel 2  –  "What?" : the 30-value vector with concrete numbers
    # =====================================================================
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title(
        f"② What cell ({hi_r},{hi_c}) predicts  —  "
        f"{B*5+C} values", fontsize=12, fontweight="bold", pad=8)
    ax2.axis("off")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    total = B * 5 + C

    # --- Segmented colour bar (proportional widths) --------------------
    bar_left, bar_right = 0.06, 0.94
    bar_w = bar_right - bar_left
    scale = bar_w / total

    y_bar  = 0.78          # top of bar
    h_bar  = 0.10          # bar height

    sections = [
        ("Box 1", 5, COL_BOX1),
        ("Box 2", 5, COL_BOX2),
        (f"C={C} class probs", C, COL_CLASS),
    ]

    x_cur = bar_left
    for label, count, colour in sections:
        bw = count * scale
        rect = patches.FancyBboxPatch(
            (x_cur, y_bar), bw, h_bar,
            boxstyle="round,pad=0.005",
            fc=colour, ec="white", lw=1.5, alpha=0.88)
        ax2.add_patch(rect)

        # Group label inside
        ax2.text(x_cur + bw / 2, y_bar + h_bar / 2, label,
                 ha="center", va="center", fontsize=7,
                 fontweight="bold", color="white")

        # Index range just below
        idx0 = round((x_cur - bar_left) / scale)
        idx1 = idx0 + count - 1
        ax2.text(x_cur + bw / 2, y_bar - 0.02,
                 f"[{idx0}–{idx1}]",
                 ha="center", va="top", fontsize=6.5, color="#666")
        x_cur += bw

    # --- Totals line beneath the index ranges -------------------------
    ax2.text(0.50, y_bar - 0.08,
             f"B×5 + C  =  {B}×5 + {C}  =  {total} values per cell     "
             f"(full tensor: {S}×{S}×{total} = {S*S*total:,})",
             ha="center", va="top", fontsize=8.5, color="#555")

    # --- Concrete values – structured table below the bar -------------
    tbl_top = 0.58          # y where the value table starts

    def _val_line(y, colour, text, fontsize=8.5):
        ax2.text(0.06, y, text, va="top", fontsize=fontsize,
                 color=colour, family="monospace")

    _val_line(tbl_top, COL_BOX1,
              f"Box 1:  x={obj_cx_frac:.2f}   y={obj_cy_frac:.2f}   "
              f"w={obj_w_frac:.2f}   h={obj_h_frac:.2f}   "
              f"conf={obj_conf}")
    _val_line(tbl_top - 0.08, COL_BOX2,
              f"Box 2:  x={box2_cx_frac:.2f}   y={box2_cy_frac:.2f}   "
              f"w={box2_w_frac:.2f}   h={box2_h_frac:.2f}   "
              f"conf={box2_conf}")
    _val_line(tbl_top - 0.16, COL_CLASS,
              f"Classes: P({obj_class})={obj_class_prob}   "
              f"P(cat)=0.02   ...   (20 values)")

    # --- Detection-score formula at the bottom ------------------------
    score = obj_conf * obj_class_prob
    ax2.text(0.50, 0.12,
             f"Detection score  =  conf × P(class|obj)  =  "
             f"{obj_conf} × {obj_class_prob}  =  {score:.2f}"
             f"   \"{obj_class}\"",
             ha="center", va="center", fontsize=10, color="#333",
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.35",
                       fc=to_rgba(COL_PRED, 0.15),
                       ec=COL_PRED, lw=1.5))

    # =====================================================================
    # Panel 3  –  "Result" : decoded box drawn on the image
    # =====================================================================
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(image)
    ax3.set_title("③ Decoded bounding box", fontsize=12,
                  fontweight="bold", pad=8)

    for i in range(1, S):
        ax3.axhline(i * cell_h, color="white", lw=0.3, alpha=0.3)
        ax3.axvline(i * cell_w, color="white", lw=0.3, alpha=0.3)

    ax3.add_patch(patches.Rectangle(
        (x1, y1), abs_w, abs_h,
        lw=3, edgecolor=COL_PRED, facecolor="none"))

    ax3.text(x1, y1 - 4,
             f'{obj_class}  {score:.2f}',
             fontsize=10, color="white", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", fc=COL_PRED, alpha=0.9))

    ax3.plot(abs_cx, abs_cy, marker="+", ms=12, mew=2, color=COL_DOT,
             zorder=5)

    ax3.set_xlim(0, w); ax3.set_ylim(h, 0)
    ax3.set_xticks([]); ax3.set_yticks([])

    plt.tight_layout()
    plt.show()


# ─── Decoder: raw (S,S,30) tensor → boxes, scores, class IDs ─────────────────

def decode_yolov1_tensor(prediction: np.ndarray,
                          S: int = 7, B: int = 2, C: int = 20,
                          conf_thresh: float = 0.3,
                          image_hw: tuple[int, int] = (448, 448)):
    """Decode a raw YOLOv1 output tensor into boxes, scores, and labels.

    This is a **teaching implementation** — it mirrors the maths from the
    lecture rather than optimising for speed.

    Parameters
    ----------
    prediction : np.ndarray, shape (S, S, B*5+C)
        Raw (linear) network output.
    S, B, C : int
        Grid size, boxes per cell, number of classes.
    conf_thresh : float
        Minimum *class-specific confidence* to keep a detection.
    image_hw : (int, int)
        (height, width) of the original image, used to scale boxes.

    Returns
    -------
    boxes : np.ndarray (N, 4)  — [x1, y1, x2, y2] in pixel coordinates
    scores : np.ndarray (N,)
    class_ids : np.ndarray (N,)
    """
    img_h, img_w = image_hw
    cell_h, cell_w = img_h / S, img_w / S

    all_boxes, all_scores, all_classes = [], [], []

    for row in range(S):
        for col in range(S):
            cell = prediction[row, col]                       # shape (B*5+C,)
            class_probs = cell[B * 5:]                        # shape (C,)

            for b in range(B):
                offset = b * 5
                x, y, w, h, conf = cell[offset:offset + 5]

                # Class-specific confidence = P(Class_i | Obj) × Conf
                class_scores = class_probs * conf             # shape (C,)
                best_class = int(np.argmax(class_scores))
                best_score = class_scores[best_class]

                if best_score < conf_thresh:
                    continue

                # Convert to absolute pixel coordinates
                cx = (col + x) * cell_w     # x is relative to cell
                cy = (row + y) * cell_h     # y is relative to cell
                bw = w * img_w              # w is relative to image
                bh = h * img_h              # h is relative to image

                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(best_score)
                all_classes.append(best_class)

    if len(all_boxes) == 0:
        return (np.empty((0, 4)),
                np.empty((0,)),
                np.empty((0,), dtype=int))

    return (np.array(all_boxes),
            np.array(all_scores),
            np.array(all_classes, dtype=int))


# ─── Exercise verification plot ──────────────────────────────────────────────

def decode_and_plot(
    tensor,
    image: np.ndarray,
    *,
    S: int = 7,
    B: int = 2,
    C: int = 20,
    img_hw: tuple[int, int] = (448, 448),
    class_names: list[str] | None = None,
    gt_box: tuple[float, float, float, float] | None = None,
    gt_class_name: str | None = None,
):
    """Decode a (S, S, B*5+C) YOLOv1 target tensor and overlay the result.

    Parameters
    ----------
    tensor : array-like, shape (S, S, B*5+C)
        The encoded YOLOv1 target tensor (e.g. built by a student exercise).
    image : ndarray, shape (H, W, 3)
        The original image (any size — will be resized for display).
    S, B, C : int
        Grid size, boxes per cell, number of classes.
    img_hw : (int, int)
        The (height, width) of the YOLOv1 coordinate system (default 448×448).
    class_names : list[str] or None
        Class name list; indices must match the tensor's class-probability order.
    gt_box : (x_min, y_min, x_max, y_max) or None
        Ground-truth box in ``img_hw`` coordinates for comparison overlay.
    gt_class_name : str or None
        Class label for the ground-truth box.
    """
    import cv2
    import matplotlib.patches as mpatches

    img_h, img_w = img_hw
    ch, cw = img_h / S, img_w / S
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Resize image to target coordinate system for display
    disp = cv2.resize(image, (img_w, img_h))
    ax.imshow(disp)

    # Draw grid
    for i in range(1, S):
        ax.axhline(i * ch, color='white', lw=0.4, alpha=0.4)
        ax.axvline(i * cw, color='white', lw=0.4, alpha=0.4)

    found = False
    for r in range(S):
        for c_idx in range(S):
            cell = tensor[r, c_idx]
            class_probs = cell[B * 5:]
            for b_idx in range(B):
                bx, by, bw, bh, conf = cell[b_idx * 5 : b_idx * 5 + 5]
                scores = class_probs * conf
                best_cls = int(scores.argmax()) if hasattr(scores, 'argmax') else int(np.argmax(scores))
                best_score = float(scores[best_cls])
                if best_score < 0.3:
                    continue
                found = True
                abs_cx = (c_idx + float(bx)) * cw
                abs_cy = (r + float(by)) * ch
                abs_w  = float(bw) * img_w
                abs_h  = float(bh) * img_h
                x1, y1 = abs_cx - abs_w / 2, abs_cy - abs_h / 2

                label = (f'{class_names[best_cls]} {best_score:.2f}'
                         if class_names else f'cls{best_cls} {best_score:.2f}')
                ax.add_patch(mpatches.Rectangle(
                    (x1, y1), abs_w, abs_h,
                    lw=2.5, edgecolor='lime', facecolor='none'))
                ax.text(x1, y1 - 3, label,
                        fontsize=9, color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  fc='green', alpha=0.85))

    # Draw ground-truth box in red for comparison (if provided)
    if gt_box is not None:
        gx_min, gy_min, gx_max, gy_max = gt_box
        ax.add_patch(mpatches.Rectangle(
            (gx_min, gy_min), gx_max - gx_min, gy_max - gy_min,
            lw=2, edgecolor='red', facecolor='none', linestyle='--'))
        gt_label = f'GT: {gt_class_name}' if gt_class_name else 'GT'
        ax.text(gx_min, gy_max + 12, gt_label,
                fontsize=9, color='white',
                bbox=dict(boxstyle='round,pad=0.2', fc='red', alpha=0.85))

    ax.set_xlim(0, img_w); ax.set_ylim(img_h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Green = your encoding → decoded  |  Red dashed = ground truth',
                 fontsize=10)
    if not found:
        ax.text(img_w / 2, img_h / 2,
                'No detection found!\nCheck your TODOs.',
                ha='center', va='center', fontsize=14, color='yellow',
                fontweight='bold',
                bbox=dict(boxstyle='round', fc='black', alpha=0.7))
    plt.tight_layout()
    plt.show()
