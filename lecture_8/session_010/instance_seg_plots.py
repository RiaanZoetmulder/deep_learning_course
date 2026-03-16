"""Instance segmentation evaluation visualizations.

Functions
---------
plot_instance_maps            Side-by-side GT vs predicted instance label maps.
plot_mask_iou_computation     4-panel walkthrough of mask IoU for one pair.
plot_mask_iou_matrix          IoU matrix heatmap with greedy matching overlay.
plot_pq_diagnostic            2x3 grid: SQ vs RQ diagnostic for two scenarios.
run_pq_diagnostic_demo        End-to-end PQ diagnostic: data, metrics, plot, summary.
plot_per_instance_dsc         Histogram + bar chart of per-instance DSC values.
plot_exercise_instances       Side-by-side GT vs predicted for exercise scenario.
make_circle_mask              Binary circle mask on an (H, W) canvas.
panoptic_quality              Compute PQ, SQ, RQ from instance mask lists.
run_mask_vs_box_iou_demo      Mask IoU vs Box IoU comparison on synthetic data.
run_instance_matching_demo    Greedy matching walkthrough on synthetic data.
run_pq_factorization_grid     2x2 grid: four SQ/RQ combinations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion


# ------------------------------------------------------------------
# 1. Instance label maps (1x2)
# ------------------------------------------------------------------

def plot_instance_maps(gt_instances, pred_instances, H, W):
    """Side-by-side colored label maps of GT and predicted instances.

    Parameters
    ----------
    gt_instances   : list of 2D binary masks (one per GT instance).
    pred_instances : list of 2D binary masks (one per predicted instance).
    H, W           : int, image dimensions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    gt_label_map = np.zeros((H, W))
    for i, m in enumerate(gt_instances):
        gt_label_map[m > 0] = i + 1

    pred_label_map = np.zeros((H, W))
    for i, m in enumerate(pred_instances):
        pred_label_map[m > 0] = i + 1

    n_max = max(len(gt_instances), len(pred_instances))
    cmap_inst = plt.cm.get_cmap("tab10", n_max + 1)

    axes[0].imshow(gt_label_map, cmap=cmap_inst, vmin=0, vmax=n_max)
    axes[0].set_title(
        f"Ground Truth: {len(gt_instances)} instances", fontsize=13, fontweight="bold"
    )
    axes[0].axis("off")
    for i, m in enumerate(gt_instances):
        ys, xs = np.where(m > 0)
        axes[0].text(
            xs.mean(), ys.mean(), f"GT {i}", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
        )

    axes[1].imshow(pred_label_map, cmap=cmap_inst, vmin=0, vmax=n_max)
    axes[1].set_title(
        f"Prediction: {len(pred_instances)} instances", fontsize=13, fontweight="bold"
    )
    axes[1].axis("off")
    for i, m in enumerate(pred_instances):
        ys, xs = np.where(m > 0)
        if len(ys) > 0:
            axes[1].text(
                xs.mean(), ys.mean(), f"P{i}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
            )

    fig.suptitle(
        "Instance Segmentation: Ground Truth vs Prediction",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 2. Mask IoU computation walkthrough (1×4)
# ------------------------------------------------------------------

def plot_mask_iou_computation(pred_mask, gt_mask,
                              pred_label="P0", gt_label="GT 0"):
    """Four-panel figure showing how mask IoU is computed for one pair.

    Panels: GT mask | Pred mask | Intersection (green) | Union (purple).
    The suptitle shows the IoU formula with concrete pixel counts.
    """
    pred_b = pred_mask.astype(bool)
    gt_b = gt_mask.astype(bool)

    intersection = pred_b & gt_b
    union = pred_b | gt_b

    n_gt = int(np.sum(gt_b))
    n_pred = int(np.sum(pred_b))
    n_inter = int(np.sum(intersection))
    n_union = int(np.sum(union))
    iou = n_inter / n_union if n_union > 0 else 0.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    # Panel 1 – GT mask
    vis = np.zeros((*gt_mask.shape, 3))
    vis[gt_b] = [0.25, 0.55, 0.85]
    axes[0].imshow(vis)
    axes[0].set_title(f"{gt_label}\n{n_gt} pixels", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2 – Pred mask
    vis = np.zeros((*pred_mask.shape, 3))
    vis[pred_b] = [0.90, 0.50, 0.20]
    axes[1].imshow(vis)
    axes[1].set_title(f"{pred_label}\n{n_pred} pixels", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Panel 3 – Intersection
    vis = np.zeros((*gt_mask.shape, 3))
    vis[gt_b] = [0.82, 0.90, 1.0]           # light blue (GT-only region)
    vis[pred_b] = [1.0, 0.90, 0.82]         # light orange (Pred-only region)
    vis[intersection] = [0.18, 0.75, 0.18]  # green (overlap)
    axes[2].imshow(vis)
    axes[2].set_title(
        f"Intersection\n|P ∩ G| = {n_inter} pixels",
        fontsize=12, fontweight="bold", color="#2d8a2d",
    )
    axes[2].axis("off")

    # Panel 4 – Union
    vis = np.zeros((*gt_mask.shape, 3))
    vis[union] = [0.55, 0.27, 0.62]  # purple
    axes[3].imshow(vis)
    axes[3].set_title(
        f"Union\n|P ∪ G| = {n_union} pixels",
        fontsize=12, fontweight="bold", color="#8e44ad",
    )
    axes[3].axis("off")

    fig.suptitle(
        f"Mask IoU({pred_label}, {gt_label})  =  "
        f"|P ∩ G| / |P ∪ G|  =  {n_inter} / {n_union}  =  {iou:.4f}",
        fontsize=14, fontweight="bold", y=0.02,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()


# ------------------------------------------------------------------
# 3. IoU matrix + greedy matching heatmap
# ------------------------------------------------------------------

def plot_mask_iou_matrix(gt_instances, pred_instances, pred_scores,
                         iou_thresh=0.5):
    """IoU-matrix heatmap with greedy-matching overlay.

    Rows are predictions sorted by descending confidence; columns are GT
    instances.  Matched cells get a green border and ✓ label; the step-by-step
    matching transcript is printed below the figure.

    Parameters
    ----------
    gt_instances   : list of 2-D binary masks.
    pred_instances : list of 2-D binary masks.
    pred_scores    : array-like, confidence per prediction.
    iou_thresh     : float, matching threshold (default 0.5).
    """
    scores = np.asarray(pred_scores, dtype=float)
    n_pred = len(pred_instances)
    n_gt = len(gt_instances)

    # ---- 1. Compute IoU matrix ----
    iou_matrix = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        pi = pred_instances[i].astype(bool)
        for j in range(n_gt):
            gj = gt_instances[j].astype(bool)
            inter = np.sum(pi & gj)
            union = np.sum(pi | gj)
            iou_matrix[i, j] = inter / union if union > 0 else 0.0

    # ---- 2. Greedy matching (replay for transcript) ----
    order = np.argsort(-scores)
    gt_matched = np.zeros(n_gt, dtype=bool)
    matches = []        # (pred_idx, gt_idx, iou)
    fp_list = []        # (pred_idx, reason_str)

    for idx in order:
        best_iou, best_gt = 0.0, -1
        for j in range(n_gt):
            if gt_matched[j]:
                continue
            if iou_matrix[idx, j] > best_iou:
                best_iou = iou_matrix[idx, j]
                best_gt = j

        if best_iou >= iou_thresh and best_gt >= 0:
            matches.append((int(idx), int(best_gt), best_iou))
            gt_matched[best_gt] = True
        else:
            raw_max = iou_matrix[idx].max()
            if raw_max == 0:
                reason = "no overlap with any GT"
            elif best_gt >= 0:
                reason = (f"best unmatched GT {best_gt} has "
                          f"IoU={best_iou:.3f} < {iou_thresh}")
            else:
                raw_j = int(iou_matrix[idx].argmax())
                reason = f"GT {raw_j} already matched"
            fp_list.append((int(idx), reason))

    fn_indices = [j for j in range(n_gt) if not gt_matched[j]]

    # ---- 3. Heatmap (rows = preds in confidence order) ----
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * n_gt), max(4, 1.4 * n_pred)))
    display_matrix = iou_matrix[order]

    im = ax.imshow(display_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    for rank, i in enumerate(order):
        for j in range(n_gt):
            val = iou_matrix[i, j]
            is_match = any(m[0] == i and m[1] == j for m in matches)
            color = "white" if val > 0.5 else "black"
            weight = "bold" if is_match else "normal"
            label = f"{val:.3f}"
            if is_match:
                label += "\n✓ TP"
            ax.text(j, rank, label, ha="center", va="center",
                    fontsize=11, fontweight=weight, color=color)

    for pi, gi, _ in matches:
        rank = list(order).index(pi)
        rect = plt.Rectangle(
            (gi - 0.5, rank - 0.5), 1, 1,
            linewidth=3, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)

    ax.set_xticks(range(n_gt))
    ax.set_xticklabels([f"GT {j}" for j in range(n_gt)], fontsize=11)
    ax.set_yticks(range(n_pred))
    ax.set_yticklabels(
        [f"P{i}  (conf={scores[i]:.2f})" for i in order], fontsize=11,
    )
    ax.set_xlabel("Ground Truth Instances", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predictions (by confidence ↓)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Mask IoU Matrix  –  Greedy Matching (τ = {iou_thresh})",
        fontsize=14, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mask IoU")
    plt.tight_layout()
    plt.show()

    # ---- 4. Text transcript ----
    print("Greedy Matching – step by step")
    print("=" * 55)
    for step, idx in enumerate(order, 1):
        match = next((m for m in matches if m[0] == idx), None)
        if match:
            print(f"  Step {step}: P{idx} (conf={scores[idx]:.2f})  →  "
                  f"GT {match[1]} (IoU={match[2]:.3f} ≥ {iou_thresh})  → TP ✓")
        else:
            reason = next(r for p, r in fp_list if p == idx)
            print(f"  Step {step}: P{idx} (conf={scores[idx]:.2f})  →  "
                  f"{reason}  → FP ✗")

    if fn_indices:
        print(f"\n  Unmatched GT: {', '.join(f'GT {j}' for j in fn_indices)}  → FN")

    print(f"\n  Result:  TP = {len(matches)},  "
          f"FP = {len(fp_list)},  FN = {len(fn_indices)}")

    return matches, fp_list, fn_indices, iou_matrix


# ------------------------------------------------------------------
# 4. PQ diagnostic (2x3)
# ------------------------------------------------------------------

def plot_pq_diagnostic(
    gt_diag,
    pred_diag_A,
    pred_diag_B,
    sq_A,
    rq_A,
    pq_A,
    sq_B,
    rq_B,
    pq_B,
    det_A,
    det_B,
    img_shape=(180, 260),
):
    """2x3 grid comparing two failure scenarios with a text score panel.

    Parameters
    ----------
    gt_diag     : list of 2D GT masks.
    pred_diag_A : list of 2D prediction masks (Scenario A: all detected, noisy).
    pred_diag_B : list of 2D prediction masks (Scenario B: few detected, good).
    sq_A, rq_A, pq_A : float, metrics for Scenario A.
    sq_B, rq_B, pq_B : float, metrics for Scenario B.
    det_A, det_B : dict with 'tp', 'fp', 'fn' counts.
    img_shape    : tuple (H, W) for the visualization grid.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                             gridspec_kw={"width_ratios": [1, 1, 0.7]})
    H, W = img_shape

    # Row 1: Scenario A
    gt_map_A = np.zeros((H, W))
    for i, m in enumerate(gt_diag):
        gt_map_A[m > 0] = i + 1
    pred_map_A = np.zeros((H, W))
    for i, m in enumerate(pred_diag_A):
        pred_map_A[m > 0] = i + 1

    axes[0, 0].imshow(gt_map_A, cmap="tab10", vmin=0, vmax=7)
    axes[0, 0].set_title("GT (6 instances)", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pred_map_A, cmap="tab10", vmin=0, vmax=7)
    axes[0, 1].set_title("Scenario A: All detected, noisy masks", fontsize=12)
    axes[0, 1].axis("off")

    # Score panel A
    ax = axes[0, 2]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    lines_A = [
        ("Scenario A", 0.92, 14, "bold", "black"),
        ("Good detection, bad masks", 0.82, 11, "normal", "#555"),
        (f"TP = {det_A['tp']}   FP = {det_A['fp']}   FN = {det_A['fn']}", 0.68, 11, "normal", "black"),
        (f"SQ = {sq_A:.3f}", 0.54, 13, "bold", "#3498db"),
        (f"RQ = {rq_A:.3f}", 0.42, 13, "bold", "#2ecc71"),
        (f"PQ = {pq_A:.3f}", 0.26, 14, "bold", "#9b59b6"),
    ]
    for text, y, fs, fw, color in lines_A:
        ax.text(0.5, y, text, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=color,
                fontfamily="monospace" if "=" in text else "sans-serif")

    # Row 2: Scenario B
    pred_map_B = np.zeros((H, W))
    for i, m in enumerate(pred_diag_B):
        pred_map_B[m > 0] = i + 1

    axes[1, 0].imshow(gt_map_A, cmap="tab10", vmin=0, vmax=7)
    axes[1, 0].set_title("GT (6 instances)", fontsize=12)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred_map_B, cmap="tab10", vmin=0, vmax=7)
    axes[1, 1].set_title("Scenario B: Only 3 detected, good masks", fontsize=12)
    axes[1, 1].axis("off")

    # Score panel B
    ax = axes[1, 2]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    lines_B = [
        ("Scenario B", 0.92, 14, "bold", "black"),
        ("Bad detection, good masks", 0.82, 11, "normal", "#555"),
        (f"TP = {det_B['tp']}   FP = {det_B['fp']}   FN = {det_B['fn']}", 0.68, 11, "normal", "black"),
        (f"SQ = {sq_B:.3f}", 0.54, 13, "bold", "#3498db"),
        (f"RQ = {rq_B:.3f}", 0.42, 13, "bold", "#2ecc71"),
        (f"PQ = {pq_B:.3f}", 0.26, 14, "bold", "#9b59b6"),
    ]
    for text, y, fs, fw, color in lines_B:
        ax.text(0.5, y, text, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=color,
                fontfamily="monospace" if "=" in text else "sans-serif")

    fig.suptitle(
        "PQ Diagnostic: SQ vs RQ Reveals the Source of Errors",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 3. Per-instance DSC histogram + bar chart (1x2)
# ------------------------------------------------------------------

def plot_per_instance_dsc(gt_instances, gt_dscs, mean_gt_dsc, overall_pixel_dsc):
    """Bar chart of per-instance DSC for each GT object.

    Parameters
    ----------
    gt_instances      : list of GT masks (only used for count / x-labels).
    gt_dscs           : 1D array, DSC per GT instance (0 for missed).
    mean_gt_dsc       : float, mean per-instance DSC.
    overall_pixel_dsc : float, overall pixel-level DSC.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(gt_instances))
    colors_bar = ["steelblue" if d > 0 else "firebrick" for d in gt_dscs]
    bars = ax.bar(x, gt_dscs, color=colors_bar, edgecolor="black", width=0.6)
    for b, d in zip(bars, gt_dscs):
        label = f"{d:.3f}" if d > 0 else "MISSED"
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.axhline(
        y=mean_gt_dsc, color="red", linestyle="--", linewidth=1.5,
        label=f"Mean per-instance DSC = {mean_gt_dsc:.3f}",
    )
    ax.axhline(
        y=overall_pixel_dsc, color="green", linestyle="--", linewidth=1.5,
        label=f"Overall pixel DSC = {overall_pixel_dsc:.3f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"GT {i}" for i in range(len(gt_instances))], fontsize=10)
    ax.set_ylabel("DSC", fontsize=12)
    ax.set_title("Per-Instance DSC by GT Object", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 4. Exercise instance visualisation (1x2)
# ------------------------------------------------------------------

def plot_exercise_instances(exercise_gt, exercise_pred, EX_H, EX_W):
    """Side-by-side GT vs predicted label maps for the exercise scenario.

    Parameters
    ----------
    exercise_gt   : list of 2D GT masks.
    exercise_pred : list of 2D prediction masks.
    EX_H, EX_W   : int, image dimensions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    gt_map_ex = np.zeros((EX_H, EX_W))
    for i, m in enumerate(exercise_gt):
        gt_map_ex[m > 0] = i + 1
    pred_map_ex = np.zeros((EX_H, EX_W))
    for i, m in enumerate(exercise_pred):
        pred_map_ex[m > 0] = i + 1

    axes[0].imshow(gt_map_ex, cmap="tab10", vmin=0, vmax=9)
    axes[0].set_title(
        f"Ground Truth: {len(exercise_gt)} cells", fontsize=13, fontweight="bold"
    )
    axes[0].axis("off")
    for i, m in enumerate(exercise_gt):
        ys, xs = np.where(m > 0)
        if len(ys) > 0:
            axes[0].text(
                xs.mean(), ys.mean(), f"GT{i}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6),
            )

    axes[1].imshow(pred_map_ex, cmap="tab10", vmin=0, vmax=9)
    axes[1].set_title(
        f"Predictions: {len(exercise_pred)} instances", fontsize=13, fontweight="bold"
    )
    axes[1].axis("off")
    for i, m in enumerate(exercise_pred):
        ys, xs = np.where(m > 0)
        if len(ys) > 0:
            axes[1].text(
                xs.mean(), ys.mean(), f"P{i}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6),
            )

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 6. Helper: circle mask generator
# ------------------------------------------------------------------

def make_circle_mask(H, W, cy, cx, r):
    """Binary mask with a filled circle of radius *r* at (cy, cx)."""
    yy, xx = np.ogrid[:H, :W]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).astype(np.float64)


# ------------------------------------------------------------------
# 7. Helper: Panoptic Quality computation
# ------------------------------------------------------------------

def panoptic_quality(pred_instances, gt_instances, iou_thresh=0.5):
    """Compute Panoptic Quality and its SQ / RQ factorization.

    Parameters
    ----------
    pred_instances : list of 2-D binary masks.
    gt_instances   : list of 2-D binary masks.
    iou_thresh     : float, IoU threshold for a true-positive match.

    Returns
    -------
    pq, sq, rq : float
    det : dict with keys 'tp', 'fp', 'fn'.
    """
    n_pred = len(pred_instances)
    n_gt = len(gt_instances)

    iou_matrix = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        pi = pred_instances[i].astype(bool)
        for j in range(n_gt):
            gj = gt_instances[j].astype(bool)
            inter = np.sum(pi & gj)
            union = np.sum(pi | gj)
            iou_matrix[i, j] = inter / union if union > 0 else 0.0

    gt_matched = np.zeros(n_gt, dtype=bool)
    pred_matched = np.zeros(n_pred, dtype=bool)
    matched_ious = []

    # Greedy matching by descending IoU
    while True:
        if iou_matrix.size == 0:
            break
        best = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        best_iou = iou_matrix[best]
        if best_iou < iou_thresh:
            break
        pi, gj = best
        matched_ious.append(best_iou)
        pred_matched[pi] = True
        gt_matched[gj] = True
        iou_matrix[pi, :] = 0
        iou_matrix[:, gj] = 0

    tp = len(matched_ious)
    fp = int(n_pred - np.sum(pred_matched))
    fn = int(n_gt - np.sum(gt_matched))

    sq = float(np.mean(matched_ious)) if tp > 0 else 0.0
    rq = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    pq = sq * rq

    return pq, sq, rq, {"tp": tp, "fp": fp, "fn": fn}


# ------------------------------------------------------------------
# 8. End-to-end PQ diagnostic demo
# ------------------------------------------------------------------

def run_pq_diagnostic_demo(seed=99):
    """Generate synthetic data, compute PQ for two scenarios, plot and print.

    Scenario A: all 6 GT instances detected but masks are noisy  (high RQ, low SQ).
    Scenario B: only 3 of 6 detected but masks are near-perfect (low RQ, high SQ).
    """
    rng = np.random.RandomState(seed)
    H, W = 180, 260
    centers = [(40, 40), (40, 130), (40, 220), (130, 40), (130, 130), (130, 220)]

    # Ground truth
    gt_diag = [make_circle_mask(H, W, cy, cx, 20) for cy, cx in centers]

    # Scenario A: all detected, noisy masks
    pred_diag_A = []
    for cy, cx in centers:
        clean = make_circle_mask(H, W, cy, cx, 20)
        noise = (rng.rand(H, W) > 0.23).astype(np.float64)
        noisy = clean * noise
        noisy = binary_dilation(
            binary_erosion(noisy.astype(bool), iterations=1), iterations=1
        ).astype(np.float64)
        shift_mask = np.zeros_like(noisy)
        dy, dx = rng.randint(-3, 4), rng.randint(-3, 4)
        y_s, y_e = max(0, dy), min(H, H + dy)
        x_s, x_e = max(0, dx), min(W, W + dx)
        sy_s, sy_e = max(0, -dy), min(H, H - dy)
        sx_s, sx_e = max(0, -dx), min(W, W - dx)
        shift_mask[y_s:y_e, x_s:x_e] = noisy[sy_s:sy_e, sx_s:sx_e]
        pred_diag_A.append(shift_mask)

    # Scenario B: only 3 detected, near-perfect masks
    pred_diag_B = [
        make_circle_mask(H, W, cy + 1, cx - 1, 20)
        for cy, cx in centers[:3]
    ]

    # Compute PQ
    pq_A, sq_A, rq_A, det_A = panoptic_quality(pred_diag_A, gt_diag)
    pq_B, sq_B, rq_B, det_B = panoptic_quality(pred_diag_B, gt_diag)

    # Visualization
    plot_pq_diagnostic(
        gt_diag, pred_diag_A, pred_diag_B,
        sq_A, rq_A, pq_A, sq_B, rq_B, pq_B,
        det_A, det_B, img_shape=(H, W),
    )

    # Summary table
    print("PQ Diagnostic Summary")
    print("=" * 65)
    print(f"  {'':25s}  {'Scenario A':>12s}  {'Scenario B':>12s}")
    print(f"  {'':25s}  {'(all det, bad masks)':>12s}  {'(3/6 det, good masks)':>12s}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*12}")
    print(f"  {'TP':25s}  {det_A['tp']:>12d}  {det_B['tp']:>12d}")
    print(f"  {'FP':25s}  {det_A['fp']:>12d}  {det_B['fp']:>12d}")
    print(f"  {'FN':25s}  {det_A['fn']:>12d}  {det_B['fn']:>12d}")
    print(f"  {'Segmentation Quality (SQ)':25s}  {sq_A:>12.4f}  {sq_B:>12.4f}")
    print(f"  {'Recognition Quality (RQ)':25s}  {rq_A:>12.4f}  {rq_B:>12.4f}")
    print(f"  {'Panoptic Quality (PQ)':25s}  {pq_A:>12.4f}  {pq_B:>12.4f}")
    print()
    print(" : Scenario A: RQ is high (found all objects) but SQ is low (masks are noisy)")
    print("    Action: improve mask head / post-processing")
    print(" : Scenario B: SQ is high (masks are near-perfect) but RQ is low (missed 3 objects)")
    print("    Action: improve detection backbone / lower confidence threshold")


# ------------------------------------------------------------------
# 9. Per-instance DSC demo
# ------------------------------------------------------------------

def _dice_score(a, b):
    """Dice coefficient between two binary masks."""
    a_b = a.astype(bool)
    b_b = b.astype(bool)
    inter = np.sum(a_b & b_b)
    total = np.sum(a_b) + np.sum(b_b)
    return 2.0 * inter / total if total > 0 else 0.0


def _match_instances_iou(pred_instances, gt_instances, iou_thresh=0.5):
    """Greedy IoU matching. Returns (matches, fp_indices, fn_indices)."""
    n_pred = len(pred_instances)
    n_gt = len(gt_instances)
    iou_mat = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        pi = pred_instances[i].astype(bool)
        for j in range(n_gt):
            gj = gt_instances[j].astype(bool)
            inter = np.sum(pi & gj)
            union = np.sum(pi | gj)
            iou_mat[i, j] = inter / union if union > 0 else 0.0

    gt_matched = np.zeros(n_gt, dtype=bool)
    pred_matched = np.zeros(n_pred, dtype=bool)
    matches = []
    while True:
        if iou_mat.size == 0:
            break
        best = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
        if iou_mat[best] < iou_thresh:
            break
        pi, gj = best
        matches.append((int(pi), int(gj), float(iou_mat[best])))
        pred_matched[pi] = True
        gt_matched[gj] = True
        iou_mat[pi, :] = 0
        iou_mat[:, gj] = 0

    fp_indices = [i for i in range(n_pred) if not pred_matched[i]]
    fn_indices = [j for j in range(n_gt) if not gt_matched[j]]
    return matches, fp_indices, fn_indices


def run_per_instance_dsc_demo(pred_instances, gt_instances, iou_thresh=0.5):
    """Match instances by IoU, compute per-instance DSC, and plot the bar chart.

    Parameters
    ----------
    pred_instances : list of 2-D binary masks (predictions).
    gt_instances   : list of 2-D binary masks (ground truth).
    iou_thresh     : float, IoU threshold for matching.
    """
    matches, fp_indices, fn_indices = _match_instances_iou(
        pred_instances, gt_instances, iou_thresh
    )

    n_gt = len(gt_instances)
    gt_dscs = np.zeros(n_gt)
    for pi, gi, _ in matches:
        gt_dscs[gi] = _dice_score(pred_instances[pi], gt_instances[gi])

    mean_gt_dsc = float(np.mean(gt_dscs))

    gt_combined = np.clip(sum(gt_instances), 0, 1)
    pred_combined = np.clip(sum(pred_instances), 0, 1)
    overall_pixel_dsc = _dice_score(pred_combined, gt_combined)

    plot_per_instance_dsc(gt_instances, gt_dscs, mean_gt_dsc, overall_pixel_dsc)


# ------------------------------------------------------------------
# 10.  Mask IoU vs Box IoU comparison demo
# ------------------------------------------------------------------

def _bbox_from_mask(mask):
    """Return (r_min, c_min, r_max, c_max) bounding box of a binary mask."""
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return (0, 0, 0, 0)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _box_iou(box_a, box_b):
    """IoU between two (r_min, c_min, r_max, c_max) boxes."""
    r0 = max(box_a[0], box_b[0])
    c0 = max(box_a[1], box_b[1])
    r1 = min(box_a[2], box_b[2])
    c1 = min(box_a[3], box_b[3])
    inter = max(0, r1 - r0 + 1) * max(0, c1 - c0 + 1)
    area_a = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    area_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0


def _make_irregular_mask(H, W, cy, cx, r, rng, noise_level=0.25):
    """Circle mask with boundary noise to make it non-rectangular."""
    yy, xx = np.ogrid[:H, :W]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # Add per-angle noise to the radius
    angles = np.arctan2(yy - cy, xx - cx)
    r_noise = r + rng.uniform(-r * noise_level, r * noise_level, angles.shape)
    return (dist <= r_noise).astype(np.float64)


def run_mask_vs_box_iou_demo(seed=42):
    """Show mask IoU vs bounding box IoU on the same GT-prediction pair.

    Creates an irregular GT mask and a shifted/noisy prediction, then
    draws a 2-row figure: top row = mask IoU walkthrough, bottom row =
    bounding box IoU walkthrough.
    """
    rng = np.random.RandomState(seed)
    H, W = 120, 160

    gt_mask = _make_irregular_mask(H, W, 55, 80, 30, rng, noise_level=0.15)
    pred_mask = _make_irregular_mask(H, W, 50, 72, 28, rng, noise_level=0.35)

    gt_b = gt_mask.astype(bool)
    pred_b = pred_mask.astype(bool)

    # Mask IoU
    m_inter = gt_b & pred_b
    m_union = gt_b | pred_b
    mask_iou = np.sum(m_inter) / np.sum(m_union) if np.sum(m_union) > 0 else 0.0

    # Box IoU
    gt_box = _bbox_from_mask(gt_mask)
    pred_box = _bbox_from_mask(pred_mask)
    box_iou = _box_iou(gt_box, pred_box)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    # ---- Row 0: Mask IoU ----
    vis = np.zeros((H, W, 3))
    vis[gt_b] = [0.25, 0.55, 0.85]
    axes[0, 0].imshow(vis); axes[0, 0].set_title("GT mask", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    vis = np.zeros((H, W, 3))
    vis[pred_b] = [0.90, 0.50, 0.20]
    axes[0, 1].imshow(vis); axes[0, 1].set_title("Predicted mask", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    vis = np.zeros((H, W, 3))
    vis[gt_b] = [0.82, 0.90, 1.0]
    vis[pred_b] = [1.0, 0.90, 0.82]
    vis[m_inter] = [0.18, 0.75, 0.18]
    axes[0, 2].imshow(vis); axes[0, 2].set_title("Intersection", fontsize=11, fontweight="bold", color="#2d8a2d")
    axes[0, 2].axis("off")

    vis = np.zeros((H, W, 3))
    vis[m_union] = [0.55, 0.27, 0.62]
    axes[0, 3].imshow(vis); axes[0, 3].set_title("Union", fontsize=11, fontweight="bold", color="#8e44ad")
    axes[0, 3].axis("off")

    axes[0, 0].set_ylabel("Mask IoU", fontsize=13, fontweight="bold", rotation=0,
                           labelpad=70, va="center")

    # ---- Row 1: Box IoU ----
    def _draw_box_panel(ax, title, gt_box_, pred_box_, fill_gt=True, fill_pred=True,
                        fill_inter=False, fill_union=False):
        canvas = np.ones((H, W, 3)) * 0.95
        ax.imshow(canvas)
        if fill_union:
            # Fill union area
            r0 = min(gt_box_[0], pred_box_[0]); c0 = min(gt_box_[1], pred_box_[1])
            r1 = max(gt_box_[2], pred_box_[2]); c1 = max(gt_box_[3], pred_box_[3])
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((c0-0.5, r0-0.5), c1-c0+1, r1-r0+1,
                                   facecolor=(0.55, 0.27, 0.62, 0.4), edgecolor="none"))
        if fill_gt:
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle(
                (gt_box_[1]-0.5, gt_box_[0]-0.5),
                gt_box_[3]-gt_box_[1]+1, gt_box_[2]-gt_box_[0]+1,
                facecolor=(0.25, 0.55, 0.85, 0.3), edgecolor=(0.25, 0.55, 0.85),
                linewidth=2.5, linestyle="-",
            ))
        if fill_pred:
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle(
                (pred_box_[1]-0.5, pred_box_[0]-0.5),
                pred_box_[3]-pred_box_[1]+1, pred_box_[2]-pred_box_[0]+1,
                facecolor=(0.90, 0.50, 0.20, 0.3), edgecolor=(0.90, 0.50, 0.20),
                linewidth=2.5, linestyle="--",
            ))
        if fill_inter:
            ir0 = max(gt_box_[0], pred_box_[0]); ic0 = max(gt_box_[1], pred_box_[1])
            ir1 = min(gt_box_[2], pred_box_[2]); ic1 = min(gt_box_[3], pred_box_[3])
            if ir1 >= ir0 and ic1 >= ic0:
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((ic0-0.5, ir0-0.5), ic1-ic0+1, ir1-ir0+1,
                                       facecolor=(0.18, 0.75, 0.18, 0.5), edgecolor="none"))
        ax.set_xlim(-0.5, W-0.5); ax.set_ylim(H-0.5, -0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    _draw_box_panel(axes[1, 0], "GT box", gt_box, pred_box, fill_pred=False)
    _draw_box_panel(axes[1, 1], "Predicted box", gt_box, pred_box, fill_gt=False)
    _draw_box_panel(axes[1, 2], "Intersection", gt_box, pred_box, fill_inter=True)
    axes[1, 2].set_title("Intersection", fontsize=11, fontweight="bold", color="#2d8a2d")
    _draw_box_panel(axes[1, 3], "Union", gt_box, pred_box, fill_union=True)
    axes[1, 3].set_title("Union", fontsize=11, fontweight="bold", color="#8e44ad")

    axes[1, 0].set_ylabel("Box IoU", fontsize=13, fontweight="bold", rotation=0,
                           labelpad=70, va="center")

    fig.suptitle(
        f"Mask IoU = {mask_iou:.3f}   vs   Box IoU = {box_iou:.3f}   "
        f"(gap = {box_iou - mask_iou:+.3f})",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout(rect=[0.06, 0, 1, 0.95])
    plt.show()

    print(f"Mask IoU = {mask_iou:.4f}")
    print(f"Box  IoU = {box_iou:.4f}")
    print(f"Gap      = {box_iou - mask_iou:+.4f}")
    print()
    print("The bounding box smooths away all shape detail, so Box IoU is")
    print("almost always higher than Mask IoU for the same pair of objects.")
    print("Mask IoU penalizes any pixel-level disagreement in shape; Box IoU does not.")


# ------------------------------------------------------------------
# 11.  Instance matching walkthrough demo
# ------------------------------------------------------------------

def run_instance_matching_demo(seed=42):
    """Create synthetic instances and run through the full matching pipeline.

    Produces:
    1. Side-by-side instance label maps (GT vs predictions)
    2. IoU matrix heatmap with greedy matching overlay
    """
    rng = np.random.RandomState(seed)
    H, W = 180, 260

    # GT: 5 instances at known positions
    gt_centers = [(40, 50), (40, 130), (40, 210), (120, 90), (120, 180)]
    gt_radii = [22, 18, 25, 20, 22]
    gt_instances = [make_circle_mask(H, W, cy, cx, r)
                    for (cy, cx), r in zip(gt_centers, gt_radii)]

    # Predictions: 6 instances (5 detections + 1 spurious)
    # P0: good match for GT0 (shifted slightly)
    # P1: good match for GT1 (slightly smaller)
    # P2: overlaps GT2 but very noisy → might still match
    # P3: good match for GT3
    # P4: spurious detection (no GT overlap) → FP
    # (GT4 is unmatched → FN)
    pred_masks = []
    pred_scores = []

    # P0 → GT0
    pred_masks.append(_make_irregular_mask(H, W, 42, 52, 21, rng, noise_level=0.1))
    pred_scores.append(0.95)

    # P1 → GT1
    pred_masks.append(make_circle_mask(H, W, 41, 128, 16))
    pred_scores.append(0.88)

    # P2 → GT2 (noisy, borderline)
    pred_masks.append(_make_irregular_mask(H, W, 35, 205, 20, rng, noise_level=0.40))
    pred_scores.append(0.72)

    # P3 → GT3
    pred_masks.append(_make_irregular_mask(H, W, 122, 88, 19, rng, noise_level=0.15))
    pred_scores.append(0.65)

    # P4 → spurious (no GT)
    pred_masks.append(make_circle_mask(H, W, 155, 40, 14))
    pred_scores.append(0.40)

    # Show instance maps
    plot_instance_maps(gt_instances, pred_masks, H, W)

    # Show IoU matrix + matching
    matches, fp_list, fn_indices, iou_mat = plot_mask_iou_matrix(
        gt_instances, pred_masks, pred_scores, iou_thresh=0.5
    )

    return gt_instances, pred_masks, pred_scores


# ------------------------------------------------------------------
# 12.  PQ factorization 2x2 grid
# ------------------------------------------------------------------

def run_pq_factorization_grid(seed=42):
    """2x2 grid showing all four SQ/RQ combinations.

    | High SQ, High RQ | High SQ, Low RQ  |
    | Low SQ,  High RQ | Low SQ,  Low RQ  |
    """
    rng = np.random.RandomState(seed)
    H, W = 140, 140
    centers = [(35, 35), (35, 105), (105, 35), (105, 105)]
    gt = [make_circle_mask(H, W, cy, cx, 22) for cy, cx in centers]

    # ---- Scenario definitions ----

    # (a) High SQ, High RQ: all 4 detected, near-perfect masks
    pred_a = [make_circle_mask(H, W, cy + 1, cx - 1, 22)
              for cy, cx in centers]

    # (b) High SQ, Low RQ: only 2 detected, but masks are near-perfect
    pred_b = [make_circle_mask(H, W, cy + 1, cx, 22)
              for cy, cx in centers[:2]]

    # (c) Low SQ, High RQ: all 4 detected, but masks are noisy/shifted
    pred_c = []
    for cy, cx in centers:
        noisy = _make_irregular_mask(H, W, cy + 5, cx - 4, 18, rng, noise_level=0.45)
        pred_c.append(noisy)

    # (d) Low SQ, Low RQ: only 2 detected, and masks are noisy
    pred_d = []
    for cy, cx in centers[:2]:
        noisy = _make_irregular_mask(H, W, cy + 3, cx - 3, 19, rng, noise_level=0.35)
        pred_d.append(noisy)

    scenarios = [
        ("High SQ, High RQ\n(good model)", pred_a),
        ("High SQ, Low RQ\n(missed objects)", pred_b),
        ("Low SQ, High RQ\n(sloppy masks)", pred_c),
        ("Low SQ, Low RQ\n(both bad)", pred_d),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1]})

    # Consistent colormap
    n_max = len(centers)
    cmap_inst = plt.cm.get_cmap("tab10", n_max + 1)

    for idx, (label, preds) in enumerate(scenarios):
        row, col = divmod(idx, 2)
        # Left sub-column: GT
        ax_gt = axes[row, col * 2]
        gt_map = np.zeros((H, W))
        for i, m in enumerate(gt):
            gt_map[m > 0] = i + 1
        ax_gt.imshow(gt_map, cmap=cmap_inst, vmin=0, vmax=n_max)
        ax_gt.set_title("GT (4 inst.)", fontsize=10)
        ax_gt.axis("off")

        # Right sub-column: Prediction + scores
        ax_pred = axes[row, col * 2 + 1]
        pred_map = np.zeros((H, W))
        for i, m in enumerate(preds):
            pred_map[m > 0] = i + 1
        ax_pred.imshow(pred_map, cmap=cmap_inst, vmin=0, vmax=n_max)
        ax_pred.axis("off")

        pq, sq, rq, det = panoptic_quality(preds, gt)
        ax_pred.set_title(
            f"{label}\nSQ={sq:.2f}  RQ={rq:.2f}  PQ={pq:.2f}",
            fontsize=10, fontweight="bold",
        )

    fig.suptitle(
        "PQ Factorization: How SQ and RQ Diagnose Different Failure Modes",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

    # Print summary
    print("PQ Factorization Summary")
    print("=" * 60)
    print(f"  {'Scenario':<30s}  {'SQ':>6s}  {'RQ':>6s}  {'PQ':>6s}  {'TP':>3s}  {'FP':>3s}  {'FN':>3s}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*3}  {'-'*3}  {'-'*3}")
    for label, preds in scenarios:
        pq, sq, rq, det = panoptic_quality(preds, gt)
        short = label.split("\n")[0]
        print(f"  {short:<30s}  {sq:>6.3f}  {rq:>6.3f}  {pq:>6.3f}  "
              f"{det['tp']:>3d}  {det['fp']:>3d}  {det['fn']:>3d}")
