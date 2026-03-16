"""Object detection evaluation visualizations.

Functions
---------
plot_greedy_matching   Side-by-side: all boxes vs. matching result with TP/FP/FN.
plot_pr_curve          Raw PR curve + confidence-colored PR curve.
plot_ap_methods        3-panel: raw PR, VOC 11-point, COCO all-point.
plot_map_vs_iou        Per-class AP bars + mAP vs IoU-threshold curve.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ------------------------------------------------------------------
# 1. Greedy matching visualisation (1x2)
# ------------------------------------------------------------------

def plot_greedy_matching(
    match_gt_boxes,
    match_pred_boxes,
    match_pred_scores,
    match_gt_matched,
    match_sorted_indices,
    match_results,
    match_pairs,
):
    """Side-by-side: all boxes (left) vs. greedy-matching result (right).

    Parameters
    ----------
    match_gt_boxes      : (N_gt, 4) array of [x1, y1, x2, y2].
    match_pred_boxes    : (N_pred, 4) array.
    match_pred_scores   : (N_pred,) confidence scores.
    match_gt_matched    : (N_gt,) bool, whether each GT was matched.
    match_sorted_indices: sorted prediction indices (descending score).
    match_results       : list of 'TP'/'FP' strings for each sorted prediction.
    match_pairs         : list of (pred_idx, gt_idx) for TP matches.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left panel: All boxes ---
    ax = axes[0]
    ax.set_xlim(-10, 270)
    ax.set_ylim(-10, 270)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("All Boxes (before matching)", fontsize=13, fontweight="bold")

    for i, box in enumerate(match_gt_boxes):
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2.5,
            edgecolor="green",
            facecolor="green",
            alpha=0.15,
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(
            box[0] + 2, box[1] - 3, f"GT {i}",
            color="green", fontsize=10, fontweight="bold",
        )

    for i, (box, score) in enumerate(zip(match_pred_boxes, match_pred_scores)):
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor="red",
            facecolor="red",
            alpha=0.1,
        )
        ax.add_patch(rect)
        ax.text(
            box[2] - 2, box[3] + 12, f"P{i} ({score:.2f})",
            color="red", fontsize=9, ha="right", fontweight="bold",
        )

    ax.legend(
        [
            mpatches.Patch(
                edgecolor="green", facecolor="green", alpha=0.15, linestyle="--"
            ),
            mpatches.Patch(edgecolor="red", facecolor="red", alpha=0.1),
        ],
        ["Ground Truth", "Prediction"],
        loc="lower right",
        fontsize=10,
    )

    # --- Right panel: Matching result ---
    ax = axes[1]
    ax.set_xlim(-10, 270)
    ax.set_ylim(-10, 270)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("After Greedy Matching (IoU >= 0.5)", fontsize=13, fontweight="bold")

    tp_fp_colors = {"TP": "blue", "FP": "red"}

    for i, box in enumerate(match_gt_boxes):
        color = "green" if match_gt_matched[i] else "orange"
        label = "Matched" if match_gt_matched[i] else "FN"
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.15,
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(
            box[0] + 2, box[1] - 3, f"GT {i} ({label})",
            color=color, fontsize=9, fontweight="bold",
        )

    for step, pred_idx in enumerate(match_sorted_indices):
        box = match_pred_boxes[pred_idx]
        result = match_results[step]
        color = tp_fp_colors[result]
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=0.15,
        )
        ax.add_patch(rect)
        ax.text(
            box[2] - 2,
            box[3] + 12,
            f"P{pred_idx} ({match_pred_scores[pred_idx]:.2f}) | {result}",
            color=color,
            fontsize=9,
            ha="right",
            fontweight="bold",
        )

    # Draw arrows for TP matches
    for pred_idx, gt_idx in match_pairs:
        p = match_pred_boxes[pred_idx]
        g = match_gt_boxes[gt_idx]
        p_center = ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2)
        g_center = ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)
        ax.annotate(
            "",
            xy=g_center,
            xytext=p_center,
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5, linestyle="--"),
        )

    ax.legend(
        [
            mpatches.Patch(edgecolor="blue", facecolor="blue", alpha=0.15),
            mpatches.Patch(edgecolor="red", facecolor="red", alpha=0.15),
            mpatches.Patch(edgecolor="orange", facecolor="orange", alpha=0.15),
        ],
        ["TP", "FP", "FN (unmatched GT)"],
        loc="lower right",
        fontsize=10,
    )

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 2. PR curve (1x2)
# ------------------------------------------------------------------

def plot_pr_curve(recalls, precisions, sorted_scores):
    """Left: raw PR step curve.  Right: PR curve colored by confidence.

    Parameters
    ----------
    recalls       : 1D array of recall values.
    precisions    : 1D array of precision values.
    sorted_scores : 1D array of confidence scores (sorted descending).
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: raw PR curve
    ax = axes[0]
    ax.step(recalls, precisions, where="post", color="steelblue", linewidth=2, label="PR curve")
    ax.scatter(recalls, precisions, color="steelblue", s=40, zorder=5)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Raw Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    for i in range(min(5, len(recalls))):
        ax.annotate(
            f"({recalls[i]:.2f}, {precisions[i]:.2f})",
            (recalls[i], precisions[i]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            color="gray",
        )

    # Right: colored by confidence
    ax = axes[1]
    scatter = ax.scatter(
        recalls, precisions, c=sorted_scores, cmap="RdYlGn",
        s=80, edgecolors="black", linewidth=0.5, zorder=5,
    )
    ax.step(recalls, precisions, where="post", color="gray", linewidth=1, alpha=0.5)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("PR Curve: Colored by Confidence", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.colorbar(scatter, ax=ax, label="Confidence Score")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 3. AP methods comparison (1x3)
# ------------------------------------------------------------------

def plot_ap_methods(
    recalls,
    precisions,
    recall_levels_11,
    interp_prec_11,
    ap_voc11,
    mrec_coco,
    mpre_coco,
    ap_coco,
):
    """3-panel: raw PR, VOC 11-point interpolation, COCO all-point interpolation.

    Parameters
    ----------
    recalls, precisions : 1D arrays for the raw PR curve.
    recall_levels_11    : 11 recall sample points (0, 0.1, ..., 1.0).
    interp_prec_11      : interpolated precision at those 11 points.
    ap_voc11            : float, VOC 11-point AP.
    mrec_coco, mpre_coco: COCO-style interpolated recall/precision arrays.
    ap_coco             : float, COCO all-point AP.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Raw PR
    ax = axes[0]
    ax.step(recalls, precisions, where="post", color="steelblue", linewidth=2, label="Raw PR curve")
    ax.scatter(recalls, precisions, color="steelblue", s=30, zorder=5)
    ax.fill_between(recalls, precisions, step="post", alpha=0.15, color="steelblue")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Raw PR Curve", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)

    # Panel 2: VOC 11-point
    ax = axes[1]
    ax.step(recalls, precisions, where="post", color="lightgray", linewidth=1, label="Raw PR")
    ax.bar(
        recall_levels_11, interp_prec_11, width=0.08, color="orange", alpha=0.6,
        edgecolor="darkorange", label=f"11-point AP = {ap_voc11:.4f}",
    )
    ax.scatter(recall_levels_11, interp_prec_11, color="darkorange", s=50, zorder=5)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"VOC 11-Point Interpolation\nAP = {ap_voc11:.4f}", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)

    # Panel 3: COCO all-point
    ax = axes[2]
    ax.step(recalls, precisions, where="post", color="lightgray", linewidth=1, label="Raw PR")
    ax.step(mrec_coco, mpre_coco, where="post", color="green", linewidth=2, label="Interpolated PR")
    ax.fill_between(
        mrec_coco, mpre_coco, step="post", alpha=0.2, color="green",
        label=f"All-point AP = {ap_coco:.4f}",
    )
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"COCO All-Point Interpolation\nAP = {ap_coco:.4f}", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 4. mAP vs IoU threshold (1x2)
# ------------------------------------------------------------------

def plot_map_vs_iou(
    classes,
    class_names,
    per_class_05,
    mAP_05,
    per_thr,
    mAP_75,
    coco_ap,
):
    """Left: per-class AP bars at IoU=0.5.  Right: mAP vs IoU threshold curve.

    Parameters
    ----------
    classes       : list of class indices.
    class_names   : dict mapping class index to name.
    per_class_05  : dict mapping class index to AP@0.5.
    mAP_05        : float, mAP at IoU=0.5.
    per_thr       : dict mapping IoU threshold to mAP value.
    mAP_75        : float, mAP at IoU=0.75.
    coco_ap       : float, COCO AP@[.5:.95].
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: AP per class at IoU=0.5
    ax = axes[0]
    class_labels = [class_names[c] for c in classes]
    ap_values = [per_class_05[c] for c in classes]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    bars = ax.bar(class_labels, ap_values, color=colors, edgecolor="black", alpha=0.8)
    ax.axhline(
        y=mAP_05, color="black", linestyle="--", linewidth=2,
        label=f"mAP@0.5 = {mAP_05:.4f}",
    )
    ax.set_ylabel("AP@0.5", fontsize=12)
    ax.set_title("Per-Class AP at IoU=0.5", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    for bar, val in zip(bars, ap_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    # Panel 2: mAP vs IoU threshold
    ax = axes[1]
    thresholds = sorted(per_thr.keys())
    mAP_values = [per_thr[t] for t in thresholds]
    ax.plot(thresholds, mAP_values, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.fill_between(thresholds, mAP_values, alpha=0.15, color="steelblue")
    ax.axhline(
        y=coco_ap, color="red", linestyle="--", linewidth=2,
        label=f"AP@[.5:.95] = {coco_ap:.4f}",
    )
    ax.axvline(x=0.5, color="green", linestyle=":", alpha=0.5, label=f"AP@.5 = {per_thr[0.5]:.4f}")
    ax.axvline(x=0.75, color="orange", linestyle=":", alpha=0.5, label=f"AP@.75 = {mAP_75:.4f}")
    ax.set_xlabel("IoU Threshold", fontsize=12)
    ax.set_ylabel("mAP", fontsize=12)
    ax.set_title("mAP vs. IoU Threshold", fontsize=13, fontweight="bold")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.show()
