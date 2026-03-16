"""Precision-Recall curve demo for detection evaluation.

Provides:
- ``bbox_iou``               IoU between two (x1,y1,x2,y2) boxes.
- ``match_predictions_to_gt`` Greedy VOC-style matching.
- ``compute_pr_curve``        Cumulative precision/recall from TP/FP labels.
- ``pr_curve_demo``           End-to-end: build demo data, match, print table, plot.
"""

import numpy as np
from session_010.detection_plots import plot_pr_curve as _plot_pr_curve


# ------------------------------------------------------------------
# IoU
# ------------------------------------------------------------------

def bbox_iou(box_a, box_b):
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
# Greedy matching (VOC protocol)
# ------------------------------------------------------------------

def match_predictions_to_gt(pred_boxes, pred_scores, gt_boxes,
                            iou_threshold=0.5):
    """Greedy matching of predictions to ground-truth boxes.

    Parameters
    ----------
    pred_boxes   : (N, 4) array  [x1, y1, x2, y2]
    pred_scores  : (N,) confidence scores
    gt_boxes     : (M, 4) array
    iou_threshold: float

    Returns
    -------
    tp_fp         : (N,) int array, 1=TP / 0=FP  (confidence-sorted order)
    sorted_scores : (N,) float array (descending)
    n_gt          : int
    """
    sorted_idx = np.argsort(-pred_scores)
    sorted_scores = pred_scores[sorted_idx]
    sorted_boxes = pred_boxes[sorted_idx]

    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    gt_matched = np.zeros(n_gt, dtype=bool)
    tp_fp = np.zeros(n_pred, dtype=int)

    for i in range(n_pred):
        best_iou = 0.0
        best_gt = -1
        for j in range(n_gt):
            iou = bbox_iou(sorted_boxes[i], gt_boxes[j])
            if iou > best_iou:
                best_iou = iou
                best_gt = j
        if best_iou >= iou_threshold and best_gt >= 0 and not gt_matched[best_gt]:
            tp_fp[i] = 1
            gt_matched[best_gt] = True

    return tp_fp, sorted_scores, n_gt


# ------------------------------------------------------------------
# PR curve computation
# ------------------------------------------------------------------

def compute_pr_curve(tp_fp, n_gt):
    """Cumulative precision and recall from TP/FP labels.

    Parameters
    ----------
    tp_fp : (N,) int array, 1=TP / 0=FP  (sorted by descending confidence)
    n_gt  : int

    Returns
    -------
    precisions : (N,) array
    recalls    : (N,) array
    """
    cum_tp = np.cumsum(tp_fp)
    cum_fp = np.cumsum(1 - tp_fp)
    precisions = cum_tp / (cum_tp + cum_fp)
    recalls = cum_tp / n_gt
    return precisions, recalls


# ------------------------------------------------------------------
# Demo data
# ------------------------------------------------------------------

def _make_demo_data():
    """Return (gt_boxes, pred_boxes, pred_scores) for a 10-GT / 15-pred scene."""
    gt_boxes = np.array([
        [10,  10,  40,  40],
        [60,  10,  100, 50],
        [120, 10,  160, 45],
        [10,  70,  50,  110],
        [70,  70,  110, 100],
        [130, 70,  170, 110],
        [10,  130, 45,  170],
        [60,  130, 105, 170],
        [120, 130, 155, 160],
        [170, 130, 210, 170],
    ])

    pred_boxes = np.array([
        [12,  12,  38,  38],    # Good match for GT 0
        [62,  8,   98,  48],    # Good match for GT 1
        [118, 12,  158, 46],    # Good match for GT 2
        [8,   72,  48,  108],   # Good match for GT 3
        [72,  72,  108, 98],    # Good match for GT 4
        [132, 72,  168, 108],   # Good match for GT 5
        [11,  131, 44,  168],   # Good match for GT 6
        [58,  128, 103, 168],   # Good match for GT 7
        [15,  15,  35,  35],    # Duplicate of GT 0  (FP)
        [200, 200, 230, 230],   # No match            (FP)
        [250, 10,  280, 40],    # No match            (FP)
        [125, 132, 153, 158],   # Decent match for GT 8
        [168, 128, 208, 168],   # Good match for GT 9
        [65,  12,  95,  45],    # Duplicate of GT 1  (FP)
        [180, 50,  220, 80],    # No match            (FP)
    ])

    pred_scores = np.array([
        0.98, 0.95, 0.93, 0.90, 0.88, 0.85, 0.80, 0.75,
        0.72, 0.65, 0.60, 0.55, 0.50, 0.40, 0.30,
    ])

    return gt_boxes, pred_boxes, pred_scores


# ------------------------------------------------------------------
# Public demo function
# ------------------------------------------------------------------

def pr_curve_demo(iou_threshold=0.5):
    """Run the full PR curve demo: match, print table, plot.

    Uses a fixed scene with 10 GT boxes and 15 predictions so results
    are reproducible.
    """
    gt_boxes, pred_boxes, pred_scores = _make_demo_data()

    tp_fp, sorted_scores, n_gt = match_predictions_to_gt(
        pred_boxes, pred_scores, gt_boxes, iou_threshold=iou_threshold,
    )
    precisions, recalls = compute_pr_curve(tp_fp, n_gt)

    # --- Print step-by-step table ---
    print("Step-by-step Precision-Recall computation")
    print("=" * 65)
    print(f"{'Step':>4s}  {'Conf':>5s}  {'TP/FP':>5s}  "
          f"{'Cum TP':>6s}  {'Cum FP':>6s}  {'Prec':>6s}  {'Rec':>6s}")
    print("-" * 65)
    for i in range(len(tp_fp)):
        cum_tp = int(np.sum(tp_fp[:i + 1]))
        cum_fp = (i + 1) - cum_tp
        label = "TP" if tp_fp[i] == 1 else "FP"
        print(f"{i + 1:4d}  {sorted_scores[i]:5.2f}  {label:>5s}  "
              f"{cum_tp:6d}  {cum_fp:6d}  "
              f"{precisions[i]:6.4f}  {recalls[i]:6.4f}")
    print("-" * 65)
    final_tp = int(np.sum(tp_fp))
    print(f"\nTotal GT boxes: {n_gt},  Final TP: {final_tp},  "
          f"Final FP: {len(tp_fp) - final_tp},  FN: {n_gt - final_tp}")

    # --- Plot ---
    _plot_pr_curve(recalls, precisions, sorted_scores)

    print("\nLeft:  The PR curve as a step function. As we lower the confidence")
    print("       threshold, recall increases but precision generally decreases.")
    print("\nRight: Each point colored by its confidence score.")
    print("       Green (high confidence) = upper-left.")
    print("       Red (low confidence) = lower-right.")
