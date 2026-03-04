"""Visualise Faster R-CNN detection results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_proposal_grid(image, proposals, class_logits, class_names,
                       n_show=8, ncols=4,
                       suptitle='RoI Align — top proposals with predicted class'):
    """Show top-N proposals in a grid, each with its predicted class label.

    Parameters
    ----------
    image : np.ndarray
        HxWx3 RGB image (same coordinate space as *proposals*).
    proposals : np.ndarray
        (N, 4) boxes in (x1, y1, x2, y2) format.
    class_logits : array-like
        (N, C) raw class logits (torch tensor or ndarray).
        Softmax is applied internally.
    class_names : list[str]
        Class name for each logit index.
    n_show : int
        Number of proposals to display.
    ncols : int
        Number of columns in the grid.
    suptitle : str
        Figure super-title.
    """
    import torch

    n = min(n_show, len(proposals))
    if n == 0:
        print("No proposals to show.")
        return

    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes)

    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        if idx < n:
            ax.imshow(image)
            b = proposals[idx]
            c = plt.cm.tab10.colors[idx % 10]
            ax.add_patch(patches.Rectangle(
                (b[0], b[1]), b[2] - b[0], b[3] - b[1],
                lw=3, edgecolor='white', facecolor='none'))

            if isinstance(class_logits, torch.Tensor):
                probs = torch.softmax(class_logits[idx], dim=0)
                top_cls = probs.argmax().item()
                conf = probs[top_cls].item()
            else:
                from scipy.special import softmax as sp_softmax
                probs = sp_softmax(class_logits[idx])
                top_cls = int(np.argmax(probs))
                conf = probs[top_cls]

            ax.set_title(f'Proposal {idx} → \u201c{class_names[top_cls]}\u201d'
                         f' ({conf:.2f})', fontsize=11)
        else:
            ax.set_visible(False)

        ax.axis('off')

    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_detections_vs_gt(image, det_boxes, det_scores, det_labels,
                          class_names, gt_box, iou_fn,
                          suptitle='Inference result vs Ground Truth'):
    """Plot final detections overlaid on the image with GT comparison.

    Parameters
    ----------
    image : np.ndarray
        HxWx3 RGB image (original coordinates).
    det_boxes : np.ndarray
        (K, 4) detected boxes after NMS + score filtering.
    det_scores : np.ndarray
        (K,) confidence scores.
    det_labels : np.ndarray
        (K,) integer class indices.
    class_names : list[str]
        Class name for each label index.
    gt_box : tuple
        (x1, y1, x2, y2) ground-truth box.
    iou_fn : callable
        ``iou_fn(box_a, box_b) -> float``.
    suptitle : str
        Figure super-title.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)

    for i in range(len(det_boxes)):
        b = det_boxes[i]
        c = plt.cm.tab10.colors[i % 10]
        ax.add_patch(patches.Rectangle(
            (b[0], b[1]), b[2] - b[0], b[3] - b[1],
            lw=3, edgecolor=c, facecolor=c, alpha=0.15))
        label = f'\u201c{class_names[det_labels[i]]}\u201d {det_scores[i]:.2f}'
        ax.text(b[0], b[1] - 5, label, fontsize=12, color=c, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Ground truth (dashed cyan)
    ax.add_patch(patches.Rectangle(
        (gt_box[0], gt_box[1]),
        gt_box[2] - gt_box[0], gt_box[3] - gt_box[1],
        lw=3, edgecolor='cyan', facecolor='none', ls='--'))

    if len(det_boxes) > 0:
        det_ious = [iou_fn(tuple(b), gt_box) for b in det_boxes]
        best_i = int(np.argmax(det_ious))
        title = (f'{suptitle}\n'
                 f'Best match: \u201c{class_names[det_labels[best_i]]}\u201d '
                 f'(IoU = {det_ious[best_i]:.3f}, score = {det_scores[best_i]:.2f})')
    else:
        title = 'No detections above threshold'

    ax.set_title(title, fontsize=13)
    ax.legend(handles=[
        patches.Patch(edgecolor='cyan', facecolor='none', ls='--',
                      label='Ground Truth'),
    ], loc='upper left', fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
