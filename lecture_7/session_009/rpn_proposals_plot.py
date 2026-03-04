"""Visualise RPN proposals on an image."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_rpn_proposals(image, proposals, top_k=20, top_highlight=5,
                       suptitle='Predicted RPN proposals'):
    """Show top-K RPN proposals (left) and top-N highlighted proposals (right).

    Parameters
    ----------
    image : np.ndarray
        HxWx3 RGB image (same coordinate space as *proposals*).
    proposals : np.ndarray
        (N, 4) array of boxes in (x1, y1, x2, y2) format, sorted by
        descending objectness score.
    top_k : int
        Number of proposals to draw in the overview panel (left).
    top_highlight : int
        Number of proposals to highlight individually (right).
    suptitle : str
        Figure super-title.
    """
    n = len(proposals)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: top-K proposals
    axes[0].imshow(image)
    for i in range(min(top_k, n)):
        b = proposals[i]
        c = plt.cm.plasma(i / max(top_k, 1))
        axes[0].add_patch(patches.Rectangle(
            (b[0], b[1]), b[2] - b[0], b[3] - b[1],
            lw=1.5, edgecolor=c, facecolor='none'))
    axes[0].set_title(f'Top {min(top_k, n)} RPN proposals', fontsize=12)
    axes[0].axis('off')

    # Right: top-N highlighted proposals
    axes[1].imshow(image)
    for i in range(min(top_highlight, n)):
        b = proposals[i]
        c = plt.cm.tab10.colors[i]
        axes[1].add_patch(patches.Rectangle(
            (b[0], b[1]), b[2] - b[0], b[3] - b[1],
            lw=3, edgecolor=c, facecolor=c, alpha=0.15))
        axes[1].text(b[0], b[1] - 4, f'#{i+1}', fontsize=11, color=c,
                     fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2',
                               facecolor='white', alpha=0.8))
    axes[1].set_title(f'Top {min(top_highlight, n)} proposals (highest objectness)',
                      fontsize=12)
    axes[1].axis('off')

    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
