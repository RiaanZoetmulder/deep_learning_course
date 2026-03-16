"""Segmentation metric visualizations for the CV section.

Functions
---------
plot_segmentation_grids     Binary vs multi-class segmentation pixel grids.
plot_per_class_iou          Per-class IoU bar chart with mIoU reference line.
plot_fwiou_comparison       mIoU vs FWIoU: weights and contributions side-by-side.
plot_dsc_iou_relationship   DSC-IoU monotonic curve + per-class grouped bars.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_segmentation_grids(binary_mask, multiclass_mask, class_labels=None):
    """Side-by-side pixel grids for binary and multi-class segmentation.

    Mimics the grid style used for connected-component visualizations:
    coloured cells with the class label printed inside each pixel.

    Parameters
    ----------
    binary_mask     : 2-D int array with values in {0, 1}.
    multiclass_mask : 2-D int array with values in {0, 1, 2, ...}.
    class_labels    : list of class-name strings (one per unique value in
                      *multiclass_mask*).  If *None* a default is used.
    """
    n_classes = int(multiclass_mask.max()) + 1
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(n_classes)]

    # --- shared palette (colours 0 and 1 identical in both panels) ---
    palette = ["#2b2b2b", "#4a90d9", "#e07b3a", "#5bb55b", "#c44e52"]
    text_colors = {0: "#cccccc", 1: "white", 2: "white", 3: "white", 4: "white"}
    cmap_bin = ListedColormap(palette[:2])
    cmap_mc = ListedColormap(palette[:n_classes])

    rows, cols = binary_mask.shape

    fig, axes = plt.subplots(1, 2, figsize=(2.8 * cols + 2, 1.5 * rows))

    # --- helper to draw one grid ---
    def _draw(ax, mask, cmap, vmax, title):
        ax.imshow(mask, cmap=cmap, vmin=0, vmax=vmax, origin="upper")
        for r in range(rows):
            for c in range(cols):
                v = int(mask[r, c])
                ax.text(
                    c, r, str(v),
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color=text_colors.get(v, "white"),
                )
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.tick_params(length=0, labelsize=0)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    _draw(axes[0], binary_mask, cmap_bin, 1,
          "Binary Segmentation\n(background / foreground)")

    _draw(axes[1], multiclass_mask, cmap_mc, n_classes - 1,
          f"{n_classes}-Class Segmentation")

    # --- legends ---
    from matplotlib.patches import Patch

    bin_handles = [
        Patch(facecolor=palette[0], edgecolor="white", label="0  background"),
        Patch(facecolor=palette[1], edgecolor="white", label="1  foreground"),
    ]
    axes[0].legend(handles=bin_handles, loc="upper center",
                   bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=9, frameon=False)

    mc_handles = [
        Patch(facecolor=palette[i], edgecolor="white",
              label=f"{i}  {class_labels[i]}")
        for i in range(n_classes)
    ]
    axes[1].legend(handles=mc_handles, loc="upper center",
                   bbox_to_anchor=(0.5, -0.04), ncol=n_classes, fontsize=9,
                   frameon=False)

    plt.tight_layout()
    plt.show()


def plot_per_class_iou(class_names, ious, miou, cmap_classes, C):
    """Bar chart of per-class IoU with mIoU reference line.

    Parameters
    ----------
    class_names : list[str]
    ious : dict[int, float]
    miou : float
    cmap_classes : matplotlib colormap
    C : int, number of classes
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [cmap_classes(c) for c in range(C)]
    bars = ax.bar(
        class_names,
        [ious[c] for c in range(C)],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.axhline(
        y=miou, color="red", linestyle="--", linewidth=2, label=f"mIoU = {miou:.4f}"
    )
    for bar, c in zip(bars, range(C)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{ious[c]:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_ylabel("IoU", fontsize=13)
    ax.set_title("Per-Class IoU with mIoU Reference Line", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_fwiou_comparison(class_names, ious, freqs, miou, fwiou, C):
    """Side-by-side bar chart comparing mIoU and FWIoU weighting.

    Parameters
    ----------
    class_names : list[str]
    ious : dict[int, float]
    freqs : dict[int, float]
    miou : float
    fwiou : float
    C : int
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(C)
    width = 0.35

    # Left: per-class weights
    axes[0].bar(
        x - width / 2,
        [1 / C] * C,
        width,
        label="mIoU weight (uniform)",
        color="steelblue",
        edgecolor="black",
    )
    axes[0].bar(
        x + width / 2,
        [freqs[c] for c in range(C)],
        width,
        label="FWIoU weight (frequency)",
        color="coral",
        edgecolor="black",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, fontsize=11)
    axes[0].set_ylabel("Weight", fontsize=12)
    axes[0].set_title("Class Weights: mIoU vs FWIoU", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)

    # Right: weighted contributions
    miou_contribs = [ious[c] / C for c in range(C)]
    fwiou_contribs = [freqs[c] * ious[c] for c in range(C)]
    axes[1].bar(
        x - width / 2,
        miou_contribs,
        width,
        label=f"mIoU contributions (sum={miou:.3f})",
        color="steelblue",
        edgecolor="black",
    )
    axes[1].bar(
        x + width / 2,
        fwiou_contribs,
        width,
        label=f"FWIoU contributions (sum={fwiou:.3f})",
        color="coral",
        edgecolor="black",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, fontsize=11)
    axes[1].set_ylabel("Weighted IoU contribution", fontsize=12)
    axes[1].set_title(
        "Per-Class Contributions to Final Metric", fontsize=13, fontweight="bold"
    )
    axes[1].legend(fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(
        "mIoU vs FWIoU: How Class Frequency Affects the Metric",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_dsc_iou_relationship(class_names, ious, dscs, miou, cmap_classes, C):
    """DSC-IoU monotonic curve + per-class grouped bar chart.

    Parameters
    ----------
    class_names : list[str]
    ious : dict[int, float]
    dscs : dict[int, float]
    miou : float
    cmap_classes : matplotlib colormap
    C : int
    """
    iou_range = np.linspace(0, 1, 500)
    dsc_range = 2 * iou_range / (1 + iou_range)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: DSC vs IoU curve
    ax = axes[0]
    ax.plot(
        iou_range,
        dsc_range,
        "b-",
        linewidth=2.5,
        label=r"$\mathrm{DSC} = \frac{2 \cdot \mathrm{IoU}}{1 + \mathrm{IoU}}$",
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="y = x (reference)")
    ax.fill_between(
        iou_range, iou_range, dsc_range, alpha=0.15, color="blue", label="DSC > IoU region"
    )
    for c in range(C):
        ax.plot(
            ious[c],
            dscs[c],
            "o",
            color=cmap_classes(c),
            markersize=10,
            markeredgecolor="black",
            label=f"{class_names[c]}",
            zorder=5,
        )
    ax.set_xlabel("IoU", fontsize=13)
    ax.set_ylabel("DSC", fontsize=13)
    ax.set_title(
        "DSC is Always >= IoU (Monotonic Relationship)", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # Right: IoU vs DSC bar chart per class
    ax = axes[1]
    x = np.arange(C)
    width = 0.3
    ax.bar(
        x - width / 2,
        [ious[c] for c in range(C)],
        width,
        label="IoU",
        color="steelblue",
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        [dscs[c] for c in range(C)],
        width,
        label="DSC (F1)",
        color="darkorange",
        edgecolor="black",
    )
    mean_dsc = np.mean([dscs[c] for c in range(C)])
    ax.axhline(
        y=miou,
        color="steelblue",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"mIoU = {miou:.3f}",
    )
    ax.axhline(
        y=mean_dsc,
        color="darkorange",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"mean DSC = {mean_dsc:.3f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("IoU vs DSC Per Class", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    return mean_dsc
