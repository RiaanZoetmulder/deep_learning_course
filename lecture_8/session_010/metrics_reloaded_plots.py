"""Metrics Reloaded summary heatmap.

Functions
---------
plot_metric_category_heatmap  Metric-vs-category heatmap (primary / secondary / N/A).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def plot_metric_category_heatmap(data, metrics, categories, labels_map=None, colors_map=None):
    """Heatmap of metric applicability across problem categories.

    Parameters
    ----------
    data       : 2D int array, shape (n_metrics, n_categories). 0=N/A, 1=secondary, 2=primary.
    metrics    : list of metric name strings.
    categories : list of category name strings.
    labels_map : dict mapping int to label text (default provided).
    colors_map : dict mapping int to text colour (default provided).
    """
    if labels_map is None:
        labels_map = {0: " - ", 1: "secondary", 2: "primary"}
    if colors_map is None:
        colors_map = {0: "#999999", 1: "#1f4e79", 2: "white"}

    cmap = ListedColormap(["#f0f0f0", "#a1c9f4", "#1f4e79"])

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=2)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Grid lines
    for i in range(len(metrics) + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.5)
    for j in range(len(categories) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.5)

    # Cell annotations
    for i in range(len(metrics)):
        for j in range(len(categories)):
            val = data[i, j]
            ax.text(
                j, i, labels_map[val], ha="center", va="center",
                fontsize=9, color=colors_map[val],
                fontweight="bold" if val == 2 else "normal",
            )

    # Group dividers
    ax.axhline(11.5, color="#333333", linewidth=2)
    ax.axhline(15.5, color="#333333", linewidth=2)

    # Group labels
    ax.text(
        len(categories) - 0.2, 5.5,
        "Semantic\nSegmentation\nMetrics",
        ha="left", va="center", fontsize=9, fontstyle="italic", color="#555555",
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        len(categories) - 0.2, 13.5,
        "Object\nDetection\nMetrics",
        ha="left", va="center", fontsize=9, fontstyle="italic", color="#555555",
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        len(categories) - 0.2, 17.5,
        "Instance\nSeg. Metrics",
        ha="left", va="center", fontsize=9, fontstyle="italic", color="#555555",
        transform=ax.get_xaxis_transform(),
    )

    ax.set_title(
        "Metrics Reloaded: Metric / Problem Category Mapping\n",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    legend_elements = [
        Patch(facecolor="#1f4e79", label="Primary metric for this category"),
        Patch(facecolor="#a1c9f4", label="Secondary / complementary"),
        Patch(facecolor="#f0f0f0", edgecolor="#cccccc", label="Not applicable"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower center",
        bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=9, frameon=False,
    )

    plt.tight_layout()
    plt.show()
