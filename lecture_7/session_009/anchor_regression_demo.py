"""
Interactive demo: Anchor Box Regression Parametrization.

Shows how the four regression deltas (tx, ty, tw, th) transform an anchor
box into a predicted box, using the standard parametrization from
R-CNN / Fast R-CNN / Faster R-CNN.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display, clear_output
import ipywidgets as widgets


def apply_anchor_regression(
    anchor: tuple,
    tx: float,
    ty: float,
    tw: float,
    th: float,
) -> tuple:
    """
    Apply the standard anchor box regression parametrization.

    Given an anchor (x_min, y_min, x_max, y_max) and four offsets,
    decode a predicted box using:

        x_pred = tx * w_a + x_a
        y_pred = ty * h_a + y_a
        w_pred = w_a * exp(tw)
        h_pred = h_a * exp(th)

    Returns:
        (x_min, y_min, x_max, y_max) of the predicted box.
    """
    x1_a, y1_a, x2_a, y2_a = anchor
    w_a = x2_a - x1_a
    h_a = y2_a - y1_a
    cx_a = x1_a + w_a / 2
    cy_a = y1_a + h_a / 2

    # Decode
    cx_pred = tx * w_a + cx_a
    cy_pred = ty * h_a + cy_a
    w_pred = w_a * np.exp(tw)
    h_pred = h_a * np.exp(th)

    x1_pred = cx_pred - w_pred / 2
    y1_pred = cy_pred - h_pred / 2
    x2_pred = cx_pred + w_pred / 2
    y2_pred = cy_pred + h_pred / 2

    return (x1_pred, y1_pred, x2_pred, y2_pred)


def _draw_box(ax, box, color, label, linewidth=2.5, linestyle='-'):
    """Draw a single bounding box on an axis."""
    x1, y1, x2, y2 = box
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=linewidth, linestyle=linestyle,
        edgecolor=color, facecolor='none', label=label,
    )
    ax.add_patch(rect)
    # Mark center
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    ax.plot(cx, cy, marker='+', markersize=10, color=color, markeredgewidth=2)


def _plot_regression(anchor, tx, ty, tw, th):
    """Render anchor + predicted box side-by-side with offset annotations."""
    pred = apply_anchor_regression(anchor, tx, ty, tw, th)

    x1_a, y1_a, x2_a, y2_a = anchor
    w_a = x2_a - x1_a
    h_a = y2_a - y1_a
    cx_a = x1_a + w_a / 2
    cy_a = y1_a + h_a / 2

    x1_p, y1_p, x2_p, y2_p = pred
    w_p = x2_p - x1_p
    h_p = y2_p - y1_p
    cx_p = (x1_p + x2_p) / 2
    cy_p = (y1_p + y2_p) / 2

    # Determine view bounds with padding
    all_x = [x1_a, x2_a, x1_p, x2_p]
    all_y = [y1_a, y2_a, y1_p, y2_p]
    pad = max(w_a, h_a) * 0.4
    view_x0 = min(all_x) - pad
    view_x1 = max(all_x) + pad
    view_y0 = min(all_y) - pad
    view_y1 = max(all_y) + pad

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_xlim(view_x0, view_x1)
    ax.set_ylim(view_y1, view_y0)  # invert y to match image convention
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Anchor Box Regression', fontsize=14, fontweight='bold')

    # Draw anchor (blue dashed)
    _draw_box(ax, anchor, color='#1f77b4', label='Anchor', linewidth=2.5, linestyle='--')

    # Draw predicted box (orange solid)
    _draw_box(ax, pred, color='#ff7f0e', label='Predicted', linewidth=2.5, linestyle='-')

    # Draw arrow from anchor center to predicted center
    if abs(cx_p - cx_a) > 0.5 or abs(cy_p - cy_a) > 0.5:
        ax.annotate(
            '', xy=(cx_p, cy_p), xytext=(cx_a, cy_a),
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2),
        )

    # Annotations showing the formulas with actual values
    info_lines = [
        f"Anchor:  center=({cx_a:.0f}, {cy_a:.0f})  size=({w_a:.0f} × {h_a:.0f})",
        f"",
        f"Offsets: tx={tx:+.2f}  ty={ty:+.2f}  tw={tw:+.2f}  th={th:+.2f}",
        f"",
        f"Decode center:",
        f"  x = tx·w_a + cx_a = {tx:+.2f}·{w_a:.0f} + {cx_a:.0f} = {cx_p:.1f}",
        f"  y = ty·h_a + cy_a = {ty:+.2f}·{h_a:.0f} + {cy_a:.0f} = {cy_p:.1f}",
        f"",
        f"Decode size:",
        f"  w = w_a·exp(tw) = {w_a:.0f}·exp({tw:+.2f}) = {w_p:.1f}",
        f"  h = h_a·exp(th) = {h_a:.0f}·exp({th:+.2f}) = {h_p:.1f}",
    ]
    info = '\n'.join(info_lines)
    ax.text(
        0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
        fontfamily='monospace', va='bottom', ha='left',
        bbox=dict(facecolor='white', alpha=0.9, pad=6, edgecolor='gray'),
    )

    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plt.show()


def anchor_regression_demo(anchor=(100, 100, 260, 220)):
    """
    Launch an interactive widget demo for anchor box regression.

    Parameters
    ----------
    anchor : tuple
        (x_min, y_min, x_max, y_max) of the anchor box in pixel coords.
    """
    output = widgets.Output()

    tx_slider = widgets.FloatSlider(
        value=0.0, min=-1.0, max=1.0, step=0.05,
        description='tx (center x):', style={'description_width': '100px'},
        layout=widgets.Layout(width='450px'),
    )
    ty_slider = widgets.FloatSlider(
        value=0.0, min=-1.0, max=1.0, step=0.05,
        description='ty (center y):', style={'description_width': '100px'},
        layout=widgets.Layout(width='450px'),
    )
    tw_slider = widgets.FloatSlider(
        value=0.0, min=-1.0, max=1.0, step=0.05,
        description='tw (width):', style={'description_width': '100px'},
        layout=widgets.Layout(width='450px'),
    )
    th_slider = widgets.FloatSlider(
        value=0.0, min=-1.0, max=1.0, step=0.05,
        description='th (height):', style={'description_width': '100px'},
        layout=widgets.Layout(width='450px'),
    )

    def _on_change(change):
        with output:
            clear_output(wait=True)
            _plot_regression(anchor, tx_slider.value, ty_slider.value,
                             tw_slider.value, th_slider.value)

    for s in [tx_slider, ty_slider, tw_slider, th_slider]:
        s.observe(_on_change, names='value')

    # Initial render
    with output:
        _plot_regression(anchor, 0.0, 0.0, 0.0, 0.0)

    controls = widgets.VBox([tx_slider, ty_slider, tw_slider, th_slider])
    display(widgets.VBox([controls, output]))
