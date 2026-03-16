"""Medical Image Analysis segmentation metric visualizations.

Functions
---------
plot_erosion_boundary_demo    1x3: mask M, erode(M), boundary with formula title.
plot_hd_buildup               1x5: progressive build-up of directed Hausdorff Distance.
plot_hausdorff_steps          HD step-by-step: boundaries, distance map, worst point.
plot_hd_vs_hd95_histogram     Histogram of boundary distances with HD / HD95 lines.
plot_boundary_metrics_summary 2x3 grid: masks + NSD / HD,HD95,ASSD / DSC bars.
plot_volume_pitfall           1x4 masks: GT + 3 preds with RVD / DSC annotations.
plot_cldice_vessel_demo       1x4: GT vessel, dilated pred, gapped pred, skeletons.
create_cldice_gap_dashboard   Interactive widget: gaps + hallucinations vs all metrics.
compute_seg_metrics           Compute all standard segmentation metrics.
plot_small_object_scenarios   2×N grid: predictions + confusion maps for N scenarios.
create_small_object_dashboard Interactive widget: toggle failure scenarios, metric table.
plot_bland_altman             Bland-Altman plot for volume agreement.
plot_inter_rater_agreement    3-panel: scatter, Bland-Altman, metric comparison.
plot_skeleton_demo            1x3: tube, Y-vessel, blob with skeleton overlays.
plot_topology_examples        1x4: shapes illustrating Betti numbers β₀ and β₁.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withStroke
from scipy.ndimage import binary_erosion, distance_transform_edt


# ------------------------------------------------------------------
# 0. Erosion / boundary extraction demo
# ------------------------------------------------------------------

def plot_erosion_boundary_demo(mask: np.ndarray) -> None:
    """Show mask M, erode(M) and boundary side by side.

    Parameters
    ----------
    mask : 2D bool/int array, the binary mask to decompose.
    """
    mask = mask.astype(bool)
    eroded = binary_erosion(mask)
    boundary = mask & np.logical_not(eroded)

    # Colours
    fg_colour = np.array([0.30, 0.65, 0.90])   # blue
    eroded_colour = np.array([0.18, 0.80, 0.25])  # green
    boundary_colour = np.array([0.90, 0.20, 0.15])  # red
    bg_colour = np.array([0.95, 0.95, 0.95])     # light grey

    def _to_rgb(bool_mask, colour):
        h, w = bool_mask.shape
        img = np.zeros((h, w, 3), dtype=np.float64)
        img[bool_mask] = colour
        img[np.logical_not(bool_mask)] = bg_colour
        return img

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: original mask M
    axes[0].imshow(_to_rgb(mask, fg_colour), interpolation="nearest")
    axes[0].set_title("Mask  $M$", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: eroded mask
    axes[1].imshow(_to_rgb(eroded, eroded_colour), interpolation="nearest")
    axes[1].set_title("erode$(M)$", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: boundary
    axes[2].imshow(_to_rgb(boundary, boundary_colour), interpolation="nearest")
    axes[2].set_title(
        r"Boundary  $\partial M = M \;\cap\; \neg\,$erode$(M)$",
        fontsize=14,
        fontweight="bold",
    )
    axes[2].axis("off")

    fig.suptitle(
        r"Boundary extraction:  $\partial M \;=\; M \;\cap\; \neg\,$erode$(M)$",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 0b. HD build-up: 5-panel progressive visualisation
# ------------------------------------------------------------------

# Shared palette for sample points (matches binary_seg_demo)
_HD_POINT_COLORS = [
    (0.90, 0.10, 0.29),   # red
    (0.24, 0.71, 0.29),   # green
    (0.26, 0.39, 0.85),   # blue
    (0.96, 0.51, 0.19),   # orange
    (0.57, 0.12, 0.71),   # purple
]

_GT_BD_COLOR  = [0.0, 0.35, 0.85]   # blue
_PRED_BD_COLOR = [0.85, 0.15, 0.10]  # red


def _circular_mask_static(grid_size, cx, cy, diameter):
    yy, xx = np.mgrid[:grid_size, :grid_size]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= (diameter / 2.0) ** 2


def _extract_boundary_static(mask):
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask & np.logical_not(binary_erosion(mask))


def _make_canvas(grid_size, gt_mask, pred_mask, gt_bd, pred_bd):
    """Shared background for all five panels."""
    canvas = np.ones((grid_size, grid_size, 3)) * 0.95
    gt_only = gt_mask & np.logical_not(pred_mask)
    pred_only = pred_mask & np.logical_not(gt_mask)
    overlap = gt_mask & pred_mask
    canvas[gt_only]         = [0.78, 0.88, 1.0]
    canvas[pred_only]       = [1.0, 0.84, 0.84]
    canvas[overlap]         = [0.85, 0.80, 0.95]
    canvas[gt_bd]           = _GT_BD_COLOR
    canvas[pred_bd]         = _PRED_BD_COLOR
    canvas[gt_bd & pred_bd] = [0.5, 0.0, 0.5]
    return canvas


def plot_hd_buildup(
    grid_size: int = 128,
    gt_cx: int = 64, gt_cy: int = 64, gt_diam: int = 60,
    pred_cx: int = 79, pred_cy: int = 79, pred_diam: int = 55,
    n_sample: int = 5,
    n_fan_lines: int = 25,
) -> None:
    """Five-panel figure that progressively builds the directed Hausdorff Distance.

    Panels
    ------
    1. "Boundaries"         — ∂P and ∂R drawn on the canvas.
    2. "All distances"      — From one sample point p, lines fan-out to
                              many points on ∂R.
    3. "Nearest distance"   — Same point; only the shortest line remains
                              (= min_r d(p,r)).
    4. "Repeat for subset"  — All n_sample points with their nearest arrows.
    5. "Take the maximum"   — Same, but the longest arrow is highlighted
                              (= directed HD).

    Parameters
    ----------
    grid_size : int
    gt_cx, gt_cy, gt_diam : GT circle parameters.
    pred_cx, pred_cy, pred_diam : Pred circle parameters.
    n_sample : int, number of evenly-spaced sample points.
    n_fan_lines : int, number of ∂R points to draw in panel 2.
    """
    # ---- masks & boundaries ----
    gt_mask = _circular_mask_static(grid_size, gt_cx, gt_cy, gt_diam)
    pred_mask = _circular_mask_static(grid_size, pred_cx, pred_cy, pred_diam)
    gt_bd = _extract_boundary_static(gt_mask)
    pred_bd = _extract_boundary_static(pred_mask)
    gt_bd_yx = np.argwhere(gt_bd)
    pred_bd_yx = np.argwhere(pred_bd)
    dt_gt = distance_transform_edt(np.logical_not(gt_bd))

    # ---- sample points at fixed angles on pred boundary ----
    angles = np.linspace(0, 2 * np.pi, n_sample, endpoint=False)
    pred_r = pred_diam / 2.0
    sample_pts, nearest_gt, dists = [], [], []
    for angle in angles:
        tc = pred_cx + pred_r * np.cos(angle)
        tr = pred_cy + pred_r * np.sin(angle)
        d2 = (pred_bd_yx[:, 0] - tr) ** 2 + (pred_bd_yx[:, 1] - tc) ** 2
        idx = int(np.argmin(d2))
        pr, pc = int(pred_bd_yx[idx, 0]), int(pred_bd_yx[idx, 1])
        sample_pts.append((pr, pc))

        d2g = (gt_bd_yx[:, 0] - pr) ** 2 + (gt_bd_yx[:, 1] - pc) ** 2
        gidx = int(np.argmin(d2g))
        gr, gc = int(gt_bd_yx[gidx, 0]), int(gt_bd_yx[gidx, 1])
        nearest_gt.append((gr, gc))
        dists.append(float(dt_gt[pr, pc]))

    # pick the "featured" point for panels 2-3 (largest distance, most dramatic)
    feat = int(np.argmax(dists))
    fp_r, fp_c = sample_pts[feat]

    # subsample ∂R points for the fan lines in panel 2
    gt_n = len(gt_bd_yx)
    fan_indices = np.linspace(0, gt_n - 1, n_fan_lines, dtype=int)
    fan_pts = gt_bd_yx[fan_indices]

    # ---- figure ----
    canvas = _make_canvas(grid_size, gt_mask, pred_mask, gt_bd, pred_bd)
    fig, axes = plt.subplots(1, 5, figsize=(26, 5.2))

    legend_handles = [
        mpatches.Patch(color=_GT_BD_COLOR,  label="GT boundary  ∂R"),
        mpatches.Patch(color=_PRED_BD_COLOR, label="Pred boundary  ∂P"),
    ]

    # ── Panel 1: boundaries only ──
    ax = axes[0]
    ax.imshow(canvas, interpolation="nearest")
    ax.set_title("1.  Extract boundaries\n∂P  and  ∂R", fontsize=11,
                 fontweight="bold")
    ax.axis("off")
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7,
              framealpha=0.85)

    # ── Panel 2: one point → fan of lines to many ∂R points ──
    ax = axes[1]
    ax.imshow(canvas.copy(), interpolation="nearest")
    for (gr, gc) in fan_pts:
        ax.plot([fp_c, gc], [fp_r, gr], color="grey", lw=0.6, alpha=0.5)
    ax.plot(fp_c, fp_r, "o", color=_HD_POINT_COLORS[feat], markersize=9,
            markeredgecolor="k", markeredgewidth=0.8, zorder=5)
    ax.annotate("p", xy=(fp_c, fp_r), xytext=(7, -7),
                textcoords="offset points", fontsize=10, fontweight="bold",
                color=_HD_POINT_COLORS[feat],
                path_effects=[withStroke(linewidth=2, foreground="white")])
    ax.set_title("2.  For one point $p$,\nmeasure $d(p, r)$ to many $r \\in ∂R$",
                 fontsize=11, fontweight="bold")
    ax.axis("off")

    # ── Panel 3: same point, only the nearest line ──
    ax = axes[2]
    ax.imshow(canvas.copy(), interpolation="nearest")
    # faded fan
    for (gr, gc) in fan_pts:
        ax.plot([fp_c, gc], [fp_r, gr], color="grey", lw=0.4, alpha=0.15)
    # highlighted nearest
    ngr, ngc = nearest_gt[feat]
    ax.annotate("", xy=(ngc, ngr), xytext=(fp_c, fp_r),
                arrowprops=dict(arrowstyle="-|>", color=_HD_POINT_COLORS[feat],
                                lw=2.5, mutation_scale=14))
    ax.plot(fp_c, fp_r, "o", color=_HD_POINT_COLORS[feat], markersize=9,
            markeredgecolor="k", markeredgewidth=0.8, zorder=5)
    ax.annotate(f"{dists[feat]:.1f} px", xy=((fp_c + ngc) / 2, (fp_r + ngr) / 2),
                xytext=(6, -8), textcoords="offset points", fontsize=9,
                fontweight="bold", color=_HD_POINT_COLORS[feat],
                path_effects=[withStroke(linewidth=2, foreground="white")])
    ax.set_title("3.  Keep only the nearest\n" +
                 r"$d(p,\,∂R) = \min_r\, d(p, r)$",
                 fontsize=11, fontweight="bold")
    ax.axis("off")

    # ── Panel 4: all sample points with nearest arrows ──
    ax = axes[3]
    ax.imshow(canvas.copy(), interpolation="nearest")
    for i, ((pr, pc), (gr, gc), d) in enumerate(
            zip(sample_pts, nearest_gt, dists)):
        colour = _HD_POINT_COLORS[i]
        if d > 0.5:
            ax.annotate("", xy=(gc, gr), xytext=(pc, pr),
                        arrowprops=dict(arrowstyle="-|>", color=colour,
                                        lw=2.0, mutation_scale=12))
        ax.plot(pc, pr, "o", color=colour, markersize=8,
                markeredgecolor="k", markeredgewidth=0.8, zorder=5)
        ax.annotate(f"{d:.1f}", xy=(pc, pr), xytext=(6, -6),
                    textcoords="offset points", fontsize=8,
                    fontweight="bold", color=colour,
                    path_effects=[withStroke(linewidth=2, foreground="white")])
    ax.set_title("4.  Repeat for all\nboundary points (subset shown)",
                 fontsize=11, fontweight="bold")
    ax.axis("off")

    # ── Panel 5: highlight the max (= directed HD) ──
    ax = axes[4]
    ax.imshow(canvas.copy(), interpolation="nearest")
    max_i = int(np.argmax(dists))
    for i, ((pr, pc), (gr, gc), d) in enumerate(
            zip(sample_pts, nearest_gt, dists)):
        colour = _HD_POINT_COLORS[i]
        is_max = (i == max_i)
        lw = 3.0 if is_max else 1.2
        alpha = 1.0 if is_max else 0.35
        if d > 0.5:
            ax.annotate("", xy=(gc, gr), xytext=(pc, pr),
                        arrowprops=dict(arrowstyle="-|>", color=colour,
                                        lw=lw, mutation_scale=14,
                                        alpha=alpha))
        sz = 10 if is_max else 6
        ax.plot(pc, pr, "o", color=colour, markersize=sz,
                markeredgecolor="k", markeredgewidth=0.8, zorder=5,
                alpha=alpha)
        if is_max:
            ax.annotate(
                f"max = {d:.1f} px",
                xy=(pc, pr), xytext=(8, -10), textcoords="offset points",
                fontsize=10, fontweight="bold", color=colour,
                path_effects=[withStroke(linewidth=2.5, foreground="white")])
    ax.set_title("5.  Take the maximum\n" +
                 r"$\vec{HD} = \max_p\;\min_r\; d(p,r)$",
                 fontsize=11, fontweight="bold")
    ax.axis("off")

    fig.suptitle("Building the directed Hausdorff Distance, step by step",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 1. Hausdorff distance step-by-step
# ------------------------------------------------------------------

def plot_hausdorff_steps(pred_bd, gt_bd, gt_shape):
    """Three-panel HD visualisation: boundaries, distance map, worst point.

    Parameters
    ----------
    pred_bd : 2D bool array, predicted boundary mask.
    gt_bd   : 2D bool array, ground-truth boundary mask.
    gt_shape : tuple, shape of the original mask (H, W).

    Returns
    -------
    hd_val : float, HD value from the worst-case P->R direction.
    """
    dt_gt = distance_transform_edt(~gt_bd)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: both boundaries
    bd_img = np.ones((*gt_shape, 3)) * 0.95
    bd_img[gt_bd] = [0.0, 0.4, 0.9]
    bd_img[pred_bd] = [0.9, 0.2, 0.1]
    bd_img[gt_bd & pred_bd] = [0.5, 0.0, 0.5]
    axes[0].imshow(bd_img)
    axes[0].set_title(
        "Step 1: Extract boundaries  Pred (red)  GT (blue)", fontsize=12
    )
    axes[0].axis("off")

    # Panel 2: colour each pred boundary pixel by distance to GT
    dist_map = np.full(gt_shape, np.nan)
    dist_map[pred_bd] = dt_gt[pred_bd]
    im = axes[1].imshow(
        dist_map, cmap="hot_r", vmin=0, vmax=max(10, np.nanmax(dist_map))
    )
    axes[1].contour(gt_bd, colors="blue", linewidths=0.8)
    axes[1].set_title(
        "Step 2: min distance from each Pred boundary pt to GT boundary",
        fontsize=12,
    )
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], label="d(p, GT boundary) [px]", shrink=0.7)

    # Panel 3: highlight the point that determines HD
    worst_idx = np.argmax(dt_gt[pred_bd])
    pred_bd_coords = np.argwhere(pred_bd)
    worst_point = pred_bd_coords[worst_idx]
    axes[2].imshow(bd_img)
    axes[2].plot(
        worst_point[1], worst_point[0], "y*", markersize=20, markeredgecolor="k"
    )
    hd_val = dt_gt[pred_bd][worst_idx]
    axes[2].set_title(
        f"Step 3: Worst-case point (*): d = {hd_val:.1f} px", fontsize=12
    )
    axes[2].axis("off")

    fig.suptitle(
        "Hausdorff Distance: Step-by-Step (Pred A: protrusion)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
    return hd_val


# ------------------------------------------------------------------
# 2. HD vs HD95 outlier histogram
# ------------------------------------------------------------------

def plot_hd_vs_hd95_histogram(all_dists, hd_val, hd95_val):
    """Histogram of boundary-point distances with HD / HD95 vertical lines.

    Parameters
    ----------
    all_dists : 1D array of all boundary-point distances.
    hd_val    : float, Hausdorff distance.
    hd95_val  : float, 95th-percentile HD.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(all_dists, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(
        hd_val, color="red", linewidth=2, linestyle="--", label=f"HD = {hd_val:.1f}"
    )
    ax.axvline(
        hd95_val,
        color="orange",
        linewidth=2,
        linestyle="--",
        label=f"HD95 = {hd95_val:.1f}",
    )
    ax.set_xlabel("Boundary-point distance (px)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        "Distribution of boundary-point distances (Pred with 1 outlier)", fontsize=13
    )
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 3. Boundary metrics summary (2x3)
# ------------------------------------------------------------------

def plot_boundary_metrics_summary(
    gt_tumour, preds, pred_labels, metric_values, nsd_vals, dsc_vals
):
    """2x3 grid: top row = masks with contours, bottom row = NSD / HD,HD95,ASSD / DSC bars.

    Parameters
    ----------
    gt_tumour     : 2D array, ground-truth mask.
    preds         : list of 3 prediction masks.
    pred_labels   : list of 3 label strings, e.g. ["Pred A - protrusion", ...].
    metric_values : (3, 3) array, columns = [HD, HD95, ASSD] per case.
    nsd_vals      : list of 3 NSD values.
    dsc_vals      : list of 3 DSC values.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: masks with GT contour overlaid
    for col, (name, pred) in enumerate(zip(pred_labels, preds)):
        ax = axes[0, col]
        ax.imshow(pred, cmap="Oranges", vmin=0, vmax=1, alpha=0.6)
        ax.contour(gt_tumour, levels=[0.5], colors="blue", linewidths=2)
        ax.contour(pred, levels=[0.5], colors="red", linewidths=1.5, linestyles="--")
        ax.set_title(name, fontsize=13)
        ax.axis("off")

    # Row 2: bar charts
    metric_names = ["HD", "HD95", "ASSD"]
    case_names = [l.split(" - ")[0].replace("Pred ", "") + ": " + l.split(" - ")[1]
                  if " - " in l else l for l in pred_labels]
    x = np.arange(len(metric_names))
    width = 0.25
    colors = ["#e74c3c", "#f39c12", "#3498db"]

    for i, (cname, color) in enumerate(zip(case_names, colors)):
        axes[1, 1].bar(
            x + i * width,
            metric_values[i],
            width,
            label=cname,
            color=color,
            edgecolor="white",
        )
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels(metric_names, fontsize=12)
    axes[1, 1].set_ylabel("Distance (pixels)", fontsize=12)
    axes[1, 1].set_title("Boundary Distance Metrics Comparison", fontsize=13)
    axes[1, 1].legend(fontsize=10)

    # NSD bar
    axes[1, 0].bar(case_names, nsd_vals, color=colors, edgecolor="white")
    axes[1, 0].set_ylabel("NSD (tau=2)", fontsize=12)
    axes[1, 0].set_title("Normalized Surface Distance (tau = 2 px)", fontsize=13)
    axes[1, 0].set_ylim(0, 1.05)

    # DSC bar
    axes[1, 2].bar(case_names, dsc_vals, color=colors, edgecolor="white")
    axes[1, 2].set_ylabel("DSC", fontsize=12)
    axes[1, 2].set_title("Dice Score (for reference)", fontsize=13)
    axes[1, 2].set_ylim(0, 1.05)

    fig.suptitle(
        "Boundary Metrics Summary: Three Clinical Failure Modes",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 4. Volume pitfall (1x4)
# ------------------------------------------------------------------

def plot_volume_pitfall(gt_tumour, preds, pred_titles):
    """1x4 panels: GT mask + 3 prediction masks with RVD/DSC in titles.

    Parameters
    ----------
    gt_tumour   : 2D array, ground-truth mask.
    preds       : list of 3 prediction masks [pred_A, pred_B, pred_C].
    pred_titles : list of 3 title strings (include RVD and DSC info).
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    axes[0].imshow(gt_tumour, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title(f"Ground Truth\nVol = {gt_tumour.sum():.0f} px", fontsize=12)
    axes[0].axis("off")

    for i, (pred, title) in enumerate(zip(preds, pred_titles)):
        ax = axes[i + 1]
        ax.imshow(pred, cmap="Oranges", vmin=0, vmax=1)
        if i == 2:  # Pred C: add GT contour
            ax.contour(gt_tumour, levels=[0.5], colors="blue", linewidths=2)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.suptitle(
        "Volume Pitfall: Pred C has near-perfect RVD but terrible spatial overlap",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 5. clDice vessel demo (1x4)
# ------------------------------------------------------------------

def plot_cldice_vessel_demo(
    gt_vessel,
    pred_thick,
    pred_gapped,
    skel_gt,
    skel_gapped,
    dsc_thick,
    cldice_thick,
    dsc_gapped,
    cldice_gapped,
):
    """1x4 panels: GT vessel tree, dilated pred, gapped pred, skeleton overlay.

    Parameters
    ----------
    gt_vessel     : 2D bool array.
    pred_thick    : 2D bool array, over-segmented prediction (topology correct).
    pred_gapped   : 2D bool array, prediction with connectivity gaps.
    skel_gt       : 2D bool array, GT skeleton.
    skel_gapped   : 2D bool array, skeleton of gapped prediction.
    dsc_thick, cldice_thick   : float, metrics for thick prediction.
    dsc_gapped, cldice_gapped : float, metrics for gapped prediction.
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    axes[0].imshow(gt_vessel, cmap="gray")
    axes[0].set_title("GT vessel tree", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(pred_thick, cmap="gray")
    axes[1].set_title(
        f"Pred A: dilated\nDSC={dsc_thick:.3f}  clDice={cldice_thick:.3f}",
        fontsize=12,
    )
    axes[1].axis("off")

    axes[2].imshow(pred_gapped, cmap="gray")
    axes[2].set_title(
        f"Pred B: small gaps in branches\nDSC={dsc_gapped:.3f}  clDice={cldice_gapped:.3f}",
        fontsize=12,
    )
    axes[2].axis("off")

    # Skeleton overlay: show which GT skeleton pixels are covered vs missed
    skel_img = np.zeros((*gt_vessel.shape, 3))
    skel_img[gt_vessel] = [0.85, 0.85, 0.85]          # GT mask in light gray
    covered = skel_gt & pred_gapped                     # covered by pred B
    missed = skel_gt & ~pred_gapped                     # missed by pred B
    skel_img[covered] = [0.2, 0.5, 1.0]                # blue = covered
    skel_img[missed] = [1.0, 0.15, 0.15]               # red = missed
    axes[3].imshow(skel_img)
    axes[3].set_title(
        "GT skeleton on Pred B\nBlue = covered, Red = missed", fontsize=12
    )
    axes[3].axis("off")

    fig.suptitle(
        "clDice: Detecting Connectivity Gaps in Tubular Structures",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 5b. Interactive clDice topology explorer (widget)
# ------------------------------------------------------------------

def _dice_score(pred, gt):
    """Dice Similarity Coefficient for two binary masks."""
    p, g = pred.astype(bool), gt.astype(bool)
    intersection = np.sum(p & g)
    total = np.sum(p) + np.sum(g)
    return 2 * intersection / total if total > 0 else 1.0


def _cldice(pred, gt):
    """Compute centerline Dice (clDice) for tubular structures.

    Returns (clDice, TP_cl, TS_cl).
    """
    from skimage.morphology import skeletonize

    pred_bool, gt_bool = pred.astype(bool), gt.astype(bool)
    skel_pred = skeletonize(pred_bool)
    skel_gt = skeletonize(gt_bool)

    if skel_pred.sum() == 0 and skel_gt.sum() == 0:
        return 1.0, 1.0, 1.0
    if skel_pred.sum() == 0 or skel_gt.sum() == 0:
        return 0.0, 0.0, 0.0

    tp_cl = np.sum(skel_pred & gt_bool) / np.sum(skel_pred)
    ts_cl = np.sum(skel_gt & pred_bool) / np.sum(skel_gt)

    if tp_cl + ts_cl == 0:
        return 0.0, tp_cl, ts_cl
    return 2 * tp_cl * ts_cl / (tp_cl + ts_cl), tp_cl, ts_cl


def _draw_segment_on(img, r0, c0, r1, c1, width=3, size=250):
    """Draw a line segment of given pixel width onto *img* in-place."""
    length = max(abs(r1 - r0), abs(c1 - c0))
    if length == 0:
        return
    hw = width // 2
    for t in np.linspace(0, 1, length * 3):
        r = int(round(r0 + t * (r1 - r0)))
        c = int(round(c0 + t * (c1 - c0)))
        img[max(0, r - hw):min(size, r + hw + 1),
            max(0, c - hw):min(size, c + hw + 1)] = True


def _build_vessel_tree(size=250):
    """Build the synthetic vessel tree: thick trunk + 6 thin branches.

    Returns (gt_vessel, branches, halluc_branches) where branches is a
    list of (r0, c0, r1, c1) tuples for the 6 GT branches, and
    halluc_branches defines 2 potential hallucinated branches.
    """
    gt = np.zeros((size, size), dtype=bool)
    _draw_segment_on(gt, 125, 10, 125, 240, width=11, size=size)

    branches = [
        (125, 40, 40, 60),       # upper-left
        (125, 80, 40, 120),      # upper-mid-left
        (125, 120, 40, 160),     # upper-mid-right
        (125, 160, 40, 200),     # upper-right
        (125, 60, 210, 100),     # lower-left
        (125, 140, 210, 180),    # lower-right
    ]
    for r0, c0, r1, c1 in branches:
        _draw_segment_on(gt, r0, c0, r1, c1, width=3, size=size)

    # Two hallucinated branches (NOT in the GT).
    # These fork off the trunk at positions where no real branch exists.
    halluc_branches = [
        (125, 100, 220, 60),     # lower-mid, going down-left
        (125, 200, 30, 220),     # upper-far-right, going up-right
    ]

    return gt, branches, halluc_branches


def _make_prediction(gt_vessel, branches, gap_fracs,
                     halluc_branches, halluc_flags, size=250):
    """Build a prediction mask with gaps and/or hallucinated branches.

    Parameters
    ----------
    gt_vessel       : 2D bool, ground-truth vessel mask.
    branches        : list of 6 (r0, c0, r1, c1) GT branch endpoints.
    gap_fracs       : list of 6 floats in [0, 0.5], gap fraction per branch.
    halluc_branches : list of 2 (r0, c0, r1, c1) hallucination endpoints.
    halluc_flags    : list of 2 bools, whether each hallucination is active.
    size            : int, image dimension.

    Returns
    -------
    pred : 2D bool array.
    """
    pred = gt_vessel.copy()

    # Erase gaps in real branches
    for i, (r0, c0, r1, c1) in enumerate(branches):
        frac = gap_fracs[i]
        if frac <= 0:
            continue
        for t in np.linspace(0.5 - frac / 2, 0.5 + frac / 2, 600):
            r = int(round(r0 + t * (r1 - r0)))
            c = int(round(c0 + t * (c1 - c0)))
            pred[max(0, r - 2):min(size, r + 3),
                 max(0, c - 2):min(size, c + 3)] = False

    # Draw hallucinated branches
    for j, (r0, c0, r1, c1) in enumerate(halluc_branches):
        if halluc_flags[j]:
            _draw_segment_on(pred, r0, c0, r1, c1, width=3, size=size)

    return pred


def _compute_all_metrics(pred, gt):
    """Compute overlap, boundary, topology, and volume metrics.

    Returns a dict with keys: DSC, clDice, TS_cl, TP_cl, HD, HD95,
    ASSD, vol_diff_pct.
    """
    dsc = _dice_score(pred, gt)
    cl, tp_cl, ts_cl = _cldice(pred, gt)

    # Boundary metrics
    gt_bd = _extract_boundary_static(gt)
    pred_bd = _extract_boundary_static(pred)

    if not gt_bd.any() or not pred_bd.any():
        hd = hd95 = assd = float("nan")
    else:
        dt_gt = distance_transform_edt(~gt_bd)
        dt_pred = distance_transform_edt(~pred_bd)
        d_pred_to_gt = dt_gt[pred_bd]
        d_gt_to_pred = dt_pred[gt_bd]
        hd = float(max(d_pred_to_gt.max(), d_gt_to_pred.max()))
        hd95 = float(max(np.percentile(d_pred_to_gt, 95),
                         np.percentile(d_gt_to_pred, 95)))
        assd = float((d_pred_to_gt.sum() + d_gt_to_pred.sum())
                     / (len(d_pred_to_gt) + len(d_gt_to_pred)))

    vol_gt = int(gt.sum())
    vol_pred = int(pred.sum())
    vol_diff_pct = 100.0 * abs(vol_pred - vol_gt) / vol_gt if vol_gt > 0 else float("nan")

    return dict(
        DSC=dsc, clDice=cl, TS_cl=ts_cl, TP_cl=tp_cl,
        HD=hd, HD95=hd95, ASSD=assd, vol_diff_pct=vol_diff_pct,
    )


def create_cldice_gap_dashboard() -> None:
    """Interactive clDice topology explorer.

    Students control gap sizes in 6 branches (affects TS_cl) and toggle
    2 hallucinated branches (affects TP_cl). A grouped bar chart shows
    how each metric family responds, making it clear that only the
    topology metrics (clDice, TP_cl, TS_cl) detect these structural
    errors.
    """
    import ipywidgets as widgets
    from IPython.display import clear_output, display
    from skimage.morphology import skeletonize

    SIZE = 250
    gt_vessel, branches, halluc_branches = _build_vessel_tree(SIZE)
    skel_gt = skeletonize(gt_vessel)

    # ── Branch labels and gap sliders ─────────────────────────
    branch_labels = [
        "Upper-left",
        "Upper-mid-left",
        "Upper-mid-right",
        "Upper-right",
        "Lower-left",
        "Lower-right",
    ]

    gap_sliders = {}
    for i, label in enumerate(branch_labels):
        gap_sliders[i] = widgets.FloatSlider(
            value=0.0,
            min=0.0,
            max=0.50,
            step=0.01,
            description=f"Gap {i + 1} ({label}):",
            style={"description_width": "175px"},
            layout=widgets.Layout(width="370px"),
            continuous_update=False,
            readout_format=".0%",
        )

    # ── Hallucination checkboxes ──────────────────────────────
    halluc_cbs = [
        widgets.Checkbox(
            value=False,
            description="Hallucinate branch A (lower-mid)",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="330px"),
        ),
        widgets.Checkbox(
            value=False,
            description="Hallucinate branch B (upper-far-right)",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="330px"),
        ),
    ]

    # ── Preset buttons ────────────────────────────────────────
    _PRESETS = {
        "No errors": dict(
            gaps=[0.0] * 6, halluc=[False, False],
        ),
        "Gaps only (TS_cl drops)": dict(
            gaps=[0.12] * 6, halluc=[False, False],
        ),
        "Hallucinations only (TP_cl drops)": dict(
            gaps=[0.0] * 6, halluc=[True, True],
        ),
        "Both (clDice drops)": dict(
            gaps=[0.12] * 6, halluc=[True, True],
        ),
        "Large gaps": dict(
            gaps=[0.30] * 6, halluc=[False, False],
        ),
    }

    preset_buttons = {}
    for label in _PRESETS:
        short = label.split("(")[0].strip()
        preset_buttons[label] = widgets.Button(
            description=short,
            button_style="info",
            layout=widgets.Layout(width="auto"),
        )
    reset_btn = widgets.Button(
        description="Reset",
        button_style="warning",
        icon="refresh",
        layout=widgets.Layout(width="90px"),
    )

    out = widgets.Output()

    def _apply_preset(preset):
        """Apply a preset without triggering N redraws."""
        all_controls = list(gap_sliders.values()) + halluc_cbs
        for c in all_controls:
            c.unobserve(_on_change, names="value")
        for i, v in enumerate(preset["gaps"]):
            gap_sliders[i].value = v
        for j, v in enumerate(preset["halluc"]):
            halluc_cbs[j].value = v
        for c in all_controls:
            c.observe(_on_change, names="value")
        _on_change()

    for label, cfg in _PRESETS.items():
        preset_buttons[label].on_click(
            lambda _, c=cfg: _apply_preset(c)
        )
    reset_btn.on_click(
        lambda _: _apply_preset(dict(gaps=[0.0] * 6, halluc=[False, False]))
    )

    # ── Metric display config ─────────────────────────────────
    # Each group: (category_label, colour, [(key, display_name), ...])
    _METRIC_GROUPS = [
        ("Overlap", "#4472C4", [
            ("DSC", "DSC"),
        ]),
        ("Topology", "#E07B39", [
            ("clDice", "clDice"),
            ("TP_cl", "TP_cl (precision)"),
            ("TS_cl", "TS_cl (sensitivity)"),
        ]),
        ("Boundary", "#70AD47", [
            ("HD95", "HD95 (px)"),
            ("ASSD", "ASSD (px)"),
        ]),
        ("Volume", "#7B7B7B", [
            ("vol_diff_pct", "|Vol Diff| (%)"),
        ]),
    ]

    # ── Draw function ─────────────────────────────────────────
    def _draw(gap_fracs, halluc_flags):
        pred = _make_prediction(
            gt_vessel, branches, gap_fracs,
            halluc_branches, halluc_flags, SIZE,
        )
        metrics = _compute_all_metrics(pred, gt_vessel)
        skel_pred = skeletonize(pred)

        fig = plt.figure(figsize=(22, 10))
        gs = fig.add_gridspec(
            2, 3, height_ratios=[1.1, 1], hspace=0.35, wspace=0.30,
        )

        # ── Top row: three visualisation panels ──────────────
        ax_gt = fig.add_subplot(gs[0, 0])
        ax_gt.imshow(gt_vessel, cmap="gray")
        ax_gt.set_title("GT Vessel Tree", fontsize=12)
        ax_gt.axis("off")

        ax_pred = fig.add_subplot(gs[0, 1])
        ax_pred.imshow(pred, cmap="gray")
        n_gaps = sum(1 for f in gap_fracs if f > 0)
        n_halluc = sum(halluc_flags)
        parts = []
        if n_gaps:
            parts.append(f"{n_gaps} gap{'s' if n_gaps != 1 else ''}")
        if n_halluc:
            parts.append(
                f"{n_halluc} halluc. branch"
                f"{'es' if n_halluc != 1 else ''}"
            )
        subtitle = ", ".join(parts) if parts else "no errors"
        ax_pred.set_title(f"Prediction\n({subtitle})", fontsize=12)
        ax_pred.axis("off")

        # Skeleton overlay: shows BOTH TS_cl and TP_cl issues
        #   - Blue:   GT skeleton covered by prediction (TS_cl OK)
        #   - Red:    GT skeleton missed by prediction  (TS_cl penalty)
        #   - Orange: Pred skeleton outside GT mask      (TP_cl penalty)
        ax_skel = fig.add_subplot(gs[0, 2])
        skel_img = np.zeros((*gt_vessel.shape, 3))
        skel_img[gt_vessel] = [0.85, 0.85, 0.85]       # GT mask in gray
        skel_img[pred & ~gt_vessel] = [0.70, 0.70, 0.70]  # halluc area

        covered = skel_gt & pred
        missed = skel_gt & ~pred
        halluc_skel = skel_pred & ~gt_vessel.astype(bool)

        skel_img[covered] = [0.2, 0.5, 1.0]             # blue
        skel_img[missed] = [1.0, 0.15, 0.15]            # red
        skel_img[halluc_skel] = [1.0, 0.65, 0.0]        # orange

        ax_skel.imshow(skel_img)
        ax_skel.set_title(
            "Skeleton Overlay\n"
            "Blue = GT skel covered (TS_cl OK), "
            "Red = GT skel missed (TS_cl penalty)\n"
            "Orange = Pred skel outside GT (TP_cl penalty)",
            fontsize=10,
        )
        ax_skel.axis("off")

        # ── Bottom row: single vertical grouped bar chart ─────
        ax_bar = fig.add_subplot(gs[1, :])

        # Flatten all metrics into one sequence, tracking group boundaries
        all_labels = []
        all_vals = []
        all_colours = []
        group_boundaries = []   # x positions where a new group starts
        group_centers = []      # (center_x, label, colour)
        group_colours = []
        idx = 0
        for cat_label, colour, metric_list in _METRIC_GROUPS:
            group_boundaries.append(idx)
            group_start = idx
            for key, display_name in metric_list:
                val = metrics[key]
                if np.isnan(val):
                    val = 0.0
                all_labels.append(display_name)
                all_vals.append(val)
                all_colours.append(colour)
                idx += 1
            group_centers.append(
                ((group_start + idx - 1) / 2, cat_label, colour)
            )
        n_bars = len(all_labels)

        # Normalise boundary/volume metrics to [0,1] for a unified y-axis.
        # Store the raw values for annotation.
        raw_vals = list(all_vals)
        _UNIT_KEYS = {"HD95", "ASSD", "vol_diff_pct"}
        # Max across those bars (for scaling)
        unit_max = max(
            (v for k, v in zip(
                [key for _, _, ml in _METRIC_GROUPS for key, _ in ml],
                raw_vals,
            ) if k in _UNIT_KEYS),
            default=1.0,
        ) or 1.0
        normalised = []
        key_list = [key for _, _, ml in _METRIC_GROUPS for key, _ in ml]
        for i_bar, (key, val) in enumerate(zip(key_list, all_vals)):
            if key in _UNIT_KEYS:
                normalised.append(val / unit_max)
            else:
                normalised.append(val)

        x_pos = np.arange(n_bars)
        bars = ax_bar.bar(
            x_pos, normalised, color=all_colours,
            edgecolor="black", linewidth=0.6, width=0.6,
        )

        # Value labels above each bar
        for i_bar, (bar_rect, raw_v, key) in enumerate(
            zip(bars, raw_vals, key_list)
        ):
            if key in _UNIT_KEYS:
                txt = f"{raw_v:.1f}"
            else:
                txt = f"{raw_v:.4f}"
            ax_bar.text(
                bar_rect.get_x() + bar_rect.get_width() / 2,
                bar_rect.get_height() + 0.02,
                txt, ha="center", va="bottom", fontsize=9,
            )

        # X-axis: metric names (rotated to avoid overlap)
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(all_labels, fontsize=10, rotation=25,
                               ha="right")

        # Y-axis
        ax_bar.set_ylim(0, 1.22)
        ax_bar.set_ylabel("Score  (higher is better for overlap/topology;\n"
                          "lower is better for boundary/volume)",
                          fontsize=10)
        ax_bar.axhline(1.0, color="gray", linestyle="--", alpha=0.3,
                       linewidth=0.8)
        ax_bar.grid(axis="y", alpha=0.2)

        # Vertical separator lines between groups
        for bx in group_boundaries[1:]:
            ax_bar.axvline(bx - 0.5, color="gray", linestyle="-",
                           alpha=0.45, linewidth=1.0)

        # Group labels along the top
        for cx, cat_name, colour in group_centers:
            ax_bar.text(
                cx, 1.16, cat_name, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=colour,
            )

        fig.suptitle(
            "clDice Topology Explorer: Which Metrics Detect Structural "
            "Errors?",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    # ── Callbacks ─────────────────────────────────────────────
    def _on_change(_=None):
        fracs = [gap_sliders[i].value for i in range(6)]
        flags = [cb.value for cb in halluc_cbs]
        with out:
            clear_output(wait=True)
            _draw(fracs, flags)

    for s in gap_sliders.values():
        s.observe(_on_change, names="value")
    for cb in halluc_cbs:
        cb.observe(_on_change, names="value")

    # Initial render
    with out:
        _draw([0.0] * 6, [False, False])

    # ── Layout ────────────────────────────────────────────────
    gap_col_left = widgets.VBox(
        [gap_sliders[i] for i in range(3)],
        layout=widgets.Layout(margin="0 16px 0 0"),
    )
    gap_col_right = widgets.VBox(
        [gap_sliders[i] for i in range(3, 6)],
    )
    gap_row = widgets.HBox([gap_col_left, gap_col_right])

    halluc_row = widgets.HBox(
        halluc_cbs,
        layout=widgets.Layout(margin="0 0 4px 0"),
    )

    preset_row = widgets.HBox(
        [preset_buttons[k] for k in _PRESETS] + [reset_btn],
        layout=widgets.Layout(margin="4px 0 8px 0"),
    )

    gap_label = widgets.HTML(
        "<b style='font-size:12px'>Gap size per branch "
        "(affects TS_cl):</b>"
    )
    halluc_label = widgets.HTML(
        "<b style='font-size:12px'>Hallucinated branches "
        "(affects TP_cl):</b>"
    )

    ui = widgets.VBox([
        preset_row,
        gap_label, gap_row,
        halluc_label, halluc_row,
    ])
    display(widgets.VBox([ui, out]))


# ------------------------------------------------------------------
# 6. Small object bias – metric computation + scenario visualisation
# ------------------------------------------------------------------
def compute_seg_metrics(pred, gt, nsd_tau=2.0):
    """Compute all standard segmentation metrics between two binary masks.

    Returns dict with keys: dsc, iou, hd, hd95, assd, nsd, avd, rvd.
    """
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(gt, dtype=bool)

    intersection = int((p & g).sum())
    p_sum, g_sum = int(p.sum()), int(g.sum())
    union = int((p | g).sum())

    dsc = 2 * intersection / (p_sum + g_sum) if (p_sum + g_sum) > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0

    # Boundaries
    p_bnd = p & ~binary_erosion(p) if p.any() else p
    g_bnd = g & ~binary_erosion(g) if g.any() else g

    dt_p = distance_transform_edt(~p_bnd)
    dt_g = distance_transform_edt(~g_bnd)

    d_p2g = dt_g[p_bnd] if p_bnd.any() else np.array([])
    d_g2p = dt_p[g_bnd] if g_bnd.any() else np.array([])

    if d_p2g.size > 0 and d_g2p.size > 0:
        hd = float(max(d_p2g.max(), d_g2p.max()))
        hd95 = float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))
        assd = float((d_p2g.sum() + d_g2p.sum()) / (d_p2g.size + d_g2p.size))
        nsd = float(
            ((d_p2g <= nsd_tau).sum() + (d_g2p <= nsd_tau).sum())
            / (d_p2g.size + d_g2p.size)
        )
    else:
        hd = hd95 = assd = nsd = 0.0

    avd = abs(p_sum - g_sum)
    rvd = (p_sum - g_sum) / g_sum if g_sum > 0 else 0.0

    return dict(dsc=dsc, iou=iou, hd=hd, hd95=hd95, assd=assd, nsd=nsd,
                avd=avd, rvd=rvd)


def plot_small_object_scenarios(gt_scene, scenarios, met_positions,
                                met_radius, scene_size):
    """2-by-N grid: prediction masks (top) and confusion maps (bottom).

    Parameters
    ----------
    gt_scene      : 2D array, ground-truth mask (binary).
    scenarios     : list of (label, pred_array) tuples.
    met_positions : list of (cy, cx) offsets from image centre.
    met_radius    : int, metastasis radius in pixels.
    scene_size    : int, image side length.
    """
    n = len(scenarios)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10), squeeze=False)
    half = scene_size // 2

    gt_b = gt_scene.astype(bool)
    gt_boundary = gt_b & ~binary_erosion(gt_b)

    for col, (label, pred) in enumerate(scenarios):
        # Top row: prediction with GT contour overlay
        img = np.ones((*gt_scene.shape, 3)) * 0.95
        img[pred > 0] = [1.0, 0.82, 0.50]
        img[gt_boundary] = [0.2, 0.4, 0.9]

        axes[0, col].imshow(img)
        axes[0, col].set_title(label, fontsize=13, fontweight="bold")
        for cy, cx in met_positions:
            axes[0, col].add_patch(plt.Circle(
                (cx + half, cy + half), met_radius + 3,
                fill=False, color="red", lw=1.5, ls="--",
            ))
        axes[0, col].axis("off")

        # Bottom row: confusion map
        conf = np.ones((*gt_scene.shape, 3)) * 0.92
        conf[(gt_scene > 0) & (pred > 0)] = [0.2, 0.8, 0.2]
        conf[(gt_scene == 0) & (pred > 0)] = [0.9, 0.3, 0.1]
        conf[(gt_scene > 0) & (pred == 0)] = [0.9, 0.7, 0.1]

        axes[1, col].imshow(conf)
        axes[1, col].set_title("TP / FP / FN", fontsize=11)
        for cy, cx in met_positions:
            axes[1, col].add_patch(plt.Circle(
                (cx + half, cy + half), met_radius + 3,
                fill=False, color="red", lw=1.5, ls="--",
            ))
        axes[1, col].axis("off")

    legend_elems = [
        mpatches.Patch(color=[0.2, 0.8, 0.2], label="TP"),
        mpatches.Patch(color=[0.9, 0.3, 0.1], label="FP"),
        mpatches.Patch(color=[0.9, 0.7, 0.1], label="FN"),
        plt.Line2D([0], [0], color=[0.2, 0.4, 0.9], lw=2,
                   label="GT boundary"),
        plt.Line2D([0], [0], color="red", ls="--", lw=1.5,
                   label="Met location"),
    ]
    axes[1, -1].legend(handles=legend_elems, loc="lower right", fontsize=9)

    fig.suptitle("Small Object Bias: Three Scenarios, Same GT",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 7. Small object bias – interactive dashboard
# ------------------------------------------------------------------

# Colour helpers matching binary_seg_demo.py behaviour
def _overlap_color(v):
    if np.isnan(v):
        return "#cccccc"
    return "#2ca02c" if v >= 0.8 else "#e89b0c" if v >= 0.5 else "#d62728"


def _dist_color(v, good=3.0, warn=10.0):
    if np.isnan(v):
        return "#cccccc"
    return "#2ca02c" if v <= good else "#e89b0c" if v <= warn else "#d62728"


def _vol_color_avd(v, total):
    if total == 0:
        return "#cccccc"
    frac = v / total
    return "#2ca02c" if frac < 0.05 else "#e89b0c" if frac < 0.15 else "#d62728"


def _vol_color_rvd(v):
    if np.isnan(v):
        return "#cccccc"
    a = abs(v)
    return "#2ca02c" if a < 0.05 else "#e89b0c" if a < 0.15 else "#d62728"


def create_small_object_dashboard(
    nsd_tau: float = 2.0,
) -> None:
    """Interactive explorer for small-object bias in segmentation metrics.

    Five scenarios split into two groups:
    - A, B, C: 256x256 scene with a large object + 5 small objects.
    - D, E: 256x256 scene with only 5 small objects (no large object).
    Each scenario has its own ground truth and prediction.
    """
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    from matplotlib.gridspec import GridSpec
    import matplotlib.colors as mcolors

    # -- geometry -----------------------------------------------------------
    S = 256
    half = S // 2
    yy, xx = np.ogrid[-half:half, -half:half]

    # Large object: ellipse at centre
    large_obj = (xx / 90) ** 2 + (yy / 65) ** 2 <= 1
    large_px = int(large_obj.sum())

    # 5 small objects (all outside the large object)
    small_pos = [(0, 97), (0, -97), (70, 0), (-70, 0), (50, 68)]
    small_r = 5
    small_masks = []
    gt_small_only = np.zeros((S, S), dtype=bool)
    for cy, cx in small_pos:
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= small_r ** 2
        gt_small_only[m] = True
        small_masks.append(m)

    gt_with_large = gt_small_only | large_obj
    n_small = len(small_pos)
    small_px = int(gt_small_only.sum())

    # Slightly shifted large-object prediction (realistic)
    large_pred = ((xx - 1) / 89) ** 2 + ((yy + 1) / 64) ** 2 <= 1

    def _large_base():
        p = np.zeros((S, S), dtype=bool)
        p[large_pred] = True
        return p

    # -- predictions --------------------------------------------------------
    # A: large object segmented, all small objects missed
    pred_A = _large_base()

    # B: over-segmented blob covers large object + all small objects
    pred_B = np.zeros((S, S), dtype=bool)
    pred_B[(xx / 130) ** 2 + (yy / 110) ** 2 <= 1] = True

    # C: large object + 3 of 5 small objects found
    pred_C = _large_base()
    for cy, cx in small_pos[:3]:
        pred_C[((xx - cx - 1) ** 2 + (yy - cy) ** 2) <= small_r ** 2] = True

    # D: small-only GT; each small object partially overlapped (shifted 3 px)
    pred_D = np.zeros((S, S), dtype=bool)
    for cy, cx in small_pos:
        pred_D[((xx - cx - 3) ** 2 + (yy - cy) ** 2) <= small_r ** 2] = True

    # E: small-only GT; 5 predicted objects at wrong locations (same volume)
    pred_E = np.zeros((S, S), dtype=bool)
    wrong_pos = [(110, 0), (-110, 0), (0, 110), (80, 80), (-80, -80)]
    for cy, cx in wrong_pos:
        pred_E[((xx - cx) ** 2 + (yy - cy) ** 2) <= small_r ** 2] = True

    # -- scenario registry: (gt, pred, label for GT summary) ----------------
    scenarios = {
        "A: Small objects missed": (
            gt_with_large, pred_A, True,
        ),
        "B: Over-segmentation": (
            gt_with_large, pred_B, True,
        ),
        "C: 3 of 5 found": (
            gt_with_large, pred_C, True,
        ),
        "D: Partial overlap (small only)": (
            gt_small_only, pred_D, False,
        ),
        "E: Wrong locations (small only)": (
            gt_small_only, pred_E, False,
        ),
    }

    _EXPLANATIONS = {
        "A: Small objects missed": (
            "The large object is segmented well, but every small object "
            "is missed. DSC and IoU stay above 0.94 because the large "
            "object dominates the pixel count. Lesion sensitivity: 0/5. "
            "Conclusion: DSC alone is not enough."
        ),
        "B: Over-segmentation": (
            "A single bloated prediction covers the large object and "
            "engulfs all five small objects by accident. Lesion "
            "sensitivity reads 5/5 even though no small object was "
            "individually detected. DSC drops and HD rises, but the "
            "detection score looks perfect. A useless prediction that "
            "scores well on the wrong metric."
        ),
        "C: 3 of 5 found": (
            "Three small objects are correctly segmented; two are missed. "
            "Pixel-level metrics barely change compared to A (all missed) "
            "because the small objects contribute fewer than 250 pixels "
            "combined. This is the main point: missed lesions are "
            "invisible in DSC/IoU when a large structure dominates."
        ),
        "D: Partial overlap (small only)": (
            "No large object; only the 5 small objects are in the GT. "
            "Each predicted object is shifted by 3 px, giving roughly "
            "60 %% per-object overlap. All 5 are 'detected' (sensitivity "
            "5/5), but the actual segmentation quality per object is "
            "poor. Notice how quickly DSC rises from 0 with even small "
            "overlaps. Partial overlaps inflate DSC rapidly for small "
            "round objects."
        ),
        "E: Wrong locations (small only)": (
            "No large object. Five predicted objects of the correct size "
            "are placed at entirely wrong locations. AVD is near 0 and "
            "RVD is close to 0 because the total predicted volume "
            "matches the GT volume. Yet DSC is 0 and detection is 0/5. "
            "Volume and alignment metrics are meaningless without "
            "spatial correspondence."
        ),
    }

    # -- widgets ------------------------------------------------------------
    scenario_dd = widgets.Dropdown(
        options=list(scenarios),
        value="A: Small objects missed",
        description="Scenario:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="360px"),
    )
    out = widgets.Output()

    # -- draw function ------------------------------------------------------
    def _draw(scenario_name):
        gt, pred, has_large = scenarios[scenario_name]
        met = compute_seg_metrics(pred, gt.astype(float), nsd_tau=nsd_tau)

        # Object-level metrics
        p_bool = pred.astype(bool)
        detected = sum(1 for m in small_masks if (p_bool & m).any())
        recalls = []
        if has_large:
            recalls.append(
                float((p_bool & large_obj).sum() / large_obj.sum())
            )
        for m in small_masks:
            recalls.append(
                float((p_bool & m).sum() / m.sum()) if m.sum() else 0.0
            )
        mean_recall = float(np.mean(recalls))

        gt_boundary = gt & ~binary_erosion(gt)

        # --- figure layout -------------------------------------------------
        fig = plt.figure(figsize=(20, 8))
        gs = GridSpec(
            2, 3, width_ratios=[1.2, 1.2, 1.0],
            height_ratios=[1, 0.12],
            hspace=0.20, wspace=0.15,
        )

        ax_pred = fig.add_subplot(gs[0, 0])
        ax_conf = fig.add_subplot(gs[0, 1])
        ax_tbl = fig.add_subplot(gs[0, 2])
        ax_note = fig.add_subplot(gs[1, :])

        # --- prediction overlay --------------------------------------------
        img = np.ones((S, S, 3)) * 0.95
        img[pred > 0] = [1.0, 0.82, 0.50]
        img[gt_boundary] = [0.2, 0.4, 0.9]
        ax_pred.imshow(img, aspect="equal")
        ax_pred.set_title(scenario_name, fontsize=14, fontweight="bold")
        for cy, cx in small_pos:
            ax_pred.add_patch(plt.Circle(
                (cx + half, cy + half), small_r + 3,
                fill=False, color="red", lw=1.5, ls="--"))
        ax_pred.axis("off")

        # --- confusion map -------------------------------------------------
        conf = np.ones((S, S, 3)) * 0.92
        conf[gt & (pred > 0)] = [0.2, 0.8, 0.2]       # TP
        conf[~gt & (pred > 0)] = [0.9, 0.3, 0.1]      # FP
        conf[gt & (pred == 0)] = [0.9, 0.7, 0.1]       # FN
        ax_conf.imshow(conf, aspect="equal")
        ax_conf.set_title("TP / FP / FN", fontsize=12)
        for cy, cx in small_pos:
            ax_conf.add_patch(plt.Circle(
                (cx + half, cy + half), small_r + 3,
                fill=False, color="red", lw=1.5, ls="--"))
        ax_conf.axis("off")

        legend_elems = [
            mpatches.Patch(color=[0.2, 0.8, 0.2], label="TP"),
            mpatches.Patch(color=[0.9, 0.3, 0.1], label="FP"),
            mpatches.Patch(color=[0.9, 0.7, 0.1], label="FN"),
            plt.Line2D([0], [0], color=[0.2, 0.4, 0.9], lw=2,
                       label="GT boundary"),
            plt.Line2D([0], [0], color="red", ls="--", lw=1.5,
                       label="Small object"),
        ]
        ax_conf.legend(
            handles=legend_elems, loc="upper center",
            bbox_to_anchor=(0.5, -0.04), ncol=5, fontsize=9,
            frameon=False,
        )

        # --- colour-coded metric table -------------------------------------
        ax_tbl.axis("off")
        if has_large:
            tbl_title = (
                f"GT: 1 large ({large_px:,} px) + "
                f"{n_small} small ({small_px:,} px)"
            )
        else:
            tbl_title = f"GT: {n_small} small objects ({small_px:,} px)"
        ax_tbl.set_title(tbl_title, fontsize=11, fontweight="bold", pad=10)

        vol_gt = int(gt.sum())
        rows = [
            ("Overlap", "DSC", f"{met['dsc']:.4f}",
             _overlap_color(met["dsc"]), "higher better"),
            ("", "IoU", f"{met['iou']:.4f}",
             _overlap_color(met["iou"]), "higher better"),
            ("Boundary", "HD (px)", f"{met['hd']:.1f}",
             _dist_color(met["hd"]), "lower better"),
            ("", "HD95 (px)", f"{met['hd95']:.1f}",
             _dist_color(met["hd95"]), "lower better"),
            ("", "ASSD (px)", f"{met['assd']:.2f}",
             _dist_color(met["assd"]), "lower better"),
            ("", "NSD", f"{met['nsd']:.4f}",
             _overlap_color(met["nsd"]), "higher better"),
            ("Volume", "AVD (px)", f"{met['avd']:,}",
             _vol_color_avd(met["avd"], vol_gt), "lower better"),
            ("", "RVD", f"{met['rvd']:+.4f}",
             _vol_color_rvd(met["rvd"]), "closer to 0"),
            ("Object", "Obj. sens.",
             f"{detected}/{n_small}",
             "#2ca02c" if detected == n_small else "#d62728",
             "higher better"),
            ("", "Mean recall", f"{mean_recall:.3f}",
             _overlap_color(mean_recall), "higher better"),
        ]

        col_labels = ["Group", "Metric", "Value", "Direction"]
        cell_text = [[g, n, v, h] for g, n, v, _, h in rows]

        table = ax_tbl.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            colWidths=[0.18, 0.20, 0.26, 0.26],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.0, 2.0)

        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_facecolor("#404040")

        for i, (_, _, _, colour, _) in enumerate(rows):
            for j in range(len(col_labels)):
                cell = table[i + 1, j]
                cell.set_edgecolor("#cccccc")
                if j == 2:
                    rgba = list(mcolors.to_rgba(colour))
                    rgba[3] = 0.20
                    cell.set_facecolor(rgba)
                    cell.set_text_props(fontweight="bold", fontsize=13)

        # --- explanation text ----------------------------------------------
        ax_note.axis("off")
        explanation = _EXPLANATIONS.get(scenario_name, "")
        ax_note.text(
            0.5, 0.5, explanation, ha="center", va="center",
            fontsize=11, wrap=True,
            transform=ax_note.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="#f0f4ff",
                edgecolor="#8899bb", alpha=0.95,
            ),
        )

        fig.suptitle("Small Object Bias Explorer",
                     fontsize=15, fontweight="bold", y=0.98)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04)
        plt.show()

    # -- wiring -------------------------------------------------------------
    def _on_change(_=None):
        with out:
            clear_output(wait=True)
            _draw(scenario_dd.value)

    scenario_dd.observe(_on_change, names="value")
    _on_change()
    display(widgets.VBox([scenario_dd, out]))


# ------------------------------------------------------------------
# 8. Bland-Altman plot for volume agreement
# ------------------------------------------------------------------
def plot_bland_altman(
    method_a: np.ndarray,
    method_b: np.ndarray,
    *,
    label_a: str = "Method A",
    label_b: str = "Method B",
    units: str = "mm³",
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> None:
    """Bland-Altman plot: mean vs difference for two measurement methods.

    Parameters
    ----------
    method_a, method_b : array-like
        Paired measurements (e.g. volumes) from two methods or raters.
    label_a, label_b : str
        Human-readable names for each method.
    units : str
        Unit label for axes.
    title : str or None
        Optional figure title.
    ax : matplotlib Axes or None
        If provided, draw on this axes; otherwise create a new figure.
    """
    a = np.asarray(method_a, dtype=float)
    b = np.asarray(method_b, dtype=float)
    means = (a + b) / 2
    diffs = a - b
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    upper_loa = mean_diff + 1.96 * sd_diff
    lower_loa = mean_diff - 1.96 * sd_diff

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(means, diffs, s=40, alpha=0.6, edgecolors="black",
               linewidths=0.5, color="steelblue", zorder=3)

    # Mean difference line
    ax.axhline(mean_diff, color="#d62728", linestyle="-", linewidth=1.5,
               label=f"Mean diff = {mean_diff:+.1f} {units}")

    # Limits of agreement
    ax.axhline(upper_loa, color="#ff7f0e", linestyle="--", linewidth=1.2,
               label=f"+1.96 SD = {upper_loa:+.1f}")
    ax.axhline(lower_loa, color="#ff7f0e", linestyle="--", linewidth=1.2,
               label=f"−1.96 SD = {lower_loa:+.1f}")

    # Zero line
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

    # Shade limits of agreement
    xlims = ax.get_xlim()
    ax.fill_between(xlims, lower_loa, upper_loa, alpha=0.08,
                    color="#ff7f0e", zorder=0)
    ax.set_xlim(xlims)

    ax.set_xlabel(f"Mean of {label_a} and {label_b} ({units})", fontsize=11)
    ax.set_ylabel(f"Difference: {label_a} − {label_b} ({units})", fontsize=11)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    elif own_fig:
        ax.set_title("Bland-Altman Plot: Volume Agreement",
                      fontsize=13, fontweight="bold")

    if own_fig:
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------------
# 9. Inter-rater agreement: scatter + Bland-Altman + metric comparison
# ------------------------------------------------------------------
def plot_inter_rater_agreement(
    rater_a: np.ndarray,
    rater_b: np.ndarray,
    metric_name: str = "DSC",
    *,
    label_a: str = "Rater 1",
    label_b: str = "Rater 2",
) -> None:
    """Three-panel overview of inter-rater agreement.

    Panel 1 — Scatter plot (rater A vs rater B) with identity line.
    Panel 2 — Bland-Altman plot (mean vs difference).
    Panel 3 — Distribution of absolute differences (histogram).

    Parameters
    ----------
    rater_a, rater_b : array-like
        Paired metric values (e.g. DSC per case) from two raters or methods.
    metric_name : str
        Name of the metric being compared (for axis labels).
    label_a, label_b : str
        Human-readable names for each rater/method.
    """
    a = np.asarray(rater_a, dtype=float)
    b = np.asarray(rater_b, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ── Panel 1: scatter with identity line ──────────────────
    ax = axes[0]
    lo = min(a.min(), b.min()) - 0.02
    hi = max(a.max(), b.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5,
            label="Perfect agreement")
    ax.scatter(a, b, s=40, alpha=0.6, edgecolors="black", linewidths=0.5,
               color="steelblue", zorder=3)
    ax.set_xlabel(f"{label_a} ({metric_name})", fontsize=11)
    ax.set_ylabel(f"{label_b} ({metric_name})", fontsize=11)
    ax.set_title("Scatter: rater vs rater", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)

    # ── Panel 2: Bland-Altman ────────────────────────────────
    plot_bland_altman(a, b, label_a=label_a, label_b=label_b,
                      units=metric_name, ax=axes[1],
                      title="Bland-Altman")

    # ── Panel 3: histogram of absolute differences ───────────
    ax = axes[2]
    abs_diffs = np.abs(a - b)
    ax.hist(abs_diffs, bins="auto", color="steelblue", edgecolor="black",
            linewidth=0.6, alpha=0.8)
    mean_ad = float(np.mean(abs_diffs))
    ax.axvline(mean_ad, color="#d62728", linewidth=1.5, linestyle="-",
               label=f"Mean |diff| = {mean_ad:.3f}")
    ax.set_xlabel(f"|{label_a} − {label_b}|  ({metric_name})", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of disagreement", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Inter-Rater Agreement: {metric_name}",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 10. Skeleton / centerline demonstration
# ------------------------------------------------------------------

def plot_skeleton_demo():
    """1x3 panels: tube, Y-vessel, and irregular blob with skeleton overlays.

    Each panel shows the binary mask in light blue and the morphological
    skeleton (computed via skimage.morphology.skeletonize) in red.
    This demonstrates what "thinning to a one-pixel-wide backbone" means.
    """
    from skimage.morphology import skeletonize

    size = 160

    # ── Shape 1: simple tube ───────────────────────────────
    tube = np.zeros((size, size), dtype=bool)
    tube[70:90, 20:140] = True  # horizontal tube, 20 px wide

    # ── Shape 2: Y-shaped vessel ───────────────────────────
    vessel = np.zeros((size, size), dtype=bool)
    # main trunk
    for x in range(15, 90):
        vessel[73:83, x] = True
    # upper branch
    for i in range(55):
        y = 73 - i
        x = 90 + i
        if 0 <= y < size and 0 <= x < size:
            vessel[max(0, y - 4):min(size, y + 5), max(0, x - 4):min(size, x + 5)] = True
    # lower branch
    for i in range(55):
        y = 83 + i
        x = 90 + i
        if 0 <= y < size and 0 <= x < size:
            vessel[max(0, y - 4):min(size, y + 5), max(0, x - 4):min(size, x + 5)] = True

    # ── Shape 3: irregular blob ────────────────────────────
    blob = np.zeros((size, size), dtype=bool)
    yy, xx = np.mgrid[:size, :size]
    # main body
    blob |= ((yy - 80) ** 2 + (xx - 60) ** 2) < 35 ** 2
    # elongated arm to the right
    blob |= (((yy - 65) ** 2) / 12 ** 2 + ((xx - 115) ** 2) / 35 ** 2) < 1
    # small bump on top
    blob |= ((yy - 48) ** 2 + (xx - 75) ** 2) < 14 ** 2

    shapes = [
        (tube, "Simple tube"),
        (vessel, "Branching vessel (Y-shape)"),
        (blob, "Irregular blob"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, (mask, title) in zip(axes, shapes):
        skel = skeletonize(mask)

        # Build RGB overlay: mask in light blue, skeleton in red
        rgb = np.ones((size, size, 3))
        rgb[mask] = [0.7, 0.85, 1.0]           # light blue fill
        rgb[skel] = [0.9, 0.1, 0.1]            # red skeleton

        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=[0.7, 0.85, 1.0], label="Binary mask"),
        mpatches.Patch(color=[0.9, 0.1, 0.1], label="Skeleton (centerline)"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=2, fontsize=11, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Skeletonization: thinning a shape to its one-pixel-wide backbone",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 11. Topology examples: Betti numbers
# ------------------------------------------------------------------

def plot_topology_examples():
    """1x4 panels showing shapes with annotated Betti numbers (β₀, β₁).

    Demonstrates:
      - solid disk:       β₀=1, β₁=0 (1 component, 0 holes)
      - ring (annulus):   β₀=1, β₁=1 (1 component, 1 hole)
      - two separate disks: β₀=2, β₁=0 (2 components, 0 holes)
      - ring with 2 holes:  β₀=1, β₁=2 (1 component, 2 holes)
    """
    size = 120
    yy, xx = np.mgrid[:size, :size]

    # ── Shape 1: solid disk ────────────────────────────────
    disk = ((yy - 60) ** 2 + (xx - 60) ** 2) < 40 ** 2

    # ── Shape 2: ring (annulus) ────────────────────────────
    ring = (((yy - 60) ** 2 + (xx - 60) ** 2) < 40 ** 2) & \
           (((yy - 60) ** 2 + (xx - 60) ** 2) > 25 ** 2)

    # ── Shape 3: two separate disks ────────────────────────
    two_disks = (((yy - 60) ** 2 + (xx - 35) ** 2) < 22 ** 2) | \
                (((yy - 60) ** 2 + (xx - 88) ** 2) < 22 ** 2)

    # ── Shape 4: one component, two holes ──────────────────
    base = ((yy - 60) ** 2 + (xx - 60) ** 2) < 48 ** 2
    hole_a = ((yy - 50) ** 2 + (xx - 45) ** 2) < 14 ** 2
    hole_b = ((yy - 70) ** 2 + (xx - 75) ** 2) < 14 ** 2
    two_holes = base & ~hole_a & ~hole_b

    shapes = [
        (disk,      "Solid disk",         r"$\beta_0 = 1,\ \beta_1 = 0$",
         "1 component, 0 holes"),
        (ring,      "Ring (annulus)",      r"$\beta_0 = 1,\ \beta_1 = 1$",
         "1 component, 1 hole"),
        (two_disks, "Two separate disks",  r"$\beta_0 = 2,\ \beta_1 = 0$",
         "2 components, 0 holes"),
        (two_holes, "Disk with 2 holes",   r"$\beta_0 = 1,\ \beta_1 = 2$",
         "1 component, 2 holes"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, (mask, title, betti_str, plain_str) in zip(axes, shapes):
        rgb = np.ones((size, size, 3)) * 0.95
        rgb[mask] = [0.27, 0.51, 0.71]  # steel blue fill

        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.text(
            0.5, -0.06, betti_str, transform=ax.transAxes,
            ha="center", va="top", fontsize=13,
        )
        ax.text(
            0.5, -0.14, plain_str, transform=ax.transAxes,
            ha="center", va="top", fontsize=10, color="gray",
        )
        ax.axis("off")

    fig.suptitle(
        "Betti Numbers: counting connected components and holes",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()
