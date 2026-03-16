"""FROC curve and related visualizations for medical detection.

Functions
---------
plot_froc_curve           FROC curve with LUNA16 operating points.
plot_froc_score_bars      Sensitivity at each FP level, with FROC score line.
plot_patient_vs_lesion    Patient-level ROC vs lesion-level FROC side-by-side.
plot_bootstrap_histogram  Bootstrap FROC score distribution with 95% CI.
"""

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# 1. FROC curve
# ------------------------------------------------------------------

def plot_froc_curve(fpi_vals, sens_vals, luna_fp_levels=None):
    """FROC curve with LUNA16 operating-point annotations.

    Parameters
    ----------
    fpi_vals  : 1D array, false positives per image.
    sens_vals : 1D array, sensitivity values.
    luna_fp_levels : list of FP levels to annotate (default LUNA16).
    """
    if luna_fp_levels is None:
        luna_fp_levels = [0.125, 0.25, 0.5, 1, 2, 4, 8]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(fpi_vals, sens_vals, "b-", linewidth=2, label="FROC curve")

    colors_op = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(luna_fp_levels)))
    for fp_level, color in zip(luna_fp_levels, colors_op):
        if fp_level <= fpi_vals[-1]:
            idx = np.searchsorted(fpi_vals, fp_level)
            sens_at_fp = sens_vals[min(idx, len(sens_vals) - 1)]
        else:
            sens_at_fp = sens_vals[-1]

        ax.plot(fp_level, sens_at_fp, "o", color=color, markersize=10, zorder=5)
        ax.annotate(
            f"FPI={fp_level}\nSens={sens_at_fp:.2f}",
            xy=(fp_level, sens_at_fp),
            xytext=(15, -10),
            textcoords="offset points",
            fontsize=8,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    ax.set_xlabel("False Positives per Image (FPI)", fontsize=13)
    ax.set_ylabel("Sensitivity (Recall)", fontsize=13)
    ax.set_title("FROC Curve: Lung Nodule Detection", fontsize=15)
    ax.set_xlim(-0.1, max(10, fpi_vals[-1] + 0.5))
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect sensitivity")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 2. FROC score bars
# ------------------------------------------------------------------

def plot_froc_score_bars(froc_sensitivities, froc_sc):
    """Bar chart of sensitivity at each LUNA16 FP level.

    Parameters
    ----------
    froc_sensitivities : dict mapping FP-level to sensitivity.
    froc_sc            : float, overall FROC score.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    fp_levels_list = list(froc_sensitivities.keys())
    sens_list = list(froc_sensitivities.values())

    bars = ax.bar(
        [str(fp) for fp in fp_levels_list],
        sens_list,
        color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(fp_levels_list))),
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, val in zip(bars, sens_list):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.axhline(
        y=froc_sc, color="navy", linestyle="--", linewidth=2,
        label=f"FROC Score = {froc_sc:.4f}",
    )
    ax.set_xlabel("False Positives per Image", fontsize=13)
    ax.set_ylabel("Sensitivity", fontsize=13)
    ax.set_title("Sensitivity at LUNA16 FP Levels", fontsize=15)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 3. Patient-level ROC vs lesion-level FROC
# ------------------------------------------------------------------

def plot_patient_vs_lesion(
    fpr_patient,
    tpr_patient,
    auc_patient,
    fpi_vals,
    sens_vals,
    froc_sc,
):
    """Side-by-side: patient-level ROC (left) and lesion-level FROC (right).

    Parameters
    ----------
    fpr_patient, tpr_patient : 1D arrays, patient-level ROC curve.
    auc_patient : float.
    fpi_vals, sens_vals : 1D arrays, FROC curve.
    froc_sc : float, FROC score.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Patient ROC
    ax1 = axes[0]
    ax1.plot(
        fpr_patient, tpr_patient, "b-", linewidth=2.5,
        label=f"Patient-Level ROC (AUC = {auc_patient:.3f})",
    )
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random classifier")
    ax1.fill_between(fpr_patient, tpr_patient, alpha=0.15, color="blue")

    idx_90 = np.argmin(np.abs(tpr_patient - 0.90))
    ax1.plot(fpr_patient[idx_90], tpr_patient[idx_90], "ro", markersize=12, zorder=5)
    ax1.annotate(
        f"  90% Sens\n  FPR={fpr_patient[idx_90]:.2f}",
        xy=(fpr_patient[idx_90], tpr_patient[idx_90]),
        fontsize=10,
        color="red",
        fontweight="bold",
    )

    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=13)
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=13)
    ax1.set_title("Patient-Level ROC Curve", fontsize=14)
    ax1.legend(fontsize=11, loc="lower right")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Right: Lesion FROC
    ax2 = axes[1]
    ax2.plot(
        fpi_vals, sens_vals, "r-", linewidth=2.5,
        label=f"Lesion-Level FROC (Score = {froc_sc:.3f})",
    )
    for fp_level in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
        if fp_level <= fpi_vals[-1]:
            idx = np.searchsorted(fpi_vals, fp_level)
            s = sens_vals[min(idx, len(sens_vals) - 1)]
        else:
            s = sens_vals[-1]
        ax2.plot(fp_level, s, "o", color="darkred", markersize=7, zorder=5)

    ax2.set_xlabel("False Positives per Image (FPI)", fontsize=13)
    ax2.set_ylabel("Sensitivity (Recall)", fontsize=13)
    ax2.set_title("Lesion-Level FROC Curve", fontsize=14)
    ax2.legend(fontsize=11, loc="lower right")
    ax2.set_xlim(-0.1, max(10, fpi_vals[-1] + 0.5))
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Patient-Level vs Lesion-Level Evaluation", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 4. Bootstrap histogram
# ------------------------------------------------------------------

def plot_bootstrap_histogram(boot_scores, boot_ci_lo, boot_ci_hi, boot_mean, froc_sc):
    """Histogram of bootstrap FROC scores with 95% CI.

    Parameters
    ----------
    boot_scores : 1D array of bootstrap FROC scores.
    boot_ci_lo, boot_ci_hi : float, 95% CI bounds.
    boot_mean   : float, bootstrap mean.
    froc_sc     : float, point estimate.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        boot_scores, bins=40, color="steelblue", edgecolor="white",
        alpha=0.85, density=True, label="Bootstrap distribution",
    )
    ax.axvline(
        boot_ci_lo, color="red", linestyle="--", linewidth=2,
        label=f"2.5th percentile = {boot_ci_lo:.4f}",
    )
    ax.axvline(
        boot_ci_hi, color="red", linestyle="--", linewidth=2,
        label=f"97.5th percentile = {boot_ci_hi:.4f}",
    )
    ax.axvline(
        froc_sc, color="darkgreen", linestyle="-", linewidth=2.5,
        label=f"Point estimate = {froc_sc:.4f}",
    )
    ax.axvline(
        boot_mean, color="orange", linestyle="-", linewidth=2,
        label=f"Bootstrap mean = {boot_mean:.4f}",
    )
    ax.axvspan(boot_ci_lo, boot_ci_hi, alpha=0.15, color="red", label="95% CI")

    ax.set_xlabel("FROC Score", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Bootstrap Distribution of FROC Score (1000 iterations)", fontsize=15)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
