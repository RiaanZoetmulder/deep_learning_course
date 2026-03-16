"""Interactive step-through: why mAP hides false positives on healthy images.

Builds two synthetic models (A and B) for a lung-nodule screening scenario,
then walks the user through sorted predictions and the accumulating PR
curves side-by-side, showing that mAP is nearly identical despite Model B
producing far more false alarms.

Phases
------
1. **Setup** – show the scenario (100 scans, 50 GT nodules, 70 healthy).
2. **Sorted predictions** – step through detections one-by-one, building
   the PR curve for each model.  Colour-coded rows highlight TP (green),
   FP-positive-scan (orange), and FP-healthy-scan (red).
3. **Final comparison** – overlay both PR curves and print AP + clinical
   FPI numbers.
"""

import io
import threading

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


# ── Synthetic data ─────────────────────────────────────────────────

def _build_scenario():
    """Return (model_a, model_b, n_gt, info) with deterministic data.

    Each model dict has keys: 'scores', 'tp', 'src'
      - scores: confidence (descending)
      - tp:     1 = true positive, 0 = false positive
      - src:    'tp' | 'fp_pos' | 'fp_neg'  (for colour-coding)
    """
    rng = np.random.RandomState(7)

    n_gt = 50  # total ground-truth nodules

    # ── shared detections (identical for both models) ──────────
    # 45 true positives  (90% sensitivity)
    tp_scores = np.sort(rng.beta(8, 2, size=45))[::-1]  # high confidence
    tp_scores = np.clip(tp_scores, 0.40, 0.99)

    # 5 false positives on positive scans (moderate confidence)
    fp_pos_scores = np.sort(rng.beta(3, 4, size=5))[::-1]
    fp_pos_scores = np.clip(fp_pos_scores, 0.15, 0.60)

    # ── Model A: no FPs on healthy scans ───────────────────────
    a_scores = np.concatenate([tp_scores, fp_pos_scores])
    a_tp = np.concatenate([np.ones(45), np.zeros(5)])
    a_src = ['tp'] * 45 + ['fp_pos'] * 5
    order_a = np.argsort(-a_scores)
    model_a = dict(
        scores=a_scores[order_a],
        tp=a_tp[order_a].astype(int),
        src=[a_src[i] for i in order_a],
    )

    # ── Model B: same TPs + FPs, plus 70 FPs on healthy scans ─
    fp_neg_scores = np.sort(rng.beta(2, 8, size=70))[::-1]
    fp_neg_scores = np.clip(fp_neg_scores, 0.02, 0.35)

    b_scores = np.concatenate([tp_scores, fp_pos_scores, fp_neg_scores])
    b_tp = np.concatenate([np.ones(45), np.zeros(5), np.zeros(70)])
    b_src = ['tp'] * 45 + ['fp_pos'] * 5 + ['fp_neg'] * 70
    order_b = np.argsort(-b_scores)
    model_b = dict(
        scores=b_scores[order_b],
        tp=b_tp[order_b].astype(int),
        src=[b_src[i] for i in order_b],
    )

    info = dict(
        n_gt=n_gt,
        n_images=100,
        n_healthy=70,
        n_positive=30,
        n_tp=45,
        n_fp_pos=5,
        n_fp_neg_b=70,
    )
    return model_a, model_b, n_gt, info


def _compute_pr(tp_array, n_gt):
    """Cumulative precision/recall from a binary TP array."""
    cum_tp = np.cumsum(tp_array)
    cum_fp = np.arange(1, len(tp_array) + 1) - cum_tp
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / n_gt
    return precision, recall


def _compute_ap(precision, recall):
    """All-point interpolated AP (COCO style)."""
    mrec = np.concatenate(([0.0], recall, [recall[-1] + 1e-6]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx])
    return float(ap)


# ── Rendering ──────────────────────────────────────────────────────

_SRC_COLORS = {
    'tp': '#388E3C',       # green
    'fp_pos': '#F57C00',   # orange
    'fp_neg': '#D32F2F',   # red
}
_SRC_LABELS = {
    'tp': 'TP (nodule detected)',
    'fp_pos': 'FP on positive scan',
    'fp_neg': 'FP on healthy scan',
}

_CSS = ("font-family:monospace;font-size:13px;padding:10px;"
        "background:#f5f5f5;border-radius:6px;margin-top:4px;"
        "line-height:1.55")


def _fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Main widget ────────────────────────────────────────────────────

def create_map_failure_demo():
    """Interactive step-through comparing Model A and Model B PR curves."""

    model_a, model_b, n_gt, info = _build_scenario()

    prec_a_full, rec_a_full = _compute_pr(model_a['tp'], n_gt)
    prec_b_full, rec_b_full = _compute_pr(model_b['tp'], n_gt)
    ap_a = _compute_ap(prec_a_full, rec_a_full)
    ap_b = _compute_ap(prec_b_full, rec_b_full)

    # Phase sizes
    N_A = len(model_a['scores'])   # 50 steps for Model A
    N_B = len(model_b['scores'])   # 120 steps for Model B

    # --  We walk through predictions in three phases:
    #   Phase 1 (1 step):  scenario overview
    #   Phase 2 (N_B steps: step through Model B predictions, show Model A alongside)
    #   Phase 3 (1 step):  final comparison with AP numbers + clinical insight
    S1 = 1
    S2 = N_B
    S3 = 1
    TOTAL = S1 + S2 + S3

    img_widget = widgets.Image(format="png")
    info_html = widgets.HTML()

    def _draw_setup():
        """Phase 1: scenario overview figure."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: bar chart of dataset composition
        ax = axes[0]
        bars = ax.barh(
            ['Healthy scans\n(no nodules)', 'Positive scans\n(have nodules)'],
            [info['n_healthy'], info['n_positive']],
            color=['#90CAF9', '#EF9A9A'], edgecolor='#333', height=0.5,
        )
        ax.set_xlabel('Number of scans', fontsize=11)
        ax.set_title('Dataset: 100 Lung CT Scans', fontweight='bold', fontsize=12)
        ax.set_xlim(0, 85)
        for b, v in zip(bars, [info['n_healthy'], info['n_positive']]):
            ax.text(v + 1, b.get_y() + b.get_height() / 2,
                    str(v), va='center', fontweight='bold', fontsize=13)

        # Right: model comparison table as text
        ax = axes[1]
        ax.axis('off')
        rows = [
            ['', 'Model A', 'Model B'],
            ['TP (detected nodules)', '45 / 50', '45 / 50'],
            ['FP on positive scans', '5', '5'],
            ['FP on healthy scans', '0', '70  ← the difference!'],
            ['Total predictions', '50', '120'],
        ]
        table = ax.table(cellText=rows, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 1.8)
        # Header row
        for j in range(3):
            table[0, j].set_facecolor('#E0E0E0')
            table[0, j].set_text_props(fontweight='bold')
        # Highlight the FP-healthy row
        for j in range(3):
            table[3, j].set_facecolor('#FFEBEE')
        ax.set_title('Model Comparison', fontweight='bold', fontsize=12)
        fig.tight_layout()
        return fig

    def _draw_step(step_idx):
        """Phase 2: show PR curves building up to step_idx predictions."""
        k = step_idx  # number of Model B predictions processed so far

        # Model B: partial PR
        tp_b_partial = model_b['tp'][:k]
        prec_b, rec_b = _compute_pr(tp_b_partial, n_gt)

        # Model A: show all predictions up to the same number of
        # TPs + FP_pos as Model B has seen so far (i.e. excluding
        # FP_neg predictions which Model A doesn't have).
        # Simpler: show Model A fully built — it only has 50 preds,
        # and we reveal it progressively too.
        k_a = min(k, N_A)
        tp_a_partial = model_a['tp'][:k_a]
        prec_a, rec_a = _compute_pr(tp_a_partial, n_gt) if k_a > 0 else ([], [])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

        # ── Left: Model A PR curve ──
        ax = axes[0]
        if len(rec_a) > 0:
            ax.step(np.concatenate(([0], rec_a)),
                    np.concatenate(([1], prec_a)),
                    where='post', color='#1976D2', lw=2.5, zorder=3)
            ax.scatter(rec_a[-1:], prec_a[-1:], color='#1976D2',
                       s=80, zorder=5, edgecolors='k')
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ap_a_so_far = _compute_ap(prec_a, rec_a) if len(rec_a) > 1 else 0
        ax.set_title(f'Model A  —  {k_a}/{N_A} preds  |  AP = {ap_a_so_far:.3f}',
                     fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.25)
        # annotation: no FPs on healthy
        ax.text(0.50, 0.10, '0 FPs on healthy scans',
                transform=ax.transAxes, fontsize=10, color='#388E3C',
                ha='center', fontstyle='italic')

        # ── Right: Model B PR curve ──
        ax = axes[1]
        if len(rec_b) > 0:
            # Colour the curve segments by src type
            rec_ext = np.concatenate(([0], rec_b))
            pre_ext = np.concatenate(([1], prec_b))
            ax.step(rec_ext, pre_ext, where='post',
                    color='#1976D2', lw=1.5, alpha=0.4, zorder=2)
            # latest point
            src_now = model_b['src'][k - 1]
            color_now = _SRC_COLORS[src_now]
            ax.scatter(rec_b[-1:], prec_b[-1:], color=color_now,
                       s=80, zorder=5, edgecolors='k')
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel('Recall', fontsize=11)
        ap_b_so_far = _compute_ap(prec_b, rec_b) if len(rec_b) > 1 else 0
        n_fp_neg_so_far = sum(1 for s in model_b['src'][:k] if s == 'fp_neg')
        ax.set_title(f'Model B  —  {k}/{N_B} preds  |  AP = {ap_b_so_far:.3f}',
                     fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.text(0.50, 0.10,
                f'{n_fp_neg_so_far} FPs on healthy scans (invisible to mAP)',
                transform=ax.transAxes, fontsize=10, color='#D32F2F',
                ha='center', fontstyle='italic')

        fig.suptitle(
            f'Prediction {k} / {N_B}   |   conf = {model_b["scores"][k-1]:.3f}   |   '
            f'type: {_SRC_LABELS[model_b["src"][k-1]]}',
            fontsize=11, y=1.01, color=_SRC_COLORS[model_b['src'][k - 1]],
            fontweight='bold',
        )
        fig.tight_layout()
        return fig

    def _draw_final():
        """Phase 3: final overlay + clinical comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # ── Left: overlaid PR curves ──
        ax = axes[0]
        ax.step(np.concatenate(([0], rec_a_full)),
                np.concatenate(([1], prec_a_full)),
                where='post', color='#1976D2', lw=2.5, label=f'Model A  (AP={ap_a:.3f})')
        ax.step(np.concatenate(([0], rec_b_full)),
                np.concatenate(([1], prec_b_full)),
                where='post', color='#D32F2F', lw=2.5, ls='--',
                label=f'Model B  (AP={ap_b:.3f})')
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title('PR Curves Overlay', fontweight='bold', fontsize=12)
        ax.legend(fontsize=10, loc='lower left')
        ax.grid(True, alpha=0.25)

        # Highlight the difference region
        # Model B's curve dips lower at the tail where FPs on healthy accumulate
        ax.annotate(
            'FPs on healthy scans\ndrag precision down here,\n'
            'but recall is already\nsaturated → AP barely changes',
            xy=(0.90, float(prec_b_full[-1])),
            xytext=(0.55, 0.25),
            fontsize=9, color='#D32F2F',
            arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFEBEE', ec='#D32F2F', alpha=0.9),
        )

        # ── Right: clinical comparison ──
        ax = axes[1]
        ax.axis('off')

        fpi_a = (5 + 0) / info['n_images']
        fpi_b = (5 + 70) / info['n_images']

        text = (
            f"                        Model A      Model B\n"
            f"  ─────────────────────────────────────────────\n"
            f"  AP (mAP)              {ap_a:.3f}        {ap_b:.3f}\n"
            f"  Sensitivity           90.0%        90.0%\n"
            f"  FP on healthy scans   0            70\n"
            f"  FP per image (FPI)    {fpi_a:.2f}         {fpi_b:.2f}\n"
            f"  ─────────────────────────────────────────────\n\n"
            f"  AP difference: {abs(ap_a - ap_b):.3f}  ← nearly invisible!\n"
            f"  FPI difference: {fpi_b/max(fpi_a, 0.001):.0f}×  ← massive clinical impact!"
        )
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=12, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.6', fc='#FFF8E1', ec='#F57C00', alpha=0.95))
        ax.set_title('Clinical Reality vs. mAP', fontweight='bold', fontsize=12)

        fig.tight_layout()
        return fig

    # ── Build scoreboard HTML for each phase ───────────────────

    def _info_setup():
        return (
            f'<div style="{_CSS}">'
            f'<b>Scenario:</b> 100 lung CT scans &mdash; '
            f'{info["n_healthy"]} healthy, {info["n_positive"]} with nodules '
            f'({info["n_gt"]} total GT nodules)<br>'
            f'<b>Model A:</b> 45 TP + 5 FP on positive scans = 50 predictions<br>'
            f'<b>Model B:</b> same 45 TP + 5 FP, <b>plus 70 FP on healthy scans</b> '
            f'= 120 predictions<br><br>'
            f'<span style="color:#388E3C">&#9632;</span> TP &nbsp;&nbsp;'
            f'<span style="color:#F57C00">&#9632;</span> FP (positive scan) &nbsp;&nbsp;'
            f'<span style="color:#D32F2F">&#9632;</span> FP (healthy scan) &nbsp;&nbsp;'
            f'<br>Press <b>Next &#9654;</b> or <b>&#9654; Play</b> to step through predictions.</div>'
        )

    def _info_step(k):
        src = model_b['src'][k - 1]
        conf = model_b['scores'][k - 1]
        color = _SRC_COLORS[src]
        label = _SRC_LABELS[src]

        # Cumulative counts for Model B
        cum_tp = int(np.sum(model_b['tp'][:k]))
        cum_fp_pos = sum(1 for s in model_b['src'][:k] if s == 'fp_pos')
        cum_fp_neg = sum(1 for s in model_b['src'][:k] if s == 'fp_neg')
        prec_b_now = cum_tp / k
        rec_b_now = cum_tp / n_gt

        note = ''
        if src == 'fp_neg':
            note = (
                '<br><span style="color:#D32F2F">'
                '&#9888; This FP is on a healthy scan &mdash; recall does NOT increase, '
                'and precision drops only slightly because it\'s diluted among '
                f'{k} total predictions. <b>mAP barely notices!</b></span>'
            )

        return (
            f'<div style="{_CSS}">'
            f'<b>Prediction {k}/{N_B}</b> &nbsp;|&nbsp; '
            f'confidence = {conf:.3f} &nbsp;|&nbsp; '
            f'<span style="color:{color}"><b>{label}</b></span>{note}<br>'
            f'<b>Model B running totals:</b> '
            f'TP={cum_tp}, FP<sub>pos</sub>={cum_fp_pos}, '
            f'FP<sub>healthy</sub>={cum_fp_neg} &nbsp;|&nbsp; '
            f'Prec={prec_b_now:.3f}, Recall={rec_b_now:.3f}</div>'
        )

    def _info_final():
        return (
            f'<div style="{_CSS}">'
            f'<b>Result:</b> AP is nearly identical '
            f'({ap_a:.3f} vs {ap_b:.3f}, &Delta;={abs(ap_a - ap_b):.3f}), '
            f'yet Model B produces <b>70 extra false alarms</b> '
            f'on healthy scans.<br>'
            f'<b>Clinical impact:</b> a 10&times; difference in false positives per image '
            f'that mAP completely ignores.<br>'
            f'&#8594; This is why lesion screening uses '
            f'<b>FROC (False Positives per Image)</b> instead of mAP.</div>'
        )

    # ── Render dispatcher ──────────────────────────────────────

    def render(step):
        old_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        try:
            if step < S1:
                fig = _draw_setup()
                info_html.value = _info_setup()
            elif step < S1 + S2:
                k = step - S1 + 1  # 1-based prediction index
                fig = _draw_step(k)
                info_html.value = _info_step(k)
            else:
                fig = _draw_final()
                info_html.value = _info_final()
            img_widget.value = _fig_to_png(fig)
        finally:
            matplotlib.use(old_backend)

    # ── Controls (same pattern as ap_demo) ─────────────────────

    slider = widgets.IntSlider(
        min=0, max=TOTAL - 1, value=0,
        description="Step:", continuous_update=False,
        layout=widgets.Layout(width="50%"),
    )
    bprev = widgets.Button(description="\u25C0 Prev",
                           layout=widgets.Layout(width="80px"))
    bnext = widgets.Button(description="Next \u25B6",
                           layout=widgets.Layout(width="80px"))
    breset = widgets.Button(description="Reset",
                            layout=widgets.Layout(width="80px"))
    bplay = widgets.Button(description="\u25B6 Play",
                           layout=widgets.Layout(width="80px"),
                           button_style="success")
    speed = widgets.Dropdown(
        options=[("0.1 s", 0.1), ("0.25 s", 0.25), ("0.5 s", 0.5),
                 ("1 s", 1.0)],
        value=0.25, description="Delay:",
        layout=widgets.Layout(width="140px"),
    )

    _play_event = threading.Event()

    def _play_loop():
        while not _play_event.is_set():
            if slider.value >= TOTAL - 1:
                break
            slider.value = slider.value + 1
            if _play_event.wait(speed.value):
                break
        bplay.description = "\u25B6 Play"
        bplay.button_style = "success"

    def _toggle_play(_):
        if bplay.description.endswith("Play"):
            bplay.description = "\u23F8 Pause"
            bplay.button_style = "warning"
            _play_event.clear()
            t = threading.Thread(target=_play_loop, daemon=True)
            t.start()
        else:
            _play_event.set()
            bplay.description = "\u25B6 Play"
            bplay.button_style = "success"

    slider.observe(lambda c: render(c["new"]), names="value")
    bprev.on_click(lambda _: setattr(slider, "value",
                                      max(0, slider.value - 1)))
    bnext.on_click(lambda _: setattr(slider, "value",
                                      min(TOTAL - 1, slider.value + 1)))
    breset.on_click(lambda _: setattr(slider, "value", 0))
    bplay.on_click(_toggle_play)

    display(widgets.VBox([
        widgets.HBox([bprev, bnext, breset, bplay, speed, slider]),
        img_widget,
        info_html,
    ]))
    render(0)
