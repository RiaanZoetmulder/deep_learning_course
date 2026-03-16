"""Interactive step-through of Average Precision (AP) computation.

Phases:
  1. Show the raw (zig-zag) Precision-Recall curve.
  2. Smooth the curve right-to-left (running-max envelope).
  3. Sum the rectangles under the smoothed staircase (Riemann sum = AP).
"""

import io
import threading

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ipywidgets as widgets
from IPython.display import display


# ── data helpers ───────────────────────────────────────────────────


def _build_demo_data():
    """Return (precisions, recalls, n_gt, tp_fp) with visible zig-zag."""
    n_gt = 5
    # Semi-realistic: high-confidence preds are mostly TP, lower ones mixed
    tp_fp = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0])
    cum_tp = np.cumsum(tp_fp)
    precisions = cum_tp / np.arange(1, len(tp_fp) + 1)
    recalls = cum_tp / n_gt
    return precisions, recalls, n_gt, tp_fp


def _pad(precisions, recalls):
    """Standard AP padding: prepend (recall=0, prec=1), append sentinel."""
    mrec = np.concatenate(([0.0], recalls, [recalls[-1] + 1e-6]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    return mrec, mpre


def _smooth_snapshots(mpre_raw):
    """Right-to-left max sweep; return one snapshot per index processed."""
    n = len(mpre_raw)
    work = mpre_raw.copy()
    snaps = []
    for i in range(n - 2, -1, -1):
        old = work[i]
        work[i] = max(work[i], work[i + 1])
        snaps.append(dict(
            idx=i, old=old, new=work[i],
            changed=abs(work[i] - old) > 1e-9,
            mpre=work.copy(),
        ))
    return snaps, work.copy()


def _riemann_rects(mrec, mpre_smooth):
    """Rectangles at every recall change-point (skip near-zero widths)."""
    cps = np.where(mrec[1:] != mrec[:-1])[0] + 1
    rects, cum = [], 0.0
    for cp in cps:
        w = mrec[cp] - mrec[cp - 1]
        h = mpre_smooth[cp]
        if w < 1e-4:
            continue
        cum += w * h
        rects.append(dict(x=mrec[cp - 1], w=w, h=h, area=w * h, cum=cum))
    return rects


# ── main widget ────────────────────────────────────────────────────


def ap_step_demo():
    """Walk step-by-step through AP: raw curve -> smoothing -> Riemann sum."""

    prec, rec, n_gt, tp_fp = _build_demo_data()
    mrec, mpre_raw = _pad(prec, rec)
    snaps, mpre_sm = _smooth_snapshots(mpre_raw)
    rects = _riemann_rects(mrec, mpre_sm)
    total_ap = rects[-1]["cum"] if rects else 0.0

    # step counts per phase
    S_A = 1               # raw curve
    S_B = len(snaps)      # smoothing sweep
    S_C = 1               # overlay (smoothing done)
    S_D = len(rects)      # Riemann rectangles
    N = S_A + S_B + S_C + S_D

    # ── drawing helpers ────────────────────────────────────────

    def _axes(ax, title):
        ax.set_xlim(-0.05, 1.12)
        ax.set_ylim(-0.05, 1.12)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.25)

    def _raw(ax, alpha=1.0, lbl="Raw PR curve"):
        ax.step(mrec, mpre_raw, where="post", color="#1976D2",
                lw=2, alpha=alpha, label=lbl, zorder=2)
        ax.scatter(mrec[1:-1], mpre_raw[1:-1], c="#1976D2",
                   s=25, zorder=3, alpha=alpha)

    def _smooth_curve(ax, mpre, alpha=1.0, lbl="Smoothed"):
        ax.step(mrec, mpre, where="post", color="#388E3C",
                lw=2.5, alpha=alpha, label=lbl, zorder=4)

    # ── scoreboard builders ────────────────────────────────────

    _css = ("font-family:monospace;font-size:13px;padding:8px;"
            "background:#f5f5f5;border-radius:4px;margin-top:4px")

    def _sb_raw():
        return (
            f'<div style="{_css}">'
            f"<b>Phase 1 / 3 &mdash; Raw PR Curve</b><br>"
            f"{len(tp_fp)} predictions, {n_gt} GT objects<br>"
            f"TP/FP: {tp_fp.tolist()}<br>"
            f"The zig-zag makes it hard to compare models "
            f"&rarr; we smooth it next.</div>"
        )

    def _sb_smooth(sn):
        pi = sn["idx"]
        right_val = sn["mpre"][min(pi + 1, len(sn["mpre"]) - 1)]
        if sn["changed"]:
            delta = (
                f'<span style="color:#D32F2F"><b>Lifted</b></span> '
                f'{sn["old"]:.3f} &rarr; {sn["new"]:.3f}'
            )
        else:
            delta = (
                '<span style="color:#388E3C">No change</span> '
                "(already &ge; running max from the right)"
            )
        return (
            f'<div style="{_css}">'
            f"<b>Phase 2 / 3 &mdash; Smoothing</b> "
            f"(scanning right &rarr; left)<br>"
            f"Index {pi}: precision = {sn['old']:.3f}, "
            f"running max from right = {right_val:.3f}<br>"
            f"{delta}</div>"
        )

    def _sb_overlay(nc):
        return (
            f'<div style="{_css}">'
            f"<b>Phase 2 complete</b><br>"
            f"{nc} point{'s' if nc != 1 else ''} lifted. "
            f"The curve is now a non-increasing staircase.<br>"
            f"Next: measure the area under this staircase.</div>"
        )

    def _sb_rect(ri):
        r = rects[ri]
        fin = ""
        if ri == len(rects) - 1:
            fin = (
                f'<br><br><span style="color:#1565C0;font-size:15px">'
                f"<b>Final AP = {total_ap:.4f}</b></span>"
            )
        return (
            f'<div style="{_css}">'
            f"<b>Phase 3 / 3 &mdash; Riemann Sum</b> "
            f"(rectangle {ri + 1}/{len(rects)})<br>"
            f"&Delta;recall = {r['w']:.2f}, "
            f"precision = {r['h']:.4f}<br>"
            f"Area = {r['w']:.2f} &times; {r['h']:.4f} "
            f"= {r['area']:.4f}<br>"
            f"<b>Cumulative AP = {r['cum']:.4f}</b>{fin}</div>"
        )

    # ── render one step ────────────────────────────────────────

    img_widget = widgets.Image(format="png", width="800px")
    info = widgets.HTML()

    def _fig_to_png(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def render(step):
        old_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        try:
            fig, ax = plt.subplots(figsize=(8, 5))

            # Phase A: raw PR curve
            if step < S_A:
                _raw(ax)
                _axes(ax, "Phase 1 / 3: Raw PR Curve")
                ax.legend(loc="upper right")
                info.value = _sb_raw()

            # Phase B: smoothing sweep (right to left)
            elif step < S_A + S_B:
                si = step - S_A
                sn = snaps[si]
                _raw(ax, alpha=0.3, lbl="Raw")
                _smooth_curve(ax, sn["mpre"], alpha=0.9)
                # highlight current point
                pi = sn["idx"]
                ax.scatter([mrec[pi]], [sn["new"]], color="#FF8F00",
                           s=200, marker="*", zorder=10,
                           edgecolors="k", linewidths=1)
                # show lift arrow when value changed
                if sn["changed"]:
                    ax.annotate(
                        "", xy=(float(mrec[pi]), sn["new"]),
                        xytext=(float(mrec[pi]), sn["old"]),
                        arrowprops=dict(arrowstyle="->", color="#D32F2F",
                                        lw=2, mutation_scale=15),
                        zorder=11,
                    )
                    ax.scatter([mrec[pi]], [sn["old"]], c="#D32F2F",
                               s=50, marker="x", linewidths=2, zorder=9)
                tag = "LIFTED" if sn["changed"] else "no change"
                _axes(ax, f"Phase 2 / 3: Smoothing  (index {pi}  {tag})")
                ax.legend(loc="upper right", fontsize=9)
                info.value = _sb_smooth(sn)

            # Phase C: overlay (smoothing done)
            elif step < S_A + S_B + S_C:
                _raw(ax, alpha=0.35, lbl="Raw")
                _smooth_curve(ax, mpre_sm, lbl="Smoothed")
                for sn in snaps:
                    if sn["changed"]:
                        pi = sn["idx"]
                        ax.scatter([mrec[pi]], [sn["old"]], c="#D32F2F",
                                   s=30, marker="x", linewidths=1.5, zorder=5)
                        ax.annotate(
                            "", xy=(float(mrec[pi]), sn["new"]),
                            xytext=(float(mrec[pi]), sn["old"]),
                            arrowprops=dict(arrowstyle="->", color="#D32F2F",
                                            lw=1.5), zorder=6,
                        )
                nc = sum(s["changed"] for s in snaps)
                _axes(ax, f"Smoothing Complete  ({nc} points lifted)")
                ax.legend(loc="upper right")
                info.value = _sb_overlay(nc)

            # Phase D: Riemann rectangles
            else:
                ri = step - S_A - S_B - S_C
                _raw(ax, alpha=0.15, lbl="Raw")
                _smooth_curve(ax, mpre_sm, alpha=0.7, lbl="Smoothed")
                for j in range(ri + 1):
                    r = rects[j]
                    cur = j == ri
                    fc = "#FFB74D" if cur else "#90CAF9"
                    ec = "#E65100" if cur else "#1565C0"
                    al = 0.55 if cur else 0.30
                    rect = Rectangle(
                        (r["x"], 0), r["w"], r["h"],
                        fc=fc, ec=ec, lw=1.5, alpha=al, zorder=1,
                    )
                    ax.add_patch(rect)
                    ax.text(r["x"] + r["w"] / 2, r["h"] / 2,
                            f"{r['area']:.4f}",
                            ha="center", va="center",
                            fontsize=9, fontweight="bold", zorder=7)
                _axes(ax, f"Phase 3 / 3: Riemann Sum  "
                          f"(rectangle {ri + 1}/{len(rects)})")
                ax.legend(loc="upper right", fontsize=9)
                info.value = _sb_rect(ri)

            fig.tight_layout()
            img_widget.value = _fig_to_png(fig)
        finally:
            matplotlib.use(old_backend)

    # ── controls ───────────────────────────────────────────────

    slider = widgets.IntSlider(
        min=0, max=N - 1, value=0,
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
        options=[("0.25 s", 0.25), ("0.5 s", 0.5), ("1 s", 1.0), ("2 s", 2.0)],
        value=0.5, description="Delay:",
        layout=widgets.Layout(width="140px"),
    )

    _play_event = threading.Event()

    def _play_loop():
        while not _play_event.is_set():
            if slider.value >= N - 1:
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
                                      min(N - 1, slider.value + 1)))
    breset.on_click(lambda _: setattr(slider, "value", 0))
    bplay.on_click(_toggle_play)

    display(widgets.VBox([
        widgets.HBox([bprev, bnext, breset, bplay, speed, slider]),
        img_widget,
        info,
    ]))
    render(0)
