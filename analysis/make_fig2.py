#!/usr/bin/env python3
"""
Render Figure 2 (empirical distributions of pairwise graph similarity): four
filled KDEs -- between-cluster (gold), within-cluster (red), inter-extractor
(purple), test-retest (blue) -- from the distinct-case within-cluster arrays
(n=1325). Legend kept
(4 entries with means/n) and placed in the mid-right pocket between the two
annotation arrows. Includes a layout self-check (no arrow crosses the legend or
a non-target fill; legend in white space).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

D = Path("data/analysis"); F = Path("figures")
between = np.load(D / "between_sims.npy")
within = np.load(D / "within_sims.npy")
interext = np.load(D / "interext_sims.npy")
retest = np.load(D / "retest_sims.npy")

GOLD, RED, PURPLE, BLUE = "#E0A030", "#C0392B", "#7E5FA5", "#3A78C9"
xs = np.linspace(0.28, 1.02, 700)
KDE = {"between": gaussian_kde(between), "within": gaussian_kde(within),
       "interext": gaussian_kde(interext), "retest": gaussian_kde(retest)}
ymax = max(k(xs).max() for k in KDE.values())

fig, ax = plt.subplots(figsize=(10, 4.6))
for data, color, label in [
    (between, GOLD, f"Between-cluster ({between.mean():.3f}, n={len(between):,})"),
    (within, RED, f"Within-cluster ({within.mean():.3f}, n={len(within):,})"),
    (interext, PURPLE, f"Inter-extractor ({interext.mean():.3f}, n={len(interext)})"),
    (retest, BLUE, f"Test-retest ({retest.mean():.3f}, n={len(retest)})"),
]:
    y = KDE["between" if color == GOLD else "within" if color == RED
            else "interext" if color == PURPLE else "retest"](xs)
    ax.fill_between(xs, y, color=color, alpha=0.25, zorder=1)
    ax.plot(xs, y, color=color, lw=2.0, label=label, zorder=2)

# annotations: text positions (xytext) and arrowhead targets (xy)
RED_T, RED_H = (0.345, 0.92 * ymax), (0.46, 0.20 * ymax)
PUR_T, PUR_H = (0.585, 0.95 * ymax), (0.59, 0.45 * ymax)
BLU_T, BLU_H = (0.85, 0.95 * ymax), (0.86, 0.25 * ymax)
ann = dict(fontsize=10, fontstyle="italic", ha="center", va="center")
ax.annotate("Within ≈ Between\n(no diagnostic-schema clustering)", xy=RED_H, xytext=RED_T,
            color=RED, **ann, arrowprops=dict(arrowstyle="->", color=RED, lw=1.3,
            connectionstyle="arc3,rad=0.4"))
ax.annotate("Same trace,\ndifferent extractor", xy=PUR_H, xytext=PUR_T,
            color=PURPLE, **ann, arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.3))
ax.annotate("Same trace,\nsame extractor", xy=BLU_H, xytext=BLU_T,
            color=BLUE, **ann, arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))

# legend in the mid-right pocket BETWEEN the purple (x≈0.59) and blue (x≈0.85)
# arrows, above the inter-extractor/test-retest humps
leg = ax.legend(loc="center", bbox_to_anchor=(0.605, 0.56), frameon=False, fontsize=9)

ax.set_xlabel("Composite graph similarity", fontsize=11)
ax.set_xlim(0.30, 1.0)
ax.set_ylim(-0.02 * ymax, 1.05 * ymax)
ax.set_xticks(np.arange(0.3, 1.01, 0.1))
ax.yaxis.set_visible(False)
for s in ["top", "left", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.canvas.draw()


# ── layout self-check ────────────────────────────────────────────────────────
def _arc_pts(A, B, rad, n=200):
    # replicate matplotlib arc3 in DISPLAY space (x and y comparable there), then
    # transform back to data coords
    Ad = ax.transData.transform(A); Bd = ax.transData.transform(B)
    Md = (Ad + Bd) / 2; d = Bd - Ad
    Cd = Md + rad * np.array([-d[1], d[0]])
    t = np.linspace(0, 1, n)[:, None]
    disp = (1 - t)**2 * Ad + 2 * (1 - t) * t * Cd + t**2 * Bd
    return ax.transData.inverted().transform(disp)

arrows = {  # name: (path points, set of TARGET distributions)
    "red":    (_arc_pts(RED_T, RED_H, 0.4), {"within", "between"}),
    "purple": (np.linspace(PUR_T, PUR_H, 120), {"interext"}),
    "blue":   (np.linspace(BLU_T, BLU_H, 120), {"retest"}),
}
lb = leg.get_window_extent().transformed(ax.transData.inverted())
LX0, LX1, LY0, LY1 = lb.x0, lb.x1, lb.y0, lb.y1

print(f"legend box (data): x=[{LX0:.3f},{LX1:.3f}] y=[{LY0/ymax:.2f},{LY1/ymax:.2f}]ymax")
ok = True
# (1) no arrow crosses the legend box
for name, (pts, _) in arrows.items():
    hit = np.any((pts[:, 0] >= LX0) & (pts[:, 0] <= LX1) & (pts[:, 1] >= LY0) & (pts[:, 1] <= LY1))
    print(f"(1) {name:6s} arrow ∩ legend box: {'HIT' if hit else 'clear'}")
    ok &= not hit
# (2) no arrow dips under a NON-target filled curve
for name, (pts, targets) in arrows.items():
    bad = []
    for dist in KDE:
        if dist in targets:
            continue
        below = (pts[:, 1] < KDE[dist](pts[:, 0])) & (pts[:, 0] > 0.30) & (pts[:, 0] < 1.0)
        if np.any(below):
            bad.append(dist)
    print(f"(2) {name:6s} arrow under non-target fill: {bad if bad else 'clear'}")
    ok &= not bad
# (3) legend sits in white space (no curve rises into its box over its x-range)
ceil = max(KDE[d](np.linspace(LX0, LX1, 80)).max() for d in KDE)
print(f"(3) legend white space: fill ceiling over legend x = {ceil/ymax:.2f}ymax vs legend bottom {LY0/ymax:.2f}ymax "
      f"-> {'clear' if ceil < LY0 else 'OVERLAP'}")
ok &= ceil < LY0
print(f"\nALL CONSTRAINTS {'PASS' if ok else 'FAIL'}")

fig.savefig(F / "fig2_empirical.png", dpi=300, bbox_inches="tight")
fig.savefig(F / "fig2_empirical.pdf", bbox_inches="tight")
print("Saved figures/fig2_empirical.png/.pdf")
