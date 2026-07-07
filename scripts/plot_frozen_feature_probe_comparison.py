#!/usr/bin/env python
"""Publication-quality static figures comparing the frozen-feature linear-probe
"pLDDT floor" of two protein models: Complexa vs La-Proteina.

A frozen-feature linear probe fits a single nn.Linear on a model's FROZEN trunk
features to predict AF2 pLDDT. The held-out val Pearson r is the "linear pLDDT
floor": how near-linearly-predictive the frozen features are.

  - Complexa (complexa.ckpt trunk): frozen features are STRONGLY linearly
    pLDDT-predictive -> floor ~0.74. Measured on Teddymer DIMERS.
  - La-Proteina (LD1_ucond_notri_512 trunk): frozen features are NOT linearly
    pLDDT-predictive -> floor ~0. Measured on SwissProt MONOMERS (AF2 B-factor
    pLDDT).

Plotting-only. All numbers are hardcoded below; nothing is trained or loaded.
Emits three PNGs (dpi=200) into scripts/probe_figs/.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- #
# DATA (hardcoded, verbatim from the provided tables)
# --------------------------------------------------------------------------- #

# Final SGD-probe Pearson r, per variant, per model. val_eval is the floor.
# variant -> (train_fit, train_eval, val_eval)
LAPROTEINA_FINAL = {
    "s_only":    (0.0841, 0.0047, -0.0149),
    "s_latents": (0.0903, 0.0074, -0.0147),
    "z_pooled":  (0.0093, 0.0052,  0.0089),
    "s_z":       (0.1085, 0.0101, -0.0065),
}

COMPLEXA_FINAL = {
    "s_only":   (0.7418, 0.7112, 0.7311),
    "z_pooled": (0.4524, 0.4282, 0.4818),
    "s_z":      (0.7560, 0.7242, 0.7402),
    # Complexa has no s_latents variant.
}

# Ridge closed-form cross-check (fp64 self-fit Pearson) for La-Proteina, where
# logged. z_pooled / s_latents ridge not logged -> omitted.
LAPROTEINA_RIDGE = {
    "s_only": (0.0423, 0.0512, 0.0566),
    "s_z":    (0.0670, 0.0738, 0.0876),
}

# La-Proteina val_eval Pearson-vs-SGD-step learning curves, step 0..3000 by 100.
_LAP_STEPS = list(range(0, 3001, 100))
LAPROTEINA_CURVES = {
    "s_only": [
        -0.001, -0.002, -0.010, -0.013, -0.015, -0.016, -0.016, -0.016, -0.017,
        -0.017, -0.017, -0.017, -0.017, -0.018, -0.018, -0.018, -0.018, -0.017,
        -0.017, -0.017, -0.017, -0.017, -0.017, -0.017, -0.016, -0.016, -0.016,
        -0.016, -0.015, -0.015, -0.015,
    ],
    "s_latents": [
        -0.000, -0.004, -0.013, -0.015, -0.017, -0.018, -0.019, -0.019, -0.019,
        -0.019, -0.019, -0.019, -0.019, -0.019, -0.018, -0.018, -0.018, -0.017,
        -0.017, -0.017, -0.017, -0.016, -0.016, -0.016, -0.016, -0.016, -0.015,
        -0.015, -0.015, -0.015, -0.015,
    ],
    "z_pooled": [
        0.010, -0.008, -0.008, -0.000, 0.001, 0.002, 0.002, 0.003, 0.003, 0.003,
        0.004, 0.004, 0.005, 0.005, 0.005, 0.005, 0.006, 0.006, 0.006, 0.007,
        0.007, 0.007, 0.007, 0.007, 0.008, 0.008, 0.008, 0.008, 0.008, 0.009,
        0.009,
    ],
    "s_z": [
        -0.003, -0.003, 0.000, -0.004, -0.008, -0.010, -0.011, -0.012, -0.012,
        -0.012, -0.012, -0.012, -0.011, -0.011, -0.011, -0.010, -0.010, -0.010,
        -0.009, -0.009, -0.009, -0.008, -0.008, -0.008, -0.008, -0.008, -0.007,
        -0.007, -0.007, -0.007, -0.006,
    ],
}

# Complexa s_z val_eval learning curve (sparser step grid).
COMPLEXA_SZ_CURVE_STEPS = [
    0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000,
    2500, 3000,
]
COMPLEXA_SZ_CURVE_VALS = [
    -0.015, -0.001, 0.211, 0.510, 0.618, 0.663, 0.686, 0.700, 0.709, 0.715,
    0.720, 0.727, 0.733, 0.737, 0.739, 0.740,
]

# --------------------------------------------------------------------------- #
# Sanity checks on inlined data (fail loud if a curve was mistyped)
# --------------------------------------------------------------------------- #

for _name, _vals in LAPROTEINA_CURVES.items():
    assert len(_vals) == len(_LAP_STEPS), (
        f"La-Proteina curve '{_name}' has {len(_vals)} points, "
        f"expected {len(_LAP_STEPS)}"
    )
    # Final logged curve value must match the FINAL table's val_eval to within
    # the curve dump's 3-decimal rounding (table is logged to 4 decimals).
    assert abs(_vals[-1] - LAPROTEINA_FINAL[_name][2]) < 6e-4, (
        f"La-Proteina curve '{_name}' endpoint {_vals[-1]} != "
        f"final val_eval {LAPROTEINA_FINAL[_name][2]}"
    )
assert len(COMPLEXA_SZ_CURVE_STEPS) == len(COMPLEXA_SZ_CURVE_VALS)
assert abs(COMPLEXA_SZ_CURVE_VALS[-1] - COMPLEXA_FINAL["s_z"][2]) < 6e-4

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #

# Colorblind-safe (Wong / Okabe-Ito based) palette.
C_COMPLEXA = "#0072B2"   # blue   -> Complexa
C_LAPROT = "#D55E00"     # vermillion -> La-Proteina
C_ZERO = "#555555"       # neutral gray for the r=0 reference
C_FLOOR = "#009E73"      # bluish green for the r=0.74 reference

# Per-variant hues (used where variants are compared within the curve plots).
VARIANT_COLORS = {
    "s_only":    "#0072B2",  # blue
    "s_latents": "#E69F00",  # orange
    "z_pooled":  "#009E73",  # green
    "s_z":       "#CC79A7",  # reddish purple
}
VARIANT_LABELS = {
    "s_only":    "s_only (768-d)",
    "s_latents": "s_latents (776-d, adaptor input)",
    "z_pooled":  "z_pooled (768-d)",
    "s_z":       "s_z (concat 1536-d)",
}

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.fontsize": 9.5,
        "legend.framealpha": 0.9,
    }
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_figs")
os.makedirs(OUT_DIR, exist_ok=True)

FLOOR_REF = 0.74  # Complexa's headline linear floor

CAPTION = (
    "Model-vs-native-data comparison: Complexa measured on Teddymer DIMERS, "
    "La-Proteina on SwissProt MONOMERS (AF2 B-factor pLDDT) - not identical "
    "structures. La-Proteina val_eval sampled 200 readable proteins from its "
    "val split (~10% of staged .pt files unreadable/skipped), so val_eval "
    "approximates the training-run val membership. A near-zero LINEAR floor "
    "does NOT mean La-Proteina features are useless for pLDDT: the trained QG "
    "head (adaptor + Boltz pairformer) reached val Pearson 0.97 - the signal "
    "is NONLINEAR, unlike Complexa's high LINEAR floor."
)


def _finalize(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- #
# Figure 1: headline grouped bar chart of the final val_eval floor
# --------------------------------------------------------------------------- #

def fig_headline() -> str:
    # Variant order along x. La-Proteina-only s_latents shown as an extra bar.
    variants = ["s_only", "s_latents", "z_pooled", "s_z"]
    x = np.arange(len(variants))
    width = 0.38

    comp_vals = [COMPLEXA_FINAL.get(v, (None, None, None))[2] for v in variants]
    lap_vals = [LAPROTEINA_FINAL.get(v, (None, None, None))[2] for v in variants]

    fig, ax = plt.subplots(figsize=(10.0, 6.2))

    # Reference bands: near r=0 and near r=0.74.
    ax.axhspan(-0.03, 0.03, color=C_ZERO, alpha=0.12, zorder=0)
    ax.axhspan(FLOOR_REF - 0.02, FLOOR_REF + 0.02, color=C_FLOOR, alpha=0.14,
               zorder=0)
    ax.axhline(0.0, color=C_ZERO, lw=1.0, ls="--", alpha=0.7, zorder=1)
    ax.axhline(FLOOR_REF, color=C_FLOOR, lw=1.0, ls="--", alpha=0.8, zorder=1)

    def _draw(offset, vals, color, label):
        bars = []
        for xi, v in zip(x, vals):
            if v is None:
                continue
            b = ax.bar(xi + offset, v, width, color=color, edgecolor="black",
                       linewidth=0.6, zorder=3,
                       label=label if not bars else None)
            bars.append((xi + offset, v))
        return bars

    comp_bars = _draw(-width / 2, comp_vals, C_COMPLEXA, "Complexa (Teddymer dimers)")
    lap_bars = _draw(+width / 2, lap_vals, C_LAPROT, "La-Proteina (SwissProt monomers)")

    # Annotate each bar with its value.
    def _annotate(bars):
        for xc, v in bars:
            va = "bottom" if v >= 0 else "top"
            off = 0.012 if v >= 0 else -0.012
            ax.text(xc, v + off, f"{v:.3f}", ha="center", va=va,
                    fontsize=9.5, fontweight="bold", zorder=4)

    _annotate(comp_bars)
    _annotate(lap_bars)

    # Mark Complexa's missing s_latents.
    slat_idx = variants.index("s_latents")
    ax.text(slat_idx - width / 2, 0.02, "n/a", ha="center", va="bottom",
            fontsize=9, style="italic", color=C_COMPLEXA, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=11)
    ax.set_xlabel("frozen-feature variant")
    ax.set_ylabel("held-out val Pearson r  (linear pLDDT floor)")
    ax.set_ylim(-0.10, 1.02)
    ax.set_title("Frozen-feature linear pLDDT floor: Complexa vs La-Proteina",
                 fontweight="bold", pad=10)

    # Reference-band legend entries.
    band_handles = [
        Patch(facecolor=C_FLOOR, alpha=0.14, label="r ~= 0.74 (Complexa floor)"),
        Patch(facecolor=C_ZERO, alpha=0.12, label="r ~= 0 (no linear signal)"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    # Legend goes upper-centre: upper-left/right sit over the ~0.74 Complexa
    # bars and would occlude the s_z value annotation.
    ax.legend(handles + band_handles, labels + [h.get_label() for h in band_handles],
              loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99))

    # Caption below the axes.
    fig.text(0.5, -0.02, _wrap(CAPTION, 118), ha="center", va="top",
             fontsize=8.0, color="#333333", wrap=True)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "headline_floor_comparison.png")
    _finalize(fig, path)
    return path


# --------------------------------------------------------------------------- #
# Figure 2: La-Proteina learning curves, zoomed to show flatness at zero
# --------------------------------------------------------------------------- #

def fig_laproteina_curves() -> str:
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    ax.axhline(0.0, color=C_ZERO, lw=1.2, ls="--", alpha=0.8, zorder=1,
               label="r = 0")

    for v in ["s_only", "s_latents", "z_pooled", "s_z"]:
        ax.plot(_LAP_STEPS, LAPROTEINA_CURVES[v], marker="o", ms=3.5,
                lw=1.8, color=VARIANT_COLORS[v], label=VARIANT_LABELS[v],
                zorder=3)

    ax.set_xlim(0, 3000)
    ax.set_ylim(-0.05, 0.05)
    ax.set_xlabel("SGD probe step")
    ax.set_ylabel("held-out val Pearson r")
    ax.set_title("La-Proteina frozen-feature probe: val Pearson vs step",
                 fontweight="bold")
    ax.legend(loc="upper right", ncol=1)

    ax.annotate(
        "all variants flat at ~0 across 3000 steps - no linear signal",
        xy=(1400, 0.0), xytext=(1050, -0.030),
        ha="center", fontsize=10.5, fontweight="bold", color="#333333",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0),
    )

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "la_proteina_learning_curves.png")
    _finalize(fig, path)
    return path


# --------------------------------------------------------------------------- #
# Figure 3: overlay of the best-variant curve from each model
# --------------------------------------------------------------------------- #

def fig_overlay() -> str:
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    ax.axhline(0.0, color=C_ZERO, lw=1.0, ls="--", alpha=0.7, zorder=1)
    ax.axhline(FLOOR_REF, color=C_FLOOR, lw=1.0, ls="--", alpha=0.8, zorder=1)
    ax.text(3000, FLOOR_REF + 0.012, "r = 0.74", ha="right", va="bottom",
            fontsize=9.5, color=C_FLOOR)
    ax.text(3000, 0.012, "r = 0", ha="right", va="bottom",
            fontsize=9.5, color=C_ZERO)

    # Complexa s_z: rises fast to ~0.74.
    ax.plot(COMPLEXA_SZ_CURVE_STEPS, COMPLEXA_SZ_CURVE_VALS, marker="o", ms=4.5,
            lw=2.4, color=C_COMPLEXA, zorder=4,
            label="Complexa  s_z  (Teddymer dimers)")

    # La-Proteina s_z and s_latents: flat at ~0.
    ax.plot(_LAP_STEPS, LAPROTEINA_CURVES["s_z"], marker="s", ms=3.5, lw=2.0,
            color=C_LAPROT, zorder=3,
            label="La-Proteina  s_z  (SwissProt monomers)")
    ax.plot(_LAP_STEPS, LAPROTEINA_CURVES["s_latents"], marker="^", ms=3.5,
            lw=2.0, color=C_LAPROT, ls="--", alpha=0.85, zorder=3,
            label="La-Proteina  s_latents  (adaptor input)")

    ax.set_xlim(0, 3000)
    ax.set_ylim(-0.1, 0.8)
    ax.set_xlabel("SGD probe step")
    ax.set_ylabel("held-out val Pearson r  (linear pLDDT floor)")
    ax.set_title("Linear pLDDT floor: one model climbs to 0.74, the other stays at 0",
                 fontweight="bold")
    ax.legend(loc="center right")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "floor_curves_overlay.png")
    _finalize(fig, path)
    return path


# --------------------------------------------------------------------------- #
# Small text-wrap helper (no external deps)
# --------------------------------------------------------------------------- #

def _wrap(text: str, width: int) -> str:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    print(f"Output dir: {OUT_DIR}")
    p1 = fig_headline()
    p2 = fig_laproteina_curves()
    p3 = fig_overlay()

    print()
    print("=" * 78)
    print("FIGURES WRITTEN")
    print("=" * 78)
    for p in (p1, p2, p3):
        print(f"  {p}")

    print()
    print("Numbers plotted (final val_eval Pearson r, the linear floor):")
    print("  Complexa    : s_only=0.7311  z_pooled=0.4818  s_z=0.7402  (no s_latents)")
    print("  La-Proteina : s_only=-0.0149 s_latents=-0.0147 z_pooled=0.0089 s_z=-0.0065")

    print()
    print("CAVEATS (also captioned on the headline figure):")
    print("  * Model-vs-native-data: Complexa on Teddymer DIMERS; La-Proteina on")
    print("    SwissProt MONOMERS (AF2 B-factor pLDDT). Not identical structures.")
    print("  * La-Proteina val_eval sampled 200 readable proteins from its val split")
    print("    (~10% of staged .pt files unreadable/skipped) -> approximates the")
    print("    training-run val membership.")
    print("  * The near-zero LINEAR floor does NOT mean La-Proteina features are")
    print("    useless for pLDDT: the trained QG head (adaptor + Boltz pairformer)")
    print("    reached val Pearson 0.97 -> the signal is NONLINEAR, unlike Complexa's")
    print("    high LINEAR floor.")


if __name__ == "__main__":
    main()
