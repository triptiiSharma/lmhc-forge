#!/usr/bin/env python3
"""style.py — LMHC-Forge shared visual style module (Phase 3).

Single source of truth for the five-color palette and shared figure styling
(font, sizing, DPI, gridlines) used by every figure in visualize.py. Import
this module and call apply_style() once before creating any figure; use the
named color constants and the redundant-encoding helpers below instead of
hardcoding hex values or relying on matplotlib's default colors/black text.

Palette (exact hex — do not introduce other colors into a figure):
    WASABI          #E9F056  sparing single-highlight annotations ONLY —
                              never a bulk fill or a whole data series.
    ORANGE_TOPAZE   #FF5C34  FAIL / danger / residue-present / before-
                              mitigation — reserved exclusively for this role.
    COOL_BLUE       #D7EFFF  background tint / neutral panels / structural
                              (non-pass-fail) data series.
    CASSIS          #351E28  primary text, axis lines, borders, dark accents
                              — replaces matplotlib's default black everywhere.
    VERT_SAUGE      #AEB8A0  PASS / safe / after-mitigation — reserved
                              exclusively for this role.

Orange Topaze vs Vert Sauge is a red/green pair indistinguishable under
red-green color vision deficiency. Any figure using both must ALSO encode the
same distinction redundantly (text label, hatch pattern, or a check/cross
icon) — see pass_fail_kwargs()/pass_fail_label() and the *_ICON constants
below.
"""

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

# ── Palette ───────────────────────────────────────────────────────────────
WASABI        = "#E9F056"
ORANGE_TOPAZE = "#FF5C34"
COOL_BLUE     = "#D7EFFF"
CASSIS        = "#351E28"
VERT_SAUGE    = "#AEB8A0"

PALETTE = {
    "wasabi": WASABI,
    "orange_topaze": ORANGE_TOPAZE,
    "cool_blue": COOL_BLUE,
    "cassis": CASSIS,
    "vert_sauge": VERT_SAUGE,
}

# ── Shared typography / output settings ──────────────────────────────────
# DejaVu Sans ships bundled with matplotlib itself, so it renders identically
# on any machine with matplotlib installed -- including the WSL2/Kali Linux
# host the actual CUDA pipeline runs on, which cannot be assumed to have
# Arial/Segoe/other system fonts this Windows dev machine happens to have.
FONT_FAMILY = "DejaVu Sans"
DPI = 300  # print-resolution PNG export; PDF is vector regardless


def apply_style() -> None:
    """Set shared rcParams. Idempotent — safe to call from every figure fn
    and from repeated visualize.py runs."""
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": 9,
        "figure.titlesize": 11,
        "figure.titleweight": "bold",
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.edgecolor": CASSIS,
        "axes.labelcolor": CASSIS,
        "axes.linewidth": 0.8,
        "text.color": CASSIS,
        "xtick.color": CASSIS,
        "ytick.color": CASSIS,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.edgecolor": CASSIS,
        "legend.facecolor": "white",
        "savefig.dpi": DPI,
        "grid.color": CASSIS,
        "grid.alpha": 0.15,
        "grid.linewidth": 0.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=[CASSIS, COOL_BLUE, VERT_SAUGE, ORANGE_TOPAZE, WASABI]),
    })


# ── Redundant encoding for the Orange Topaze / Vert Sauge pairing ──────────
# Phase 3.1 accessibility requirement: this pairing is a red/green contrast
# indistinguishable under red-green color vision deficiency, so every figure
# using it must ALSO carry a non-color cue (hatch, icon, or text label).
PASS_HATCH  = ""
FAIL_HATCH  = "///"
LIMIT_HATCH = "xxx"
PASS_ICON   = "✓"   # check mark  (PASS / safe / after-mitigation)
FAIL_ICON   = "✗"   # cross mark  (FAIL / leak / before-mitigation)
LIMIT_ICON  = "⚠"   # warning triangle (known limitation, e.g. T6)
NA_ICON     = "–"   # en dash (not reached / not applicable)


def pass_fail_kwargs(is_pass: bool) -> dict:
    """matplotlib bar/patch kwargs for a PASS/FAIL value: color + hatch,
    so the distinction survives grayscale printing and red-green color
    vision deficiency, not just the Vert Sauge / Orange Topaze hue."""
    return dict(
        color=VERT_SAUGE if is_pass else ORANGE_TOPAZE,
        hatch=PASS_HATCH if is_pass else FAIL_HATCH,
        edgecolor=CASSIS,
        linewidth=0.8,
    )


def pass_fail_label(is_pass: bool, value_text: str) -> str:
    """Value label carrying the redundant icon, e.g. '✗ EI = 1.000'."""
    icon = PASS_ICON if is_pass else FAIL_ICON
    return f"{icon} {value_text}"


def savefig_both(fig, out_dir, stem: str) -> None:
    """Save fig as both 300-dpi PNG and vector PDF under out_dir/stem.*."""
    from pathlib import Path
    out_dir = Path(out_dir)
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        kw = dict(bbox_inches="tight")
        if ext == "png":
            kw["dpi"] = DPI
        fig.savefig(p, **kw)
        print(f"  {p.name}")
    plt.close(fig)
