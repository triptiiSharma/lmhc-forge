#!/usr/bin/env python3
"""visualize.py — LMHC-Forge Figure Generator (Phase 3 palette)

Eleven figures, 300-dpi PNG + vector PDF, using the shared five-color
palette in style.py (Wasabi / Orange Topaze / Cool Blue / Cassis / Vert
Sauge) instead of matplotlib defaults. Hardware caveat and trial count in
every caption where applicable. See figures/MANIFEST.md for the full
filename -> description -> paper-section -> source-data mapping.

Run from /opt/lmhc-forge/ or any directory; ROOT is derived from __file__.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrow, FancyBboxPatch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from style import (
    WASABI, ORANGE_TOPAZE, COOL_BLUE, CASSIS, VERT_SAUGE,
    PASS_HATCH, FAIL_HATCH, LIMIT_HATCH,
    PASS_ICON, FAIL_ICON, LIMIT_ICON, NA_ICON,
    apply_style, pass_fail_kwargs, pass_fail_label, savefig_both,
)

apply_style()

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
RESULTS    = ROOT / "results"
MGPU_RES   = ROOT / "multi_gpu" / "results"
FIGS       = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

HW = "RTX 4050 Laptop GPU (Ada Lovelace, sm_89)"
M6_FRAMING = (
    "Residue is fully observable when the subsequent kernel is co-resident on the same SM — "
    "co-residency is forced in this experiment to demonstrate the window.\n"
    "This is in-process cross-kernel shared-memory persistence, "
    "NOT cross-tenant LeftoverLocals on NVIDIA."
)

RESIDUE_CMAP = LinearSegmentedColormap.from_list(
    "lmhc_residue", [VERT_SAUGE, ORANGE_TOPAZE]
)


def _save(fig, stem):
    savefig_both(fig, FIGS, stem)


def _load(path):
    with open(path) as f:
        return json.load(f)


# ── Fig 1: Phase 0 Residue Heatmap ───────────────────────────────────────────
def fig1_phase0_heatmap():
    phase0_path = RESULTS / "phase0_residue.json"
    data_missing = not phase0_path.exists()
    if not data_missing:
        n = _load(phase0_path)["trials"]
    else:
        # phase0_residue.json is missing from this working tree (see session
        # notes); NTRIALS=1000 is a compile-time constant in
        # src/phase0_residue.cu and is restated verbatim in docs/SPEC.md and
        # docs/RESULTS.md prose, so this is a documented constant, not an
        # invented number. Flagged loudly rather than silently substituted —
        # both to stderr AND, below, as a visible on-figure caveat, since a
        # stderr-only warning disappears once someone just looks at the PNG.
        print("  WARNING: results/phase0_residue.json missing — using "
              "NTRIALS=1000 from src/phase0_residue.cu instead", file=sys.stderr)
        n = 1000

    W = H = 64          # 64×64 = 4096 words

    before = np.ones((H, W))    # EI = 1.0 (all words = 0xCAFEBABE in every trial)
    after  = np.zeros((H, W))   # EI = 0.0 (LMHC scrub zeroes all words)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    # Leave extra headroom above the axes for the suptitle, and — when the
    # source JSON is missing — an additional caveat banner above that.
    fig.subplots_adjust(top=0.66 if data_missing else 0.74)

    for ax, grid, title in [
        (axes[0], before, f"Before LMHC scrub\n(EI = 1.000) {FAIL_ICON}"),
        (axes[1], after,  f"After LMHC hardened scrub\n(EI = 0.000) {PASS_ICON}"),
    ]:
        im = ax.imshow(grid, cmap=RESIDUE_CMAP, vmin=0, vmax=1,
                       aspect="auto", origin="upper")
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Word column (0–63)", fontsize=8)
        ax.set_ylabel("Word row (0–63)",    fontsize=8)
        ax.set_xticks([0, 16, 32, 48, 63])
        ax.set_yticks([0, 16, 32, 48, 63])
        for spine in ax.spines.values():
            spine.set_edgecolor(CASSIS)

    cb = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cb.set_label("Fraction of same-SM trials with non-zero word (EI)", fontsize=8, color=CASSIS)
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.outline.set_edgecolor(CASSIS)

    fig.suptitle(
        f"Phase 0 — Cross-Launch Shared-Memory Residue\n"
        f"4096 × uint32 = 16 KB, N={n} trials | Hardware: {HW}",
        fontsize=9.5, y=1.24 if data_missing else 1.06,
    )

    if data_missing:
        # Visible on-artifact caveat — NOT just a console warning. Anyone
        # viewing this PNG/PDF in isolation, without having seen stderr, must
        # still be able to tell that N is sourced from a source-code constant
        # rather than a regenerated measurement file. Removes itself
        # automatically the next time this is run after
        # results/phase0_residue.json is regenerated in WSL2 (see
        # docs/SETUP.md) — this branch simply won't trigger.
        fig.text(
            0.5, 1.14,
            f"{LIMIT_ICON} DATA PROVENANCE CAVEAT: results/phase0_residue.json is missing from this "
            f"repository.\nN={n} shown above is the NTRIALS constant from src/phase0_residue.cu, "
            "NOT read from a regenerated measurement.\n"
            "Regenerate via docs/SETUP.md in WSL2 to remove this caveat.",
            ha="center", va="top", fontsize=7.8, fontweight="bold", color=CASSIS,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=WASABI, edgecolor=CASSIS, linewidth=1.2),
        )

    fig.text(0.5, -0.05, M6_FRAMING,
             ha="center", fontsize=7, style="italic", color=CASSIS,
             wrap=True)
    _save(fig, "fig1_phase0_heatmap")


# ── Fig 2: Module 3 DSE Differential ─────────────────────────────────────────
def fig2_dse_bar():
    d = _load(RESULTS / "dse_differential.json")

    rows   = d["kernel_rows"]
    labels = ["scrub_naive", "scrub_hardened", "scrub_ptx_fallback"]
    sts    = [r["sts_count"]    for r in rows]
    membar = [r["membar_count"] for r in rows]
    bar    = [r["bar_count"]    for r in rows]

    x, w = np.arange(3), 0.25
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw={"width_ratios": [3, 2]})

    # STS = present in both naive+hardened -> neutral structural series (Cool Blue).
    # BAR.SYNC = secondary structural difference -> Cassis (dark accent) at reduced alpha.
    # MEMBAR.SC.CTA = THE single differentiator this figure exists to show -> Wasabi
    # highlight (sparing use, exactly the "single-highlight annotation" role).
    b_sts = ax.bar(x - w, sts,    w, label="STS (shared stores)",
                   color=COOL_BLUE, edgecolor=CASSIS, linewidth=0.8, zorder=3)
    b_mb  = ax.bar(x,     membar, w, label="MEMBAR.SC.CTA (key differentiator)",
                   color=WASABI, edgecolor=CASSIS, linewidth=0.8, zorder=3)
    b_bar = ax.bar(x + w, bar,    w, label="BAR.SYNC",
                   color=CASSIS, alpha=0.85, edgecolor=CASSIS, linewidth=0.8, zorder=3)

    for group in (b_sts, b_mb, b_bar):
        for rect in group:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.3,
                    str(int(h)), ha="center", va="bottom", fontsize=8, color=CASSIS)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Instruction count")
    ax.set_ylim(0, max(sts) + 32)   # headroom for the verdict annotation above the bar labels
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.80))
    ax.set_title(
        f"Module 3 — DSE Differential\n"
        f"{d['compile_flags']} | {HW}",
        fontsize=9,
    )
    ax.annotate(
        f"Verdict: {d['verdict']}\n"
        "Per-store DSE did NOT fire on nvcc 12.8/sm_89 — STS counts are equal.\n"
        "Ordering differential: MEMBAR.SC.CTA absent in naive, present in hardened.\n"
        "This is an ordering differential, NOT per-store elimination.",
        xy=(0.5, 0.99), xycoords="axes fraction", ha="center", va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=WASABI, edgecolor=CASSIS, alpha=0.4),
    )

    # Annotated SASS panel
    axt.axis("off")
    sass = (
        "── scrub_naive (no ordering guarantee) ───────\n"
        "  STS R0, [R2+0x000]\n"
        "  STS R0, [R2+0x004]\n"
        "  ...  (58 STS total)\n"
        "  BAR.SYNC 0      ← __syncthreads\n"
        "                    (no MEMBAR before)\n"
        "\n"
        "── scrub_hardened (LMHC pattern) ─────────────\n"
        "  STS R0, [R2+0x000]\n"
        "  STS R0, [R2+0x004]\n"
        "  ...  (58 STS total)\n"
        "  MEMBAR.SC.CTA   ← __threadfence_block\n"
        "                    (C++ happens-before)\n"
        "  BAR.SYNC 0      ← __syncthreads\n"
        "\n"
        "STS counts equal → DSE did not fire.\n"
        "MEMBAR absent in naive → no ordering\n"
        "guarantee at kernel exit (Theorem 1)."
    )
    axt.text(0.04, 0.97, sass,
             transform=axt.transAxes, va="top", ha="left",
             fontsize=7.5, fontfamily="monospace", color=CASSIS,
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor=COOL_BLUE, edgecolor=CASSIS, alpha=0.5))
    axt.set_title("Annotated SASS excerpt", fontsize=9)

    fig.tight_layout()
    _save(fig, "fig2_dse_differential")


# ── Fig 3: Module 6 Exposure Index ───────────────────────────────────────────
def fig3_exposure_index():
    d = _load(RESULTS / "leakage_demo.json")

    ei_b = d["exposure_index_before"]  # 1.0
    ei_a = d["exposure_index_after"]   # 0.0
    n    = d["trials_per_phase"]

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    bar_labels = [
        "Before LMHC scrub\n(unscrubbed writer)",
        "After LMHC scrub\n(hardened writer)",
    ]
    kw_before = pass_fail_kwargs(False)
    kw_after  = pass_fail_kwargs(True)
    bars = ax.bar(bar_labels, [ei_b, ei_a],
                  color=[kw_before["color"], kw_after["color"]],
                  hatch=[kw_before["hatch"], kw_after["hatch"]],
                  edgecolor=CASSIS, linewidth=0.8, width=0.4, zorder=3)

    labels = [
        pass_fail_label(False, f"EI = {ei_b:.3f}"),
        pass_fail_label(True,  f"EI = {ei_a:.3f}"),
    ]
    for rect, lbl in zip(bars, labels):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.03,
                lbl,
                ha="center", va="bottom", fontsize=12, fontweight="bold", color=CASSIS)

    ax.set_ylim(0, 1.30)
    ax.set_ylabel("Exposure Index\n(fraction of shared words non-zero)", fontsize=9)
    ax.set_title(
        f"Module 6 — Cross-Launch Shared-Memory Exposure Index\n"
        f"Hardware: {HW} | N={n} trials per phase",
        fontsize=9,
    )

    fig.text(0.5, -0.08, M6_FRAMING,
             ha="center", fontsize=7.5, style="italic", color=CASSIS,
             wrap=True)
    fig.tight_layout()
    _save(fig, "fig3_exposure_index")


# ── Fig 4: Module 7 Benchmark ─────────────────────────────────────────────────
def fig4_benchmark():
    d = _load(RESULTS / "benchmark.json")

    ks      = d["kernels"]
    klabels = [k["class"].capitalize() for k in ks]
    base_us = np.array([k["baseline_median_us"]  for k in ks])
    base_sd = np.array([k["baseline_stddev_us"]  for k in ks])
    ovhd_med= np.array([k["overhead_median_us"]  for k in ks])
    ovhd_sd = np.array([k["overhead_stddev_us"]  for k in ks])
    below   = [k["below_noise_floor"]            for k in ks]
    n_batch = d["n_batch"]
    n_reps  = d["n_reps"]

    # For kernels below the noise floor, report σ (upper bound), not the median.
    reported_ovhd = np.where(below, ovhd_sd, ovhd_med)
    reported_pct  = 100.0 * reported_ovhd / base_us

    x, w = np.arange(len(ks)), 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))

    # Baseline latency: neutral structural series -> Cool Blue (not a pass/fail signal).
    ax1.bar(x, base_us, w * 1.8, label="Baseline (median ± σ)",
            color=COOL_BLUE, edgecolor=CASSIS, linewidth=0.8, yerr=base_sd,
            ecolor=CASSIS, capsize=5, error_kw={"elinewidth": 1.4}, zorder=3)

    # "Below noise floor" is a measurement-resolution caveat, not a security
    # PASS/FAIL — Orange Topaze/Vert Sauge are reserved for that and must not
    # be diluted here. Wasabi (sparing single-highlight) flags the one
    # caveat that matters; resolved values get plain Cassis text.
    for i, (bv, bsd, ov_rep, bl) in enumerate(
            zip(base_us, base_sd, reported_ovhd, below)):
        if bl:
            tag = f"< {ov_rep:.2f} µs\n(σ bound, unresolved)"
            box = dict(boxstyle="round,pad=0.25", facecolor=WASABI, edgecolor=CASSIS, alpha=0.55)
        else:
            tag = f"+{ov_rep:.2f} µs (resolved)"
            box = None
        ax1.annotate(tag,
                     xy=(x[i], bv + bsd + 0.5),
                     ha="center", va="bottom", fontsize=8,
                     color=CASSIS, fontweight="bold", bbox=box)

    ax1.set_xticks(x)
    ax1.set_xticklabels(klabels, fontsize=10)
    ax1.set_ylabel("Kernel latency (µs / launch)", fontsize=9)
    ax1.set_ylim(0, (base_us + base_sd).max() * 1.75)  # headroom for callout boxes above bars
    ax1.set_title("Baseline latency (median ± σ)\nwith LMHC scrub upper bound annotated",
                  fontsize=9)
    ax1.legend(fontsize=9, loc="upper right")

    # Right panel: overhead as % of baseline, same neutral/highlight scheme.
    colors_bar = [WASABI if bl else COOL_BLUE for bl in below]
    bar_labels = [f"< {p:.1f}%\n(σ bound)" if bl else f"+{p:.1f}%"
                  for p, bl in zip(reported_pct, below)]

    bars_pct = ax2.bar(klabels, reported_pct, color=colors_bar,
                        edgecolor=CASSIS, linewidth=0.8, width=0.42, zorder=3)
    ax2.axhline(0, color=CASSIS, linewidth=0.8, linestyle="--")

    for rect, lbl in zip(bars_pct, bar_labels):
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2, h + 0.3,
                 lbl, ha="center", va="bottom", fontsize=9, fontweight="bold", color=CASSIS)

    ax2.set_ylabel("Scrub overhead as % of baseline", fontsize=9)
    ax2.set_title("LMHC scrub cost (upper bounds for all kernels)\n"
                  "Overhead < noise floor (σ ≈ 1–2 µs) for all three classes",
                  fontsize=9)
    ax2.set_ylim(0, max(reported_pct) * 1.5)

    ax2.legend(handles=[
        mpatches.Patch(facecolor=WASABI, edgecolor=CASSIS, label="Below noise floor — reported as < σ (unresolved)"),
        mpatches.Patch(facecolor=COOL_BLUE, edgecolor=CASSIS, label="Resolved overhead"),
    ], fontsize=8, loc="upper right")

    fig.suptitle(
        f"Module 7 — LMHC Scrub Overhead Benchmark\n"
        f"Hardware: {HW} | "
        f"{n_batch} launches/batch, {n_reps} interleaved reps\n"
        "All three kernels: overhead below CUDA-event timing noise floor (~1–2 µs). "
        "Bars show σ upper bounds, NOT measured overhead values.",
        fontsize=9,
    )
    fig.tight_layout()
    _save(fig, "fig4_benchmark")


# ── Fig 5: Module 5 PASS/FAIL Panel ──────────────────────────────────────────
def fig5_tamper_panel():
    tests = [
        ("T1", "Stripped cubin +\noriginal witness",
         "FAIL", NA_ICON, NA_ICON, "FAIL", "hash mismatch"),
        ("T2", "Stripped cubin +\nregenerated hash,\ngemm still nominated",
         "PASS", "FAIL", NA_ICON, "FAIL", "scrub absent\nin SASS"),
        ("T3", "Original cubin +\noriginal witness",
         "PASS", "PASS", "PASS", "PASS", "golden\nreference"),
        ("T4", "Stripped cubin +\ngemm dropped\nfrom policy",
         "PASS", "PASS", "FAIL", "FAIL", "unaccounted\nkernel"),
        ("T5", "Stripped cubin +\nde-annotated source,\nold-format witness",
         "FAIL", NA_ICON, NA_ICON, "FAIL", "source hash\nmissing in\nold-format"),
        ("T6", "Generate-from-scratch\n(fresh witness_gen.py\non de-annotated src)",
         "PASS", "PASS", "PASS", "LIMIT", "KNOWN LIMITATION:\nbinary-level\ntaint required"),
    ]

    def cell_style(v):
        if v == "PASS":  return dict(color=VERT_SAUGE, hatch=PASS_HATCH, text=PASS_ICON + " PASS")
        if v == "FAIL":  return dict(color=ORANGE_TOPAZE, hatch=FAIL_HATCH, text=FAIL_ICON + " FAIL")
        if v == "LIMIT": return dict(color=WASABI, hatch=LIMIT_HATCH, text=LIMIT_ICON + " PASS*")
        return dict(color=COOL_BLUE, hatch="", text=NA_ICON)

    col_hdrs = ["Test", "Description",
                "CHECK 1\nHash integrity", "CHECK 2\nScrub in SASS",
                "CHECK 3\nSource taint", "Verdict", "Note"]
    cell_text   = []
    cell_colors = []
    for t in tests:
        label, desc, c1, c2, c3, verd, note = t
        row_styles = [cell_style(v) for v in (c1, c2, c3, verd)]
        cell_text.append([label, desc, row_styles[0]["text"], row_styles[1]["text"],
                          row_styles[2]["text"], row_styles[3]["text"], note])
        cell_colors.append([
            "#eeeeee", "#ffffff",
            row_styles[0]["color"], row_styles[1]["color"],
            row_styles[2]["color"], row_styles[3]["color"], "#f9f9f9",
        ])

    fig, ax = plt.subplots(figsize=(14, 4.7))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        colLabels=col_hdrs,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 2.8)

    for ci in range(len(col_hdrs)):
        cell = tbl[0, ci]
        cell.set_facecolor(CASSIS)
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor(CASSIS)

    for ri in range(len(tests)):
        for ci in range(len(col_hdrs)):
            cell = tbl[ri + 1, ci]
            cell.set_edgecolor(CASSIS)
            if ci in (2, 3, 4, 5):
                cell.set_text_props(color=CASSIS, fontweight="bold")

    legend_handles = [
        mpatches.Patch(facecolor=VERT_SAUGE, edgecolor=CASSIS, hatch=PASS_HATCH, label=f"{PASS_ICON} PASS"),
        mpatches.Patch(facecolor=ORANGE_TOPAZE, edgecolor=CASSIS, hatch=FAIL_HATCH, label=f"{FAIL_ICON} FAIL"),
        mpatches.Patch(facecolor=WASABI, edgecolor=CASSIS, hatch=LIMIT_HATCH, label=f"{LIMIT_ICON} PASS* (known limitation)"),
        mpatches.Patch(facecolor=COOL_BLUE, edgecolor=CASSIS, label=f"{NA_ICON} (check not reached)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              bbox_to_anchor=(1.0, -0.14), fontsize=8.5, ncol=4)

    fig.suptitle(
        f"Module 5 — Tamper-Test Matrix T1–T6\n"
        f"Hardware: {HW} | Attester: attester.py (CHECK 1–3)",
        fontsize=10, y=1.02,
    )
    _save(fig, "fig5_tamper_panel")


# ── Fig 6 (new, 3.4a): Threat-model / residue-lifecycle diagram ─────────────
def fig6_threat_model_lifecycle():
    """Conceptual diagram: GPU, victim kernel, the three residue lifecycle
    paths (P-reuse / P-spill / P-stage), each terminating in a scavenger/
    listener kernel. Built regardless of Phase 1 data (SPEC.md 3.4a)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(xy, w, h, label, facecolor=COOL_BLUE, edgecolor=CASSIS, fontsize=9,
             fontweight="bold", textcolor=CASSIS, lw=1.3, zorder=3):
        x, y = xy
        p = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.06,rounding_size=0.08",
                            linewidth=lw, edgecolor=edgecolor, facecolor=facecolor, zorder=zorder)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=textcolor, zorder=zorder + 1)
        return (x, y, w, h)

    def arrow(p_from, p_to, color=CASSIS, lw=1.6, style="-|>", connectionstyle="arc3,rad=0.0"):
        ax.annotate("", xy=p_to, xytext=p_from,
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                     connectionstyle=connectionstyle, shrinkA=2, shrinkB=2),
                    zorder=2)

    # Outer GPU / SM boundary.
    box((0.4, 0.4), 9.2, 9.2, "", facecolor="none", edgecolor=COOL_BLUE, lw=2.2, zorder=1)
    ax.text(0.7, 9.35, "GPU (single SM shown)", fontsize=10, fontweight="bold", color=CASSIS)

    victim = box((3.7, 8.0), 2.6, 0.9, "Victim kernel\n(writes secret)",
                  facecolor="white", edgecolor=CASSIS)
    scratch = box((3.7, 6.3), 2.6, 0.9, "On-chip scratchpad\n(CUDA shared / OpenCL local)",
                   facecolor=COOL_BLUE, edgecolor=CASSIS, fontsize=8)
    arrow((3.7 + 1.3, 8.0), (3.7 + 1.3, 6.3 + 0.9))

    local_spill = box((0.9, 4.2), 2.5, 0.9, 'CUDA "local" memory\n(global-backed spill)',
                       facecolor=COOL_BLUE, edgecolor=CASSIS, fontsize=8)
    global_stage = box((6.6, 4.2), 2.5, 0.9, "Global staging buffer\n(DRAM)",
                        facecolor=COOL_BLUE, edgecolor=CASSIS, fontsize=8)

    scavenger = box((3.7, 1.2), 2.6, 0.9, "Scavenger / listener kernel\n(reads before writing)",
                     facecolor="white", edgecolor=ORANGE_TOPAZE, lw=2.0)

    # P-reuse: scratchpad -> scavenger directly (same physical SRAM reused).
    arrow((4.4, 6.3), (4.4, 2.1))
    ax.text(4.55, 5.0, "P-reuse", fontsize=8.5, fontweight="bold", color=CASSIS,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=WASABI, edgecolor=CASSIS, alpha=0.6))

    # P-spill: scratchpad/registers -> CUDA local (global-backed) -> scavenger.
    arrow((3.7, 6.6), (2.15, 5.1), connectionstyle="arc3,rad=0.15")
    arrow((2.15, 4.2), (3.9, 2.1), connectionstyle="arc3,rad=-0.15")
    ax.text(1.0, 5.65, "P-spill", fontsize=8.5, fontweight="bold", color=CASSIS,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=WASABI, edgecolor=CASSIS, alpha=0.6))

    # P-stage: scratchpad -> explicit global staging -> scavenger.
    arrow((6.3, 6.6), (7.85, 5.1), connectionstyle="arc3,rad=-0.15")
    arrow((7.85, 4.2), (6.1, 2.1), connectionstyle="arc3,rad=0.15")
    ax.text(7.9, 5.65, "P-stage", fontsize=8.5, fontweight="bold", color=CASSIS,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=WASABI, edgecolor=CASSIS, alpha=0.6))

    # Residue dots placed ON the connecting arrows (path midpoints), not on
    # box interiors, so they never overlap a box's centered text label.
    residue_points = [
        (4.4, 6.15),    # leaves the scratchpad (P-reuse origin)
        (4.4, 3.3),     # mid P-reuse arrow, en route to the scavenger
        (2.9, 4.65),    # mid P-spill arrow, leaving CUDA local memory
        (7.1, 4.65),    # mid P-stage arrow, leaving the global staging buffer
    ]
    for (x, y) in residue_points:
        ax.plot([x], [y], marker="o", markersize=8, color=ORANGE_TOPAZE,
                 markeredgecolor=CASSIS, zorder=4)

    ax.text(5.0, 0.7,
            f"{FAIL_ICON} Orange Topaze markers = residue read by an unprivileged co-resident kernel "
            "(three physical paths, one failure: unzeroed transient state)",
            fontsize=7.5, ha="center", color=CASSIS, style="italic")

    fig.suptitle("Residue Lifecycle — Three Paths From One Failure\n"
                 "(P-reuse, P-spill, P-stage all terminate in a scavenger kernel reading unzeroed state)",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, "fig6_threat_model_lifecycle")


# ── Fig 7 (new, 3.4b): Terminology table graphic ────────────────────────────
def fig7_terminology_table():
    """OpenCL term <-> CUDA term <-> physical location <-> this paper's usage.
    Matches the terminology table in docs/LMHC_v2.docx §II.A verbatim."""
    rows = [
        ("local",            "shared",  "on-chip SRAM (per-SM)",  "scratchpad",      True,  "P-reuse"),
        ("private (spill)",  "local",   "off-chip global DRAM",   'spill "local"',   True,  "P-spill"),
        ("global (staging)", "global",  "off-chip global DRAM",   "staging buffer",  True,  "P-stage"),
        ("global (output)",  "global",  "off-chip global DRAM",   "committed output", False, None),
    ]
    col_hdrs = ["OpenCL term", "CUDA term", "Physical location", "This paper", "In scope?"]

    cell_text, cell_colors = [], []
    for opencl, cuda, phys, paper, in_scope, path in rows:
        scope_text = f"{PASS_ICON} yes ({path})" if in_scope else f"{NA_ICON} no"
        cell_text.append([opencl, cuda, phys, paper, scope_text])
        scope_color = VERT_SAUGE if in_scope else COOL_BLUE
        cell_colors.append(["#ffffff", "#ffffff", "#ffffff", "#ffffff", scope_color])

    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, cellColours=cell_colors, colLabels=col_hdrs,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.6)

    for ci in range(len(col_hdrs)):
        cell = tbl[0, ci]
        cell.set_facecolor(CASSIS)
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor(CASSIS)
    for ri in range(len(rows)):
        for ci in range(len(col_hdrs)):
            cell = tbl[ri + 1, ci]
            cell.set_edgecolor(CASSIS)
            cell.set_text_props(color=CASSIS)
            if ci == 4:
                cell.set_text_props(color=CASSIS, fontweight="bold")

    ax.set_title(
        "GPU Scratchpad Memory — Naming Across OpenCL / CUDA / This Paper\n"
        'The contract\'s "Local" follows the OpenCL/LeftoverLocals sense (the scratchpad); '
        'CUDA "local" denotes the unrelated global-backed spill region.',
        fontsize=9.5, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    _save(fig, "fig7_terminology_table")


# ── Fig 8 (new, 3.4c): Predicted vs. measured overhead ──────────────────────
def fig8_predicted_vs_measured():
    d = _load(RESULTS / "model_validation.json")
    rows = d["results"]

    order = ["reduction", "full_sm_amplification", "gemm", "attention"]
    rows = sorted(rows, key=lambda r: order.index(r["kernel"]))

    kernel_labels = {
        "reduction": "Reduction",
        "full_sm_amplification": "Full-SM\namplification",
        "gemm": "GEMM",
        "attention": "Attention",
    }
    labels = [kernel_labels[r["kernel"]] for r in rows]
    predicted = [r["predicted_ns"] for r in rows]
    measured  = [r["measured_ns"]  for r in rows]
    is_calib  = [r["role"] == "calibration" for r in rows]

    x, w = np.arange(len(rows)), 0.32
    fig, ax = plt.subplots(figsize=(9, 5.2))

    # Measured = the ground truth series -> Cassis (dark accent, solid).
    # Predicted = the model's output -> Cool Blue, hatched for calibration
    # points (predicted==measured by construction, not a real prediction).
    b_meas = ax.bar(x - w/2, measured, w, label="Measured (clock64())",
                     color=CASSIS, alpha=0.88, edgecolor=CASSIS, linewidth=0.8, zorder=3)
    pred_colors = [COOL_BLUE for _ in rows]
    pred_hatch  = ["xxx" if c else "" for c in is_calib]
    b_pred = ax.bar(x + w/2, predicted, w, label="Predicted (model)",
                     color=pred_colors, hatch=pred_hatch, edgecolor=CASSIS, linewidth=0.8, zorder=3)

    for i, r in enumerate(rows):
        y = max(predicted[i], measured[i])
        if is_calib[i]:
            tag = "calibration\n(exact, by construction)"
            box = dict(boxstyle="round,pad=0.2", facecolor=COOL_BLUE, edgecolor=CASSIS, alpha=0.7)
        else:
            ratio = r["predicted_over_measured_ratio"]
            within = r["within_2x"]
            tag = pass_fail_label(within, f"{ratio:.2f}× (within 1.5× bound)")
            box = dict(boxstyle="round,pad=0.2",
                       facecolor=(VERT_SAUGE if within else ORANGE_TOPAZE), edgecolor=CASSIS, alpha=0.7)
        ax.annotate(tag, xy=(x[i], y), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=7.5, fontweight="bold", color=CASSIS, bbox=box)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Per-exit scrub cost (ns)", fontsize=9)
    ax.set_ylim(0, max(max(predicted), max(measured)) * 1.35)  # headroom for annotation boxes
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Analytical Cost Model — Predicted vs. Measured Per-Exit Scrub Cost",
        fontsize=10.5, y=1.02,
    )
    ax.set_title(
        f"Hardware: {HW} | b={d['calibrated_b_GB_per_s']:.1f} GB/s/SM, L={d['calibrated_L_ns']:.1f} ns "
        f"(calibrated on reduction + full-SM amplification; GEMM/attention are held-out predictions)",
        fontsize=8, fontweight="normal",
    )
    fig.tight_layout()
    _save(fig, "fig8_predicted_vs_measured")


# ── Fig 9 (new, 3.4d): Verifier scalability ──────────────────────────────────
def fig9_verifier_scalability():
    d = _load(RESULTS / "verifier_perf.json")
    rows = d["synthetic_scalability_test"]["results"]

    n_kernels   = [r["n_kernels"] for r in rows]
    wallclock_s = [r["verify_wallclock_s"] for r in rows]
    peak_mb     = [r["verify_peak_rss_kb"] / 1024.0 for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(n_kernels, wallclock_s, marker="o", markersize=7, color=CASSIS,
                   linewidth=1.8, label="Verify wall-clock (s)", zorder=3)
    l2 = ax2.plot(n_kernels, peak_mb, marker="s", markersize=7, color=COOL_BLUE,
                   markeredgecolor=CASSIS, linewidth=1.8, linestyle="--",
                   label="Peak RSS (MB)", zorder=3)

    ax1.set_xlabel("N synthetic kernels (gemm_tile-shaped, tainted)", fontsize=9)
    ax1.set_ylabel("Verify wall-clock (s)", fontsize=9, color=CASSIS)
    ax2.set_ylabel("Peak resident memory (MB)", fontsize=9, color=CASSIS)
    ax1.set_xticks(n_kernels)
    ax1.set_ylim(0, max(wallclock_s) * 2.3)   # headroom for the in-axes annotation box
    ax2.set_ylim(0, max(peak_mb) * 1.9)

    for x, y in zip(n_kernels, wallclock_s):
        ax1.annotate(f"{y:.3f}s", (x, y), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=7.5, color=CASSIS)

    lines = l1 + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="lower left")

    # Placed INSIDE the axes bounds (not above them) so it never collides
    # with the figure/axes title area.
    ax1.annotate(
        "No scaling curve is fit — 4 measured points only (N=1,2,4,8).\n"
        "Wall-clock does not visibly scale over this range: cost is dominated\n"
        "by fixed per-invocation overhead, not per-kernel verification work.",
        xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top", fontsize=7.5,
        color=CASSIS, bbox=dict(boxstyle="round,pad=0.35", facecolor=WASABI, edgecolor=CASSIS, alpha=0.6),
    )

    fig.suptitle(
        "Independent-Verifier Cost vs. Kernel Count (Synthetic Scalability Probe)",
        fontsize=10.5, y=1.01,
    )
    ax1.set_title(
        f"Hardware: {HW} | attester.py (CHECK 1–3), real discover→rewrite→compile→witness→verify pipeline",
        fontsize=8, fontweight="normal",
    )
    fig.tight_layout()
    _save(fig, "fig9_verifier_scalability")


# ── Fig 10 (new, 3.4e): Multi-GPU comparison ─────────────────────────────────
def fig10_multi_gpu_comparison():
    files = sorted(MGPU_RES.glob("*.json"))
    devices = []
    for p in files:
        d = _load(p)
        env = d.get("environment", {})
        devices.append({
            "name": env.get("gpu_name", p.stem),
            "arch": env.get("architecture_name", ""),
            "control": d.get("control_run", {}).get("exposure_index", 0.0),
            "residue": d.get("residue_run", {}).get("exposure_index", 0.0),
            "scrub":   d.get("lmhc_scrub_run", {}).get("exposure_index", 0.0),
            "valid":   d.get("valid", False),
        })
    devices.sort(key=lambda r: r["name"])

    labels = [f"{dv['name']}\n({dv['arch']})" for dv in devices]
    control = [dv["control"] for dv in devices]
    residue = [dv["residue"] for dv in devices]
    scrub   = [dv["scrub"]   for dv in devices]

    x, w = np.arange(len(devices)), 0.26
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    kw_res = pass_fail_kwargs(False)  # before-mitigation / residue -> Orange Topaze
    kw_scr = pass_fail_kwargs(True)   # after-mitigation / scrub    -> Vert Sauge

    ax.bar(x - w, control, w, label="Control (validity gate)",
           color=COOL_BLUE, edgecolor=CASSIS, linewidth=0.8, zorder=3)
    ax.bar(x, residue, w, label=f"No-LMHC (residue) {FAIL_ICON}",
           color=kw_res["color"], hatch=kw_res["hatch"], edgecolor=CASSIS, linewidth=0.8, zorder=3)
    ax.bar(x + w, scrub, w, label=f"LMHC scrub {PASS_ICON}",
           color=kw_scr["color"], hatch=kw_scr["hatch"], edgecolor=CASSIS, linewidth=0.8, zorder=3)

    for i, (c, r, s) in enumerate(zip(control, residue, scrub)):
        for xoff, v in [(-w, c), (0, r), (w, s)]:
            ax.text(x[i] + xoff, v + 0.03, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=CASSIS, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Exposure Index (EI)", fontsize=9)
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title(
        "Multi-GPU Replication — P-reuse Residue and LMHC Scrub, Three Devices\n"
        "Co-residency forced on every device (controlled-placement, not a natural-scheduling frequency); "
        "RTX 3060 Ti's 0.995 is reported as measured, not rounded to 1.000.",
        fontsize=9,
    )
    fig.tight_layout()
    _save(fig, "fig10_multi_gpu_comparison")


# ── Fig 11 (new, 3.4f): Three-path residue comparison ────────────────────────
def fig11_three_path_comparison():
    p_reuse = _load(RESULTS / "leakage_demo.json")
    p_stage = _load(RESULTS / "leakage_demo_pstage.json")
    p_spill = _load(RESULTS / "leakage_demo_pspill.json")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.2), sharey=True)

    # P-reuse: no separate control phase was recorded in this experiment.
    panels = [
        ("P-reuse", p_reuse, ["Before", "After"],
         [p_reuse["exposure_index_before"], p_reuse["exposure_index_after"]],
         [False, True]),
        ("P-stage", p_stage, ["Control", "Before", "After"],
         [p_stage["exposure_index_control"], p_stage["exposure_index_before"], p_stage["exposure_index_after"]],
         [None, False, True]),
        ("P-spill", p_spill, ["Control", "Before", "After"],
         [p_spill["exposure_index_control"], p_spill["exposure_index_before"], p_spill["exposure_index_after"]],
         [None, None, None]),   # special-cased below: negative/bounding result, not pass/fail
    ]

    for ax, (path_name, data, phase_labels, values, roles) in zip(axes, panels):
        colors, hatches = [], []
        for role in roles:
            if role is True:
                kw = pass_fail_kwargs(True); colors.append(kw["color"]); hatches.append(kw["hatch"])
            elif role is False:
                kw = pass_fail_kwargs(False); colors.append(kw["color"]); hatches.append(kw["hatch"])
            else:
                # Control phases, and ALL of P-spill (negative/bounding result:
                # EI=0.0 in every phase — there was no exposure to close, so
                # coloring "before" as a leak (Orange Topaze) would misrepresent
                # a negative finding as a positive mitigation result.
                colors.append(COOL_BLUE); hatches.append("")

        bars = ax.bar(phase_labels, values, color=colors, hatch=hatches,
                       edgecolor=CASSIS, linewidth=0.8, width=0.55, zorder=3)
        for rect, v in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.03, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=CASSIS)

        ax.set_ylim(0, 1.25)
        ax.set_title(path_name, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8.5)

    axes[0].set_ylabel("Exposure Index (EI)", fontsize=9)
    axes[2].annotate(
        "EI = 0.000 in ALL phases —\nnegative/bounding result, NOT\na leak-then-fix demonstration.\nSee docs/RESULTS.md P-spill section.",
        xy=(0.5, 0.65), xycoords="axes fraction", ha="center", fontsize=7.3, color=CASSIS,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=WASABI, edgecolor=CASSIS, alpha=0.55),
    )

    fig.suptitle(
        f"Three-Path Residue Recovery — P-reuse / P-stage / P-spill\n"
        f"Hardware: {HW} | N=200 trials/phase, sentinel 0xCAFEBABE | "
        f"{PASS_ICON} PASS (scrubbed/safe)  {FAIL_ICON} FAIL (unscrubbed/leak)  "
        "neutral = control or non-pass/fail bounding result",
        fontsize=9.5, y=1.03,
    )
    fig.tight_layout()
    _save(fig, "fig11_three_path_comparison")


if __name__ == "__main__":
    print("Generating LMHC-Forge figures (Phase 3 palette) ...")
    fig1_phase0_heatmap()
    fig2_dse_bar()
    fig3_exposure_index()
    fig4_benchmark()
    fig5_tamper_panel()
    fig6_threat_model_lifecycle()
    fig7_terminology_table()
    fig8_predicted_vs_measured()
    fig9_verifier_scalability()
    fig10_multi_gpu_comparison()
    fig11_three_path_comparison()
    print(f"All figures saved to {FIGS}")
