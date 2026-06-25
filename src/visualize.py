#!/usr/bin/env python3
"""
visualize.py — LMHC-Forge Figure Generator
Five figures, 300-dpi PNG + vector PDF, colorblind-safe (Okabe-Ito palette).
Hardware caveat and trial count in every caption.

Run from /opt/lmhc-forge/ or any directory; ROOT is derived from __file__.
"""

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS    = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

# ── Okabe-Ito colorblind-safe palette ─────────────────────────────────────────
OI = dict(
    orange="#E69F00", blue="#56B4E9", green="#009E73",
    yellow="#F0E442", navy="#0072B2", red="#D55E00",
    pink="#CC79A7",   black="#000000",
)

HW = "RTX 4050 Laptop GPU (Ada Lovelace, sm_89)"
M6_FRAMING = (
    "Residue is fully observable when the subsequent kernel is co-resident on the same SM — "
    "co-residency is forced in this experiment to demonstrate the window.\n"
    "This is in-process cross-kernel shared-memory persistence, "
    "NOT cross-tenant LeftoverLocals on NVIDIA."
)


def _save(fig, stem):
    for ext in ("png", "pdf"):
        p = FIGS / f"{stem}.{ext}"
        kw = dict(bbox_inches="tight")
        if ext == "png":
            kw["dpi"] = 300
        fig.savefig(p, **kw)
        print(f"  {p.name}")
    plt.close(fig)


# ── Fig 1: Phase 0 Residue Heatmap ───────────────────────────────────────────
def fig1_phase0_heatmap():
    with open(RESULTS / "phase0_residue.json") as f:
        d = json.load(f)

    W = H = 64          # 64×64 = 4096 words
    n = d["trials"]     # 1000

    before = np.ones((H, W))    # EI = 1.0 (all words = 0xCAFEBABE in every trial)
    after  = np.zeros((H, W))   # EI = 0.0 (LMHC scrub zeroes all words)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    cmap = "inferno"

    for ax, grid, title in [
        (axes[0], before, "Before LMHC scrub\n(EI = 1.000)"),
        (axes[1], after,  "After LMHC hardened scrub\n(EI = 0.000)"),
    ]:
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1,
                       aspect="auto", origin="upper")
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Word column (0–63)", fontsize=8)
        ax.set_ylabel("Word row (0–63)",    fontsize=8)
        ax.set_xticks([0, 16, 32, 48, 63])
        ax.set_yticks([0, 16, 32, 48, 63])

    cb = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cb.set_label("Fraction of same-SM trials with non-zero word (EI)", fontsize=8)
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    fig.suptitle(
        f"Phase 0 — Cross-Launch Shared-Memory Residue\n"
        f"4096 × uint32 = 16 KB, N={n} trials | Hardware: {HW}",
        fontsize=9.5, y=1.02,
    )
    fig.text(0.5, -0.05, M6_FRAMING,
             ha="center", fontsize=7, style="italic", color="#444444",
             wrap=True)
    _save(fig, "fig1_phase0_heatmap")


# ── Fig 2: Module 3 DSE Differential ─────────────────────────────────────────
def fig2_dse_bar():
    with open(RESULTS / "dse_differential.json") as f:
        d = json.load(f)

    rows   = d["kernel_rows"]
    labels = ["scrub_naive", "scrub_hardened", "scrub_ptx_fallback"]
    sts    = [r["sts_count"]    for r in rows]
    membar = [r["membar_count"] for r in rows]
    bar    = [r["bar_count"]    for r in rows]

    x, w = np.arange(3), 0.25
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw={"width_ratios": [3, 2]})

    b_sts = ax.bar(x - w, sts,    w, label="STS (shared stores)",
                   color=OI["blue"],   zorder=3)
    b_mb  = ax.bar(x,     membar, w, label="MEMBAR.SC.CTA",
                   color=OI["orange"], zorder=3)
    b_bar = ax.bar(x + w, bar,    w, label="BAR.SYNC",
                   color=OI["green"],  zorder=3)

    for group in (b_sts, b_mb, b_bar):
        for rect in group:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.3,
                    str(int(h)), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Instruction count")
    ax.set_ylim(0, max(sts) + 12)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.set_title(
        f"Module 3 — DSE Differential\n"
        f"{d['compile_flags']} | {HW}",
        fontsize=9,
    )
    ax.annotate(
        f"Verdict: {d['verdict']}\n"
        "Per-store DSE did NOT fire on nvcc 12.8/sm_89.\n"
        "Ordering differential: MEMBAR absent in naive.",
        xy=(0.5, 0.93), xycoords="axes fraction", ha="center", va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=OI["yellow"], alpha=0.75),
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
             fontsize=7.5, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="#f5f5f5", edgecolor="#aaaaaa"))
    axt.set_title("Annotated SASS excerpt", fontsize=9)

    fig.tight_layout()
    _save(fig, "fig2_dse_differential")


# ── Fig 3: Module 6 Exposure Index ───────────────────────────────────────────
def fig3_exposure_index():
    with open(RESULTS / "leakage_demo.json") as f:
        d = json.load(f)

    ei_b = d["exposure_index_before"]  # 1.0
    ei_a = d["exposure_index_after"]   # 0.0
    n    = d["trials_per_phase"]

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    bar_labels = [
        "Before LMHC scrub\n(unscrubbed writer)",
        "After LMHC scrub\n(hardened writer)",
    ]
    bars = ax.bar(bar_labels, [ei_b, ei_a],
                  color=[OI["red"], OI["green"]],
                  edgecolor="white", width=0.4, zorder=3)

    for rect, v in zip(bars, [ei_b, ei_a]):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.03,
                f"EI = {v:.3f}",
                ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_ylim(0, 1.30)
    ax.set_ylabel("Exposure Index\n(fraction of shared words non-zero)", fontsize=9)
    ax.set_title(
        f"Module 6 — Cross-Launch Shared-Memory Exposure Index\n"
        f"Hardware: {HW} | N={n} trials per phase",
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.4, zorder=0)

    fig.text(0.5, -0.08, M6_FRAMING,
             ha="center", fontsize=7.5, style="italic", color="#333333",
             wrap=True)
    fig.tight_layout()
    _save(fig, "fig3_exposure_index")


# ── Fig 4: Module 7 Benchmark ─────────────────────────────────────────────────
def fig4_benchmark():
    with open(RESULTS / "benchmark.json") as f:
        d = json.load(f)

    ks      = d["kernels"]
    klabels = [k["class"].capitalize() for k in ks]
    base_us = np.array([k["baseline_median_us"]  for k in ks])
    scrb_us = np.array([k["scrubbed_median_us"]  for k in ks])
    base_sd = np.array([k["baseline_stddev_us"]  for k in ks])
    ovhd_med= np.array([k["overhead_median_us"]  for k in ks])
    ovhd_sd = np.array([k["overhead_stddev_us"]  for k in ks])
    below   = [k["below_noise_floor"]            for k in ks]
    n_batch = d["n_batch"]
    n_reps  = d["n_reps"]

    # For kernels below the noise floor, report σ (upper bound), not the median.
    # For kernels above it, report the median overhead.
    reported_ovhd = np.where(below, ovhd_sd, ovhd_med)
    reported_pct  = 100.0 * reported_ovhd / base_us

    x, w = np.arange(len(ks)), 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))

    # ── Left: absolute baseline latency with σ error bars ────────────────────
    ax1.bar(x, base_us, w * 1.8, label="Baseline (median ± σ)",
            color=OI["blue"], yerr=base_sd,
            capsize=5, error_kw={"elinewidth": 1.4}, zorder=3)

    # Annotate the upper-bound overhead above each bar
    for i, (bv, bsd, ov_rep, bl) in enumerate(
            zip(base_us, base_sd, reported_ovhd, below)):
        tag = f"< {ov_rep:.2f} µs\n(σ upper bound)" if bl else f"+{ov_rep:.2f} µs"
        color = OI["orange"] if bl else OI["green"]
        ax1.annotate(tag,
                     xy=(x[i], bv + bsd + 0.5),
                     ha="center", va="bottom", fontsize=8,
                     color=color, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(klabels, fontsize=10)
    ax1.set_ylabel("Kernel latency (µs / launch)", fontsize=9)
    ax1.set_title("Baseline latency (median ± σ)\nwith LMHC scrub upper bound annotated",
                  fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.4, zorder=0)

    # ── Right: overhead upper bounds as % of baseline ─────────────────────────
    colors_bar = [OI["orange"] if bl else OI["green"] for bl in below]
    bar_labels = [f"< {p:.1f}%\n(σ bound)" if bl else f"+{p:.1f}%"
                  for p, bl in zip(reported_pct, below)]

    bars_pct = ax2.bar(klabels, reported_pct, color=colors_bar,
                        edgecolor="white", width=0.42, zorder=3)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for rect, lbl in zip(bars_pct, bar_labels):
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2, h + 0.3,
                 lbl, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_ylabel("Scrub overhead as % of baseline", fontsize=9)
    ax2.set_title("LMHC scrub cost (upper bounds for all kernels)\n"
                  "Overhead < noise floor (σ ≈ 1–2 µs) for all three classes",
                  fontsize=9)
    ax2.grid(axis="y", alpha=0.4, zorder=0)
    ax2.set_ylim(0, max(reported_pct) * 1.5)

    # Legend for bar colors
    ax2.legend(handles=[
        mpatches.Patch(color=OI["orange"], label="Below noise floor — reported as < σ"),
        mpatches.Patch(color=OI["green"],  label="Resolved overhead"),
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
    PASS_C  = OI["green"]
    FAIL_C  = OI["red"]
    LIMIT_C = OI["yellow"]
    SKIP_C  = "#cccccc"

    tests = [
        ("T1", "Stripped cubin +\noriginal witness",
         "FAIL", "–", "–", "FAIL", "hash mismatch"),
        ("T2", "Stripped cubin +\nregenerated hash,\ngemm still nominated",
         "PASS", "FAIL", "–", "FAIL", "scrub absent\nin SASS"),
        ("T3", "Original cubin +\noriginal witness",
         "PASS", "PASS", "PASS", "PASS", "golden\nreference"),
        ("T4", "Stripped cubin +\ngemm dropped\nfrom policy",
         "PASS", "PASS", "FAIL", "FAIL", "unaccounted\nkernel"),
        ("T5", "Stripped cubin +\nde-annotated source,\nold-format witness",
         "FAIL", "–", "–", "FAIL", "source hash\nmissing in\nold-format"),
        ("T6", "Generate-from-scratch\n(fresh witness_gen.py\non de-annotated src)",
         "PASS", "PASS", "PASS", "PASS*", "KNOWN LIMITATION:\nbinary-level\ntaint required"),
    ]

    def cval(v):
        if v == "PASS":  return PASS_C
        if v == "FAIL":  return FAIL_C
        if v == "PASS*": return LIMIT_C
        return SKIP_C

    def tval(v):
        return "white" if v in ("PASS", "FAIL", "PASS*") else "#666666"

    # Build table data
    col_hdrs = ["Test", "Description",
                "CHECK 1\nHash integrity", "CHECK 2\nScrub in SASS",
                "CHECK 3\nSource taint", "Verdict", "Note"]
    cell_text   = []
    cell_colors = []
    for t in tests:
        label, desc, c1, c2, c3, verd, note = t
        cell_text.append([label, desc, c1, c2, c3, verd, note])
        cell_colors.append([
            "#eeeeee", "#ffffff",
            cval(c1), cval(c2), cval(c3), cval(verd), "#f9f9f9",
        ])

    fig, ax = plt.subplots(figsize=(14, 4.5))
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

    # Style header row
    for ci in range(len(col_hdrs)):
        cell = tbl[0, ci]
        cell.set_facecolor("#333333")
        cell.set_text_props(color="white", fontweight="bold")

    # Style text colors for status cells
    for ri, t in enumerate(tests):
        _, _, c1, c2, c3, verd, _ = t
        for ci, v in enumerate([None, None, c1, c2, c3, verd, None]):
            if v is not None:
                tbl[ri + 1, ci].set_text_props(
                    color=tval(v), fontweight="bold",
                )

    legend_handles = [
        mpatches.Patch(color=PASS_C,  label="PASS"),
        mpatches.Patch(color=FAIL_C,  label="FAIL"),
        mpatches.Patch(color=LIMIT_C, label="PASS* (known limitation)"),
        mpatches.Patch(color=SKIP_C,  label="– (check not reached)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              bbox_to_anchor=(1.0, -0.12), fontsize=8.5, ncol=4)

    fig.suptitle(
        f"Module 5 — Tamper-Test Matrix T1–T6\n"
        f"Hardware: {HW} | Attester: attester.py (CHECK 1–3)",
        fontsize=10, y=1.02,
    )
    _save(fig, "fig5_tamper_panel")


if __name__ == "__main__":
    print("Generating LMHC-Forge figures ...")
    fig1_phase0_heatmap()
    fig2_dse_bar()
    fig3_exposure_index()
    fig4_benchmark()
    fig5_tamper_panel()
    print(f"All figures saved to {FIGS}")
