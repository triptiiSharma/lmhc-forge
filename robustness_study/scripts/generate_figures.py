#!/usr/bin/env python3
"""
generate_figures.py — the 4 paper-ready figures requested for the final
manuscript revision. Every figure is built from data queried fresh from
the frozen results files (never hand-entered), and each writes a PDF, a
PNG, and the exact source CSV used to build it.
"""

import csv
import json
from pathlib import Path
from collections import defaultdict, Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colorblind-safe categorical palette (Okabe-Ito), one color per outcome category.
COLORS = {
    "FULL_SIGNATURE": "#0072B2",
    "PARTIAL_SIGNATURE": "#E69F00",
    "STORE_ONLY": "#CC79A7",
    "ABSENT": "#999999",
    "UNDECIDABLE": "#D55E00",
}
CATEGORY_ORDER = ["FULL_SIGNATURE", "PARTIAL_SIGNATURE", "STORE_ONLY", "ABSENT", "UNDECIDABLE"]


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def save_fig(fig, name):
    pdf_path = FIG_DIR / f"{name}.pdf"
    png_path = FIG_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf_path}, {png_path}")


# ─────────────────────────────────────────────────────────────────────────
# Figure 1 — end-to-end pipeline schematic
# ─────────────────────────────────────────────────────────────────────────
def figure1():
    stages = [
        ("Source\nvariant", "kernels/generated/*.cu\n(7 constructs x 6 contexts)"),
        ("Compilation\nmatrix", "nvcc + ptxas\n2 toolkits x 4 opt x 6 arch"),
        ("cubin", "compiled GPU\nbinary"),
        ("cuobjdump /\nnvdisasm", "two independent\ndisassembly paths"),
        ("Feature\nextraction", "zero-store, fence,\nbarrier, exit counts"),
        ("Structural\ncategory", "FULL_SIGNATURE /\nPARTIAL / STORE_ONLY /\nABSENT / UNDECIDABLE"),
        ("Deployment\ninterpretation", "static evidence only;\nnot a dynamic-path proof"),
    ]

    csv_path = FIG_DIR / "fig1_pipeline_source.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage_index", "stage_name", "stage_detail"])
        for i, (name, detail) in enumerate(stages):
            w.writerow([i, name.replace("\n", " "), detail.replace("\n", " ")])

    fig, ax = plt.subplots(figsize=(14, 2.6))
    n = len(stages)
    box_w, box_h = 1.6, 1.3
    gap = 0.55
    x = 0
    for i, (name, detail) in enumerate(stages):
        box = FancyBboxPatch((x, 0), box_w, box_h, boxstyle="round,pad=0.06,rounding_size=0.08",
                              linewidth=1.2, edgecolor="#333333", facecolor="#EAF1FB")
        ax.add_patch(box)
        ax.text(x + box_w / 2, box_h * 0.68, name, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(x + box_w / 2, box_h * 0.28, detail, ha="center", va="center", fontsize=6.6, color="#444444")
        if i < n - 1:
            arrow = FancyArrowPatch((x + box_w, box_h / 2), (x + box_w + gap, box_h / 2),
                                     arrowstyle="-|>", mutation_scale=14, color="#333333", linewidth=1.2)
            ax.add_patch(arrow)
        x += box_w + gap

    ax.set_xlim(-0.2, x)
    ax.set_ylim(-0.2, box_h + 0.2)
    ax.axis("off")
    ax.set_title("Fig. 1. End-to-end compilation, analysis, and interpretation pipeline", fontsize=10)
    save_fig(fig, "fig1_pipeline")


# ─────────────────────────────────────────────────────────────────────────
# Figure 2 — construct x context structural outcome matrix
# ─────────────────────────────────────────────────────────────────────────
def figure2():
    rows = list(csv.DictReader(open(ROOT / "results" / "classifications.csv")))
    constructs = [f"Z{i}" for i in range(7)]
    contexts = [f"K{i}" for i in range(6)]

    combo = defaultdict(Counter)
    for r in rows:
        z, k, v = r["zeroization_construct"], r["kernel_context"], r["checker_verdict"]
        if z in constructs and k in contexts:
            combo[(z, k)][v] += 1

    csv_path = FIG_DIR / "fig2_construct_context_source.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["construct", "context", "dominant_category", "count", "total_cells"])
        for z in constructs:
            for k in contexts:
                c = combo[(z, k)]
                if c:
                    dom, cnt = c.most_common(1)[0]
                    w.writerow([z, k, dom, cnt, sum(c.values())])

    # Build a category-index matrix for imshow, plus annotate counts.
    cat_index = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    mat = [[None] * len(contexts) for _ in constructs]
    for zi, z in enumerate(constructs):
        for ki, k in enumerate(contexts):
            c = combo[(z, k)]
            if c:
                dom, _ = c.most_common(1)[0]
                mat[zi][ki] = cat_index[dom]

    import numpy as np
    arr = np.array([[v if v is not None else -1 for v in row] for row in mat], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap_colors = [COLORS[c] for c in CATEGORY_ORDER]
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(cmap_colors)
    norm = BoundaryNorm(list(range(len(CATEGORY_ORDER) + 1)), cmap.N)
    im = ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels(contexts)
    ax.set_yticks(range(len(constructs)))
    ax.set_yticklabels(constructs)
    ax.set_xlabel("Kernel context")
    ax.set_ylabel("Zeroization construct")
    ax.set_title("Fig. 2. Structural outcome by construct x context\n"
                  "(aggregated across opt level, architecture, toolkit; 48 cells per square)", fontsize=9.5)

    for zi in range(len(constructs)):
        for ki in range(len(contexts)):
            c = combo[(constructs[zi], contexts[ki])]
            if c:
                dom, cnt = c.most_common(1)[0]
                total = sum(c.values())
                label = f"{cnt}/{total}" if cnt != total else f"{total}/{total}"
                ax.text(ki, zi, label, ha="center", va="center", fontsize=7,
                         color="white" if dom in ("FULL_SIGNATURE", "ABSENT") else "black")

    legend_patches = [mpatches.Patch(color=COLORS[c], label=c) for c in CATEGORY_ORDER if c != "UNDECIDABLE"]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
    save_fig(fig, "fig2_construct_context_matrix")


# ─────────────────────────────────────────────────────────────────────────
# Figure 3 — outcome distribution across the full corpus, with breakdowns
# ─────────────────────────────────────────────────────────────────────────
def figure3():
    rows = list(csv.DictReader(open(ROOT / "results" / "classifications.csv")))

    def dist_by(key):
        d = defaultdict(Counter)
        for r in rows:
            d[r[key]][r["checker_verdict"]] += 1
        return d

    by_construct = dist_by("zeroization_construct")
    by_opt = dist_by("ptxas_optimization")
    by_arch = dist_by("target_arch")
    by_toolkit = dist_by("cuda_version")

    csv_path = FIG_DIR / "fig3_outcome_distribution_source.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dimension", "value", "category", "count"])
        for dim_name, d in [("construct", by_construct), ("opt_level", by_opt),
                             ("architecture", by_arch), ("toolkit", by_toolkit)]:
            for val, counter in sorted(d.items(), key=lambda x: str(x[0])):
                for cat in CATEGORY_ORDER:
                    if counter.get(cat):
                        w.writerow([dim_name, val, cat, counter[cat]])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        (axes[0, 0], by_construct, [f"Z{i}" for i in range(7)], "Zeroization construct"),
        (axes[0, 1], by_opt, ["0", "1", "2", "3"], "ptxas optimization level"),
        (axes[1, 0], by_arch, ["sm_75", "sm_80", "sm_86", "sm_87", "sm_89", "sm_90"], "Architecture target"),
        (axes[1, 1], by_toolkit, ["cuda-12.8", "cuda-13.3"], "CUDA toolkit"),
    ]
    for ax, dist, order, title in panels:
        bottom = [0] * len(order)
        for cat in CATEGORY_ORDER:
            vals = [dist[str(o)].get(cat, dist.get(o, {}).get(cat, 0)) for o in order]
            # keys in classifications.csv for opt level are strings like "0","1"...
            vals = [dist.get(str(o), Counter()).get(cat, 0) for o in order]
            ax.bar(range(len(order)), vals, bottom=bottom, color=COLORS[cat], label=cat, width=0.65)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=0 if title != "Architecture target" else 30, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Artifact count")

    handles = [mpatches.Patch(color=COLORS[c], label=c) for c in CATEGORY_ORDER if c != "UNDECIDABLE"]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)
    fig.suptitle("Fig. 3. Structural outcome distribution across the 2,016-artifact core corpus", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_fig(fig, "fig3_outcome_distribution")


# ─────────────────────────────────────────────────────────────────────────
# Figure 4 — N6 boundary case
# ─────────────────────────────────────────────────────────────────────────
def figure4():
    controlled = load_jsonl(ROOT / "results" / "controlled_cases.jsonl")
    n6 = next(r for r in controlled if r["case_id"] == "N6")
    feat = n6["cuobjdump_features"]

    csv_path = FIG_DIR / "fig4_n6_source.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["field", "value"])
        for k in ("case_id", "intended_category", "cuobjdump_verdict", "nvdisasm_verdict",
                   "cross_tool_agreement"):
            w.writerow([k, n6.get(k)])
        for k in ("exit_count", "last_zero_store_index", "first_membar_cta_index",
                   "last_bar_sync_index", "zero_shared_store_indirect_count"):
            w.writerow([k, feat.get(k)])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")

    # Left: source-level control flow (2 logical paths)
    ax.text(0.02, 0.95, "Source (CUDA)", fontsize=10, fontweight="bold", transform=ax.transAxes)
    src_lines = [
        "if (tid < 16) return;   // 16 of 32 threads exit HERE",
        "// -- cleanup below is source-level UNCOVERED for tid<16 --",
        "volatile uint32_t *v = sbuf;",
        "for (...) v[i] = 0u;      // zero-store",
        "__threadfence_block();     // fence",
        "__syncthreads();           // barrier",
    ]
    for i, line in enumerate(src_lines):
        color = "#CC0000" if i < 2 else "#006600"
        ax.text(0.02, 0.86 - i * 0.075, line, fontsize=8.5, family="monospace",
                 color=color, transform=ax.transAxes)

    ax.annotate("", xy=(0.50, 0.55), xytext=(0.40, 0.55), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black"))
    ax.text(0.45, 0.60, "ptxas\n(predication)", fontsize=8, ha="center", transform=ax.transAxes)

    ax.text(0.55, 0.95, "Compiled SASS (both toolkits, all 8 configs)", fontsize=10,
             fontweight="bold", transform=ax.transAxes)
    sass_lines = [
        f"...predicated skip for tid<16 (no separate EXIT)...",
        f"STS [...], RZ     <- zero-store  @ line {feat.get('last_zero_store_index')}",
        f"MEMBAR.SC.CTA     <- fence        @ line {feat.get('first_membar_cta_index')}",
        f"BAR.SYNC          <- barrier      @ line {feat.get('last_bar_sync_index')}",
        f"EXIT              <- ONLY exit, count={feat.get('exit_count')}",
    ]
    for i, line in enumerate(sass_lines):
        ax.text(0.55, 0.86 - i * 0.075, line, fontsize=8.5, family="monospace",
                 color="#333333", transform=ax.transAxes)

    ax.text(0.55, 0.40,
            f"Checker verdict: {n6['cuobjdump_verdict']}  (cross-tool agreement: "
            f"{n6['cross_tool_agreement']})",
            fontsize=9.5, fontweight="bold", color=COLORS["FULL_SIGNATURE"], transform=ax.transAxes)

    ax.text(0.02, 0.30,
            "Interpretation: the checker correctly reports the recognizable static\n"
            "signature present before the single EXIT instruction in the disassembly.\n"
            "It does not, and by its documented scope cannot, determine that the\n"
            "16 threads which returned early never reached this sequence.\n\n"
            "This is direct, compiled, cross-tool-agreed evidence that a\n"
            "FULL_SIGNATURE verdict establishes static structural presence and\n"
            "ordering -- not path-complete dynamic execution.",
            fontsize=9, transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0", edgecolor="#E69F00"))

    ax.set_title("Fig. 4. Controlled case N6: a source-level uncovered cleanup path "
                  "compiled to a single-EXIT binary via predication", fontsize=10)
    save_fig(fig, "fig4_n6_boundary_case")


if __name__ == "__main__":
    print("Figure 1...")
    figure1()
    print("Figure 2...")
    figure2()
    print("Figure 3...")
    figure3()
    print("Figure 4...")
    figure4()
    print("\nAll figures generated.")
