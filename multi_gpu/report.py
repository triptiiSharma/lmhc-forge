#!/usr/bin/env python3
"""report.py — LMHC-Forge per-device visual report

Reads one multi_gpu_repro JSON and emits a self-contained HTML report.
If matplotlib is available an inline PNG bar chart is embedded; otherwise
a CSS bar chart is used. No external dependencies beyond stdlib are required.

Usage:
    python3 report.py <result.json> [--output <report.html>]

Depicts ONLY values present in the JSON. No cross-device aggregates,
leakage rates, prevalence claims, or derived metrics are computed or shown.
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

# Mandatory caption — must appear on every report.
CAPTION = (
    "Controlled forced-co-residency, in-process measurement on the named device. "
    "Not a natural-scheduling frequency or a claim about untested hardware."
)

FRAMING = (
    "Co-residency is forced via __launch_bounds__(32,1) + %smid verification. "
    "This is in-process cross-kernel shared-memory persistence, NOT cross-tenant "
    "LeftoverLocals on NVIDIA. These are controlled-placement results consistent "
    "with the adversarial threat model — not natural-scheduling frequencies, and "
    "not a claim about production prevalence or about non-tested hardware."
)


# ── Chart generation ──────────────────────────────────────────────────────────

def _try_matplotlib_chart(valid: bool, gpu_name: str,
                          ei_control: float, ei_residue: float,
                          ei_scrub: float) -> bytes | None:
    """Return PNG bytes using matplotlib, or None if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    labels = ["Control\n(validity gate)", "Residue\n(no scrub)", "LMHC\nscrub"]
    values = [ei_control, ei_residue, ei_scrub]
    # Green for control (should be 0), red-orange for residue, steel-blue for scrub.
    colors = ["#43a047", "#e53935", "#1e88e5"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors, width=0.45,
                  edgecolor="#333", linewidth=0.7)

    for bar, val in zip(bars, values):
        label = f"{val:.4f}"
        if not valid:
            label += "\n(invalid)"
        y = bar.get_height() + 0.02
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                label, ha="center", va="bottom", fontsize=9, color="#222")

    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Exposure Index (EI)", fontsize=10)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])

    title = f"Exposure Index by Phase\n{gpu_name}"
    if not valid:
        title += "\n⚠ INVALID MEASUREMENT — control non-zero"
    ax.set_title(title, fontsize=10, pad=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if not valid:
        ax.text(0.5, 0.5, "INVALID\nMEASUREMENT",
                transform=ax.transAxes, fontsize=28, color="red",
                alpha=0.12, ha="center", va="center",
                fontweight="bold", rotation=18)

    fig.text(0.5, -0.04, CAPTION,
             ha="center", fontsize=7.5, style="italic", color="#555",
             wrap=True)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _css_bar_chart(valid: bool, ei_control: float,
                   ei_residue: float, ei_scrub: float) -> str:
    """Fallback: CSS-only horizontal bar chart (no external deps)."""
    def bar_row(label: str, val: float, color: str) -> str:
        pct = min(max(val * 100, 0), 100)
        val_str = f"{val:.4f}" + (" (invalid)" if not valid else "")
        return (
            f'<tr><td class="bl">{label}</td>'
            f'<td class="bv">{val_str}</td>'
            f'<td class="bc"><div class="bf" '
            f'style="width:{pct:.1f}%;background:{color}"></div></td></tr>'
        )

    invalid_banner = ""
    if not valid:
        invalid_banner = (
            '<p class="invalid-warn">'
            "&#9888; INVALID MEASUREMENT &#8212; control non-zero. "
            "EI values below are not reliable measurements."
            "</p>"
        )

    rows = (
        bar_row("Control (validity gate)", ei_control, "#43a047")
        + bar_row("Residue (no scrub)", ei_residue, "#e53935")
        + bar_row("LMHC scrub", ei_scrub, "#1e88e5")
    )
    return f"""
{invalid_banner}
<style>
.bar-table {{border-collapse:collapse;width:100%;max-width:600px}}
.bar-table td {{padding:4px 8px;vertical-align:middle}}
.bl {{white-space:nowrap;font-size:13px;width:180px}}
.bv {{font-family:monospace;font-size:13px;width:90px;text-align:right}}
.bc {{width:300px}}
.bf {{height:22px;border-radius:3px;min-width:2px}}
</style>
<table class="bar-table">{rows}</table>
<p class="caption">{CAPTION}</p>
"""


# ── HTML generation ───────────────────────────────────────────────────────────

def _env_table(env: dict) -> str:
    fields = [
        ("GPU", env.get("gpu_name", "—")),
        ("Architecture", env.get("architecture_name", "—")),
        ("Compute capability", env.get("compute_capability", "—")),
        ("Driver version", env.get("driver_version", "—")),
        ("CUDA runtime", env.get("cuda_runtime_version", "—")),
        ("CUDA driver API", env.get("cuda_driver_api_version", "—")),
        ("nvcc compiled with", env.get("nvcc_version_compiled_with", "—")),
        ("OS", env.get("os", "—")),
        ("WSL2", str(env.get("wsl2", "—"))),
        ("SM count", str(env.get("gpu_sm_count", "—"))),
        ("Shared mem / SM", f"{env.get('shared_mem_per_sm_bytes', 0) // 1024} KB"),
    ]
    rows = "".join(
        f"<tr><td class='ek'>{k}</td><td class='ev'>{v}</td></tr>"
        for k, v in fields
    )
    return f"<table class='env-table'>{rows}</table>"


def make_html(data: dict) -> str:
    valid: bool = data.get("valid", False)
    valid_reason: str = data.get("valid_reason", "")
    env = data.get("environment", {})
    cfg = data.get("config", {})
    residue = data.get("residue_run", {})
    control = data.get("control_run", {})
    scrub = data.get("lmhc_scrub_run", {})

    gpu_name = env.get("gpu_name", "Unknown GPU")
    arch = env.get("architecture_name", "")
    ts = data.get("timestamp", "")
    result_summary = data.get("result_summary", "")

    ei_control = float(control.get("exposure_index", 0.0))
    ei_residue = float(residue.get("exposure_index", 0.0))
    ei_scrub = float(scrub.get("exposure_index", 0.0))

    same_sm = residue.get("same_sm_trials", "N/A")
    trials = cfg.get("trials_per_phase", "N/A")
    buf_words = cfg.get("buffer_words", "N/A")
    buf_bytes = cfg.get("buffer_bytes", "N/A")

    # Chart: try matplotlib, fall back to CSS bars.
    png_bytes = _try_matplotlib_chart(valid, gpu_name,
                                      ei_control, ei_residue, ei_scrub)
    if png_bytes:
        b64 = base64.b64encode(png_bytes).decode()
        chart_html = (
            f'<img src="data:image/png;base64,{b64}" '
            f'alt="EI bar chart for {gpu_name}" '
            f'style="max-width:700px;display:block;margin:12px auto">'
            f'<p class="caption">{CAPTION}</p>'
        )
        invalid_banner_for_chart = ""
        if not valid:
            invalid_banner_for_chart = (
                '<p class="invalid-warn">'
                "&#9888; INVALID MEASUREMENT &#8212; control non-zero. "
                "EI values in the chart are not reliable measurements."
                "</p>"
            )
        chart_html = invalid_banner_for_chart + chart_html
    else:
        chart_html = _css_bar_chart(valid, ei_control, ei_residue, ei_scrub)

    # Validity banner
    if valid:
        validity_html = (
            '<div class="validity-ok">'
            "&#10003; VALID &#8212; " + valid_reason +
            "</div>"
        )
    else:
        validity_html = (
            '<div class="validity-fail">'
            "&#9888; INVALID &#8212; " + valid_reason +
            "</div>"
        )

    # Trial stats table
    def _stat(label: str, val: object) -> str:
        return f"<tr><td class='ek'>{label}</td><td class='ev'>{val}</td></tr>"

    nz_residue = residue.get("nonzero_word_count_same_sm", "N/A")
    total_residue = residue.get("total_word_count_same_sm", "N/A")
    nz_control = control.get("nonzero_word_count", "N/A")
    nz_scrub = scrub.get("nonzero_word_count_same_sm", "N/A")

    stats_html = "<table class='env-table'>"
    stats_html += _stat("Trials per phase", trials)
    stats_html += _stat("Buffer size", f"{buf_words} words / {buf_bytes} bytes (16 KB, fixed)")
    stats_html += _stat("Pattern sentinel", cfg.get("pattern_sentinel", "0xCAFEBABE"))
    stats_html += _stat("Launch bounds", cfg.get("launch_bounds", "—"))
    stats_html += _stat("Same-SM trials (residue phase)", f"{same_sm} / {trials}")
    stats_html += _stat("Nonzero words — control", str(nz_control))
    stats_html += _stat("Nonzero words — residue (same-SM)", str(nz_residue))
    stats_html += _stat("Total words checked (same-SM)", str(total_residue))
    stats_html += _stat("Nonzero words — LMHC scrub", str(nz_scrub))
    stats_html += "</table>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LMHC-Forge Report: {gpu_name}</title>
<style>
  body {{font-family:system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;color:#222}}
  h1 {{font-size:1.3rem;margin-bottom:4px}}
  h2 {{font-size:1.05rem;margin-top:28px;margin-bottom:8px;border-bottom:1px solid #ddd;padding-bottom:4px}}
  .meta {{color:#666;font-size:0.85rem;margin-bottom:16px}}
  .validity-ok  {{background:#e8f5e9;border-left:4px solid #43a047;padding:10px 14px;
                  font-weight:bold;color:#2e7d32;margin:12px 0;border-radius:3px}}
  .validity-fail{{background:#ffebee;border-left:4px solid #e53935;padding:10px 14px;
                  font-weight:bold;color:#b71c1c;margin:12px 0;border-radius:3px}}
  .invalid-warn {{background:#fff3e0;border-left:4px solid #fb8c00;padding:8px 14px;
                  font-weight:bold;color:#e65100;margin:10px 0;border-radius:3px}}
  .result-summary {{background:#f5f5f5;border-left:4px solid #9e9e9e;padding:8px 14px;
                    font-size:0.9rem;margin:10px 0;border-radius:3px}}
  .caption {{font-style:italic;font-size:0.82rem;color:#555;margin-top:8px;text-align:center}}
  .env-table {{border-collapse:collapse;font-size:0.88rem;width:100%;max-width:640px}}
  .env-table td {{padding:4px 10px;border-bottom:1px solid #eee;vertical-align:top}}
  .ek {{font-weight:500;width:220px;color:#444}}
  .ev {{font-family:monospace}}
  .framing {{font-size:0.8rem;color:#666;border-top:1px solid #eee;padding-top:12px;
             margin-top:32px;line-height:1.5}}
</style>
</head>
<body>

<h1>LMHC-Forge &#8212; Device Report: {gpu_name}</h1>
<p class="meta">{arch} &nbsp;|&nbsp; {ts}</p>

{validity_html}

{"<div class='result-summary'>" + result_summary + "</div>" if result_summary else ""}

<h2>Exposure Index by Phase</h2>
{chart_html}

<h2>Environment</h2>
{_env_table(env)}

<h2>Trial Statistics</h2>
{stats_html}

<p class="framing">
<strong>Framing:</strong> {FRAMING}
<br>
Reports are generated from validated JSONs and depict controlled-placement measurements only.
</p>

</body>
</html>
"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_file",
                        help="Path to a multi_gpu_repro result JSON")
    parser.add_argument("--output", default=None,
                        help="Output HTML path (default: <json_stem>_report.html)")
    args = parser.parse_args()

    src = Path(args.json_file)
    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        sys.exit(1)

    with open(src) as f:
        data = json.load(f)

    schema = data.get("schema_version", "")
    if not schema.startswith("multi_gpu_repro"):
        print(f"WARNING: unexpected schema_version '{schema}' — "
              "this may not be a multi_gpu_repro result file", file=sys.stderr)

    html = make_html(data)

    out_path = Path(args.output) if args.output else src.with_suffix("_report.html")
    # Avoid clobbering the source JSON with a bad suffix substitution
    if out_path == src:
        out_path = src.parent / (src.stem + "_report.html")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report written to: {out_path}")
    if not data.get("valid", True):
        print("NOTE: valid=false — the report labels EI values as INVALID.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
