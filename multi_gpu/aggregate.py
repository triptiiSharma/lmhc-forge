#!/usr/bin/env python3
"""aggregate.py — LMHC-Forge Multi-GPU Replication Aggregator

Reads all per-device JSON files from multi_gpu/results/ and emits one
comparison table. No device numbers are hardcoded here; the table is
generated entirely from the JSON files as submitted.

Usage:
    python3 aggregate.py [--results-dir <path>] [--output <file>]
                         [--tsv <file>] [--html <file>]

Output formats:
    Markdown table (default, to stdout or --output file)
    TSV file (if --tsv <file> is given)
    HTML table (if --html <file> is given; valid=false rows are flagged)

Columns: GPU | Architecture | Driver | CUDA Runtime | Same-SM trials |
         EI (no scrub) | Control EI | EI (LMHC scrub) | Valid (Y/N)
"""

import argparse
import json
import sys
from pathlib import Path

COLUMNS = [
    "GPU",
    "Architecture",
    "Driver",
    "CUDA Runtime",
    "Same-SM trials",
    "EI (no scrub)",
    "Control EI",
    "EI (LMHC scrub)",
    "Valid",
]

FRAMING_NOTE = (
    "NOTE: Co-residency is forced on every device (single-block "
    "__launch_bounds__(32,1) + %smid verification). These are "
    "controlled-placement results consistent with the adversarial threat "
    "model — not natural-scheduling frequencies, and not a claim about "
    "production prevalence or about non-tested hardware. See "
    "INTERPRETATION.md for pre-committed outcome readings."
)


def load_result(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    env = d.get("environment", {})
    cfg = d.get("config", {})
    residue = d.get("residue_run", {})
    control = d.get("control_run", {})
    scrub   = d.get("lmhc_scrub_run", {})

    return {
        "file": path.name,
        "gpu":          env.get("gpu_name", "unknown"),
        "arch":         env.get("architecture_name", "unknown"),
        "driver":       env.get("driver_version", "unknown"),
        "cuda_rt":      env.get("cuda_runtime_version", "unknown"),
        "same_sm":      residue.get("same_sm_trials", "N/A"),
        "ei_residue":   residue.get("exposure_index", float("nan")),
        "ei_control":   control.get("exposure_index", float("nan")),
        "ei_scrub":     scrub.get("exposure_index", float("nan")),
        "valid":        d.get("valid", False),
        "valid_reason": d.get("valid_reason", ""),
        "timestamp":    d.get("timestamp", ""),
        "trials":       cfg.get("trials_per_phase", "N/A"),
        "buffer_words": cfg.get("buffer_words", "N/A"),
        "result_summary": d.get("result_summary", ""),
    }


def fmt_ei(val) -> str:
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return str(val)


def fmt_valid(row) -> str:
    return "Y" if row["valid"] else "N"


def build_rows(results):
    return [
        [
            row["gpu"],
            row["arch"],
            row["driver"],
            row["cuda_rt"],
            str(row["same_sm"]),
            fmt_ei(row["ei_residue"]),
            fmt_ei(row["ei_control"]),
            fmt_ei(row["ei_scrub"]),
            fmt_valid(row),
        ]
        for row in results
    ]


def markdown_table(rows) -> str:
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
              for i, h in enumerate(COLUMNS)]
    sep = "| " + " | ".join("-" * w for w, _ in zip(widths, COLUMNS)) + " |"
    header = "| " + " | ".join(h.ljust(w) for h, w in zip(COLUMNS, widths)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |")
    return "\n".join(lines)


def tsv_table(rows) -> str:
    lines = ["\t".join(COLUMNS)]
    for row in rows:
        lines.append("\t".join(row))
    return "\n".join(lines)


_HTML_CAPTION = (
    "Co-residency is forced on every device (single-block "
    "__launch_bounds__(32,1) + %smid verification). "
    "These are controlled-placement results consistent with the adversarial "
    "threat model &mdash; not natural-scheduling frequencies, and not a claim "
    "about production prevalence or about non-tested hardware. "
    "See INTERPRETATION.md for pre-committed outcome readings. "
    "Reports are generated from validated JSONs and depict "
    "controlled-placement measurements only."
)


def html_table(results: list) -> str:
    """Render an HTML comparison table; valid=false rows are visibly flagged.

    Only a table is produced. No trend charts, leakage-rate axes, or
    cross-device lines are included — a table is the appropriate form
    for this data (see INTERPRETATION.md).
    """
    th = "".join(f"<th>{c}</th>" for c in COLUMNS)
    rows_html = ""
    for result, row in zip(results, build_rows(results)):
        row_class = "" if result["valid"] else ' class="invalid-row"'
        cells = ""
        for i, cell in enumerate(row):
            # Mark the Valid cell specially
            if COLUMNS[i] == "Valid":
                if result["valid"]:
                    cells += f'<td class="valid-y">{cell}</td>'
                else:
                    cells += f'<td class="valid-n">{cell}</td>'
            else:
                cells += f"<td>{cell}</td>"
        rows_html += f"<tr{row_class}>{cells}</tr>\n"

    invalid_count = sum(1 for r in results if not r["valid"])
    invalid_note = ""
    if invalid_count:
        invalid_note = (
            f'<p class="invalid-note">&#9888; {invalid_count} device(s) marked '
            f"<strong>valid=N</strong> (control was non-zero). Their EI values "
            f"are highlighted in red and must NOT be interpreted as leak or "
            f"no-leak results.</p>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LMHC-Forge — Multi-GPU Replication Results</title>
<style>
  body {{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;
         padding:0 16px;color:#222}}
  h1 {{font-size:1.2rem}}
  table {{border-collapse:collapse;width:100%;font-size:0.88rem}}
  th {{background:#37474f;color:#fff;padding:7px 10px;text-align:left}}
  td {{padding:6px 10px;border-bottom:1px solid #e0e0e0;vertical-align:top}}
  .invalid-row td {{background:#fff8f8}}
  .valid-y {{color:#2e7d32;font-weight:bold}}
  .valid-n {{color:#b71c1c;font-weight:bold}}
  .invalid-note {{background:#fff3e0;border-left:4px solid #fb8c00;
                  padding:8px 14px;color:#e65100;margin:12px 0;border-radius:3px}}
  .caption {{font-size:0.8rem;color:#555;font-style:italic;
             border-top:1px solid #eee;padding-top:10px;margin-top:24px;line-height:1.5}}
</style>
</head>
<body>
<h1>LMHC-Forge &mdash; Multi-GPU Replication Results</h1>
{invalid_note}
<table>
  <thead><tr>{th}</tr></thead>
  <tbody>
{rows_html}  </tbody>
</table>
<p class="caption">{_HTML_CAPTION}</p>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=None,
                        help="Directory of per-device JSON files "
                             "(default: multi_gpu/results/ relative to this script)")
    parser.add_argument("--output", default=None,
                        help="Write Markdown table to this file (default: stdout)")
    parser.add_argument("--tsv", default=None,
                        help="Also write TSV table to this file")
    parser.add_argument("--html", default=None,
                        help="Also write HTML table to this file; "
                             "valid=false rows are flagged, no trend charts")
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).parent / "results"

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {results_dir}", file=sys.stderr)
        print("Run multi_gpu_repro on each GPU and place the output files there.",
              file=sys.stderr)
        sys.exit(1)

    results = []
    for p in json_files:
        try:
            results.append(load_result(p))
        except (KeyError, json.JSONDecodeError) as e:
            print(f"WARNING: skipping {p.name}: {e}", file=sys.stderr)

    if not results:
        print("No valid result files could be parsed.", file=sys.stderr)
        sys.exit(1)

    results.sort(key=lambda r: r["gpu"])
    rows = build_rows(results)

    md = (
        "## LMHC-Forge — Multi-GPU Replication Results\n\n"
        + markdown_table(rows)
        + "\n\n"
        + FRAMING_NOTE
        + "\n\n"
        + "### Notes\n"
    )
    for row in results:
        note_parts = []
        if not row["valid"]:
            note_parts.append(f"INVALID: {row['valid_reason']}")
        if row["result_summary"]:
            note_parts.append(row["result_summary"])
        if note_parts:
            md += f"- **{row['gpu']}** ({row['file']}): {' | '.join(note_parts)}\n"
        else:
            md += f"- **{row['gpu']}** ({row['file']}): trials={row['trials']}, buffer={row['buffer_words']} words\n"

    out = open(args.output, "w") if args.output else sys.stdout
    out.write(md + "\n")
    if args.output:
        out.close()
        print(f"Markdown table written to {args.output}")

    if args.tsv:
        tsv = tsv_table(rows)
        with open(args.tsv, "w") as f:
            f.write(tsv + "\n")
        print(f"TSV table written to {args.tsv}")

    if args.html:
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(html_table(results))
        print(f"HTML table written to {args.html}")

    # Print invalid-device warnings to stderr so they're visible even when
    # stdout is redirected.
    invalid = [r for r in results if not r["valid"]]
    if invalid:
        print("\nWARNING: the following devices have valid=false "
              "(control was non-zero):", file=sys.stderr)
        for r in invalid:
            print(f"  {r['gpu']} ({r['file']}): {r['valid_reason']}", file=sys.stderr)
        print("These devices' EI values are NOT reliable and must NOT be "
              "reported as leaks or non-leaks.", file=sys.stderr)


if __name__ == "__main__":
    main()
