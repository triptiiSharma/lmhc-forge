#!/usr/bin/env python3
"""
lmhc_rewriter.py — Module 2: LMHC Scrub Rewriter

Reads a .cu file, consults Module 1 discovery output (JSON), and inserts the
LMHC scrub sequence before every kernel exit for each flagged __shared__ buffer.

LMHC scrub pattern (per Theorem 1 / §4.4 of the paper):
    volatile TYPE *__lmhc_p = (volatile TYPE *)BUFNAME;
    for (int __lmhc_i = (int)threadIdx.x;
         __lmhc_i < NELEMS;
         __lmhc_i += (int)blockDim.x)
        __lmhc_p[__lmhc_i] = (TYPE)0;
    __threadfence_block();
    __syncthreads();

SCOPE LIMITATION (see SPEC.md LIMITATIONS section):
    This rewriter handles kernels with a single exit point only
    (no early returns, no goto). Kernels with >0 explicit return statements
    before the closing brace are REJECTED with an error message.
    A full libclang-based transform would handle arbitrary control flow;
    that is left as future work because all kernels in this artifact are
    single-exit by construction.

Usage:
    # Conservative (scrub every __shared__ buffer):
    python3 lmhc_rewriter.py kernel.cu --output kernel_lmhc.cu

    # Taint-refined (scrub only confidential buffers):
    python3 lmhc_rewriter.py kernel.cu --discovery results/discovery.json \\
            --output kernel_lmhc.cu

    # Dry-run (print rewritten source, don't write file):
    python3 lmhc_rewriter.py kernel.cu --dry-run
"""

import re
import sys
import json
import argparse
from pathlib import Path

# Reuse the analyser from Module 1
sys.path.insert(0, str(Path(__file__).parent))
from discover import analyse, _extract_kernel_body


# ──────────────────────────────────────────────────────────────────────────────
# Scrub code generation
# ──────────────────────────────────────────────────────────────────────────────

def _scrub_block(buf_name: str, elem_type: str, nelems: str | int, indent: str) -> str:
    """Generate the LMHC scrub for one __shared__ buffer."""
    i  = indent
    return (
        f"{i}/* BEGIN LMHC scrub — {buf_name} ({nelems} x {elem_type}) */\n"
        f"{i}{{\n"
        f"{i}    volatile {elem_type} *__lmhc_p = (volatile {elem_type} *){buf_name};\n"
        f"{i}    for (int __lmhc_i = (int)threadIdx.x;\n"
        f"{i}         __lmhc_i < (int)({nelems});\n"
        f"{i}         __lmhc_i += (int)blockDim.x)\n"
        f"{i}        __lmhc_p[__lmhc_i] = ({elem_type})0;\n"
        f"{i}}}\n"
        f"{i}__threadfence_block();\n"
        f"{i}__syncthreads();\n"
        f"{i}/* END LMHC scrub — {buf_name} */\n"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Kernel rewriter (single-exit scope)
# ──────────────────────────────────────────────────────────────────────────────

_RETURN_RE = re.compile(r'\breturn\b\s*;')


def _rewrite_kernel(source: str, kernel_info: dict,
                    body_open: int, body_close: int,
                    taint_only: bool) -> tuple[str, list[str]]:
    """
    Rewrite one kernel by inserting scrub before the closing brace.
    Returns (modified_source, list_of_inserted_buffer_names).
    Raises ValueError for kernels that are out of scope.
    """
    body = source[body_open:body_close + 1]

    # Scope check: reject if any explicit return inside the body
    # (the body includes the closing brace; we check everything before it)
    inner = source[body_open + 1:body_close]
    if _RETURN_RE.search(inner):
        raise ValueError(
            f"Kernel '{kernel_info['name']}' has early return(s) — "
            "out of scope for single-exit rewriter. Insert scrub manually."
        )

    # Determine which buffers to scrub
    buffers = kernel_info.get("shared_buffers", [])
    if taint_only:
        buffers = [b for b in buffers if b.get("sensitivity") == "confidential"]
    if not buffers:
        return source, []

    # Determine indentation from the closing brace's line
    closing_line_start = source.rfind('\n', 0, body_close) + 1
    indent_chars = len(source[closing_line_start:body_close]) - \
                   len(source[closing_line_start:body_close].lstrip())
    indent = " " * max(indent_chars, 4)

    scrub_code = ""
    for buf in buffers:
        scrub_code += _scrub_block(
            buf["name"], buf["type"], buf["nelems"], indent
        )

    # Insert scrub just before the closing brace
    new_source = source[:body_close] + scrub_code + source[body_close:]
    return new_source, [b["name"] for b in buffers]


# ──────────────────────────────────────────────────────────────────────────────
# File-level rewriter
# ──────────────────────────────────────────────────────────────────────────────

def rewrite_file(source: str,
                 discovery: list[dict] | None = None,
                 taint_only: bool = False) -> tuple[str, dict]:
    """
    Rewrite all kernels in source.
    discovery: Module 1 output for this file (list of kernel dicts).
               If None, run conservative discovery in-process.
    Returns (rewritten_source, summary_dict).
    """
    if discovery is None:
        discovery = analyse(source, do_taint=taint_only)

    # Build a map: kernel_name → kernel_info
    kernel_map = {k["name"]: k for k in discovery}

    # We must process kernels in REVERSE order so that inserting text in one
    # kernel doesn't shift the byte offsets of later ones.
    from discover import _KERNEL_SIG
    matches = list(_KERNEL_SIG.finditer(source))

    results = []
    errors  = []

    for sig_match in reversed(matches):
        kname    = sig_match.group(1)
        kinfo    = kernel_map.get(kname)
        if kinfo is None:
            continue  # not in discovery (possibly a device function)

        body_bounds = _extract_kernel_body(source, sig_match.end())
        if body_bounds is None:
            errors.append(f"{kname}: could not find kernel body")
            continue

        bopen, bclose = body_bounds
        try:
            source, scrubbed = _rewrite_kernel(
                source, kinfo, bopen, bclose, taint_only
            )
            results.append({
                "kernel":  kname,
                "status":  "rewritten",
                "scrubbed_buffers": scrubbed,
            })
        except ValueError as exc:
            errors.append(str(exc))
            results.append({"kernel": kname, "status": "skipped", "reason": str(exc)})

    summary = {
        "kernels_processed": len(results),
        "errors":  errors,
        "details": results,
    }
    return source, summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="LMHC Module 2: Insert scrub into CUDA kernels"
    )
    ap.add_argument("file",       help="Input .cu file")
    ap.add_argument("--output",   default=None, help="Output .cu path (default: <file>_lmhc.cu)")
    ap.add_argument("--discovery",default=None,
                    help="JSON from discover.py (if omitted, runs conservative discovery)")
    ap.add_argument("--taint-only", action="store_true",
                    help="Only scrub buffers marked 'confidential' in discovery JSON")
    ap.add_argument("--dry-run",  action="store_true", help="Print rewritten source, no file written")
    args = ap.parse_args()

    source = Path(args.file).read_text(encoding="utf-8")

    discovery_list = None
    if args.discovery:
        d = json.loads(Path(args.discovery).read_text(encoding="utf-8"))
        discovery_list = d.get("kernels", [])

    new_source, summary = rewrite_file(source, discovery_list, args.taint_only)

    if args.dry_run:
        print(new_source)
    else:
        out_path = args.output or Path(args.file).stem + "_lmhc.cu"
        Path(out_path).write_text(new_source, encoding="utf-8")
        print(f"[lmhc_rewriter] Written to {out_path}")

    print(json.dumps(summary, indent=2))

    if summary["errors"]:
        print("\nErrors:", file=sys.stderr)
        for e in summary["errors"]:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
