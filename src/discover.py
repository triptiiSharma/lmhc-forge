#!/usr/bin/env python3
"""
discover.py — Module 1: LMHC Sensitive Buffer Discovery Engine

Conservative pass (default): flags EVERY __shared__ buffer in every __global__ kernel.
This is an over-approximation (Definition 1 in the paper): any buffer that COULD hold
sensitive data is flagged, regardless of actual data flow.

Taint refinement (--taint): narrows the flagged set to buffers that are reachable from
parameters annotated with __confidential__ (or __attribute__((annotate("confidential")))).
A buffer is tainted if any thread writes to it from a tainted source within the kernel.
This is a forward, intra-procedural, field-insensitive taint propagation.

Usage:
    python3 discover.py <file.cu> [--taint] [--output results/discovery.json]

Limitations:
  - Regex-based parsing; does not handle macro-generated code or complex templates.
  - Taint propagation is syntactic (line-level), not semantic. It will miss indirect
    writes through function calls or pointer aliasing.
  - For a production tool, replace with libclang AST traversal.
"""

import re
import sys
import json
import argparse
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Patterns
# ──────────────────────────────────────────────────────────────────────────────

# Matches __global__ kernel signatures (possibly multi-line up to opening brace)
_KERNEL_SIG = re.compile(
    r'__global__\s+(?:void|[\w:*&]+)\s+(\w+)\s*\(([^)]*)\)',
    re.DOTALL
)

# Matches a __shared__ variable declaration inside a kernel body
# Captures: element_type, buffer_name, optional array dimension
_SHARED_DECL = re.compile(
    r'__shared__\s+'
    r'((?:unsigned\s+|signed\s+|const\s+|volatile\s+)*'   # qualifiers
    r'(?:unsigned\s+)?'
    r'(?:char|short|int|long|float|double|uint8_t|uint16_t|uint32_t|uint64_t'
    r'|int8_t|int16_t|int32_t|int64_t|half|__half|uchar4|float4|int4))'
    r'\s+(\w+)'                                             # variable name
    r'(?:\s*\[([^\]]+)\])?'                                 # optional [size]
    r'\s*;',
    re.VERBOSE
)

# __confidential__ annotation — either our macro or GCC attribute form
_CONFIDENTIAL_PARAM = re.compile(
    r'(?:__confidential__|'
    r'__attribute__\s*\(\s*\(\s*annotate\s*\(\s*"confidential"\s*\)\s*\)\s*\))'
    r'\s+(?:const\s+)?(?:\w+\s*\*+\s*|\w+\s+)(\w+)'
)

# Simple heuristic: shared buffer `name` is tainted if the kernel body contains
# an assignment into it that references a known confidential param on the RHS.
# This is syntactic, not semantic — see module docstring.
def _taint_check(body: str, buf_name: str, confidential_params: list[str]) -> bool:
    if not confidential_params:
        return False
    for param in confidential_params:
        # Look for: buf_name[...] = ... param ...
        pattern = re.compile(
            r'\b' + re.escape(buf_name) + r'\s*\[[^\]]*\]\s*=[^;]*\b'
            + re.escape(param) + r'\b'
        )
        if pattern.search(body):
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Kernel body extractor (brace-balanced)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_kernel_body(source: str, sig_end: int) -> tuple[int, int] | None:
    """Return (open_brace_pos, close_brace_pos) of the first { } block after sig_end."""
    depth = 0
    start = None
    i = sig_end
    while i < len(source):
        ch = source[i]
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                return (start, i)
        i += 1
    return None


def _count_returns(body: str) -> int:
    """Count explicit return statements in a kernel body (excluding closing })."""
    return len(re.findall(r'\breturn\b\s*;', body))


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyse(source: str, do_taint: bool) -> list[dict]:
    kernels = []
    for sig_match in _KERNEL_SIG.finditer(source):
        kernel_name = sig_match.group(1)
        param_str   = sig_match.group(2)
        sig_end     = sig_match.end()

        # Extract kernel body
        body_bounds = _extract_kernel_body(source, sig_end)
        if body_bounds is None:
            continue
        bopen, bclose = body_bounds
        body = source[bopen:bclose + 1]

        # Line number of the signature
        line_no = source[:sig_match.start()].count('\n') + 1

        # Find __shared__ buffers
        shared_buffers = []
        for decl in _SHARED_DECL.finditer(body):
            elem_type = decl.group(1).strip()
            buf_name  = decl.group(2)
            size_expr = decl.group(3) if decl.group(3) else 'unknown'
            decl_line = line_no + body[:decl.start()].count('\n')

            # Try to resolve size_expr to an integer
            try:
                nelems = int(eval(size_expr, {"__builtins__": {}}))  # safe literals only
            except Exception:
                nelems = size_expr

            shared_buffers.append({
                "name":      buf_name,
                "type":      elem_type,
                "nelems":    nelems,
                "size_expr": size_expr,
                "line":      decl_line,
                "sensitivity": "conservative",  # default: always flag
            })

        if do_taint and shared_buffers:
            # Find confidential parameters
            conf_params = [m.group(1) for m in _CONFIDENTIAL_PARAM.finditer(param_str)]
            for buf in shared_buffers:
                if _taint_check(body, buf["name"], conf_params):
                    buf["sensitivity"] = "confidential"
                else:
                    buf["sensitivity"] = "non-confidential"

        exits = _count_returns(body)
        kernels.append({
            "name":           kernel_name,
            "line":           line_no,
            "shared_buffers": shared_buffers,
            "explicit_returns": exits,
            "single_exit":    (exits <= 1),
        })

    return kernels


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="LMHC Module 1: Sensitive Buffer Discovery")
    ap.add_argument("file", help="Input .cu file")
    ap.add_argument("--taint", action="store_true",
                    help="Enable taint refinement from __confidential__ parameters")
    ap.add_argument("--output", default=None,
                    help="Write JSON to this path (default: stdout)")
    args = ap.parse_args()

    source = Path(args.file).read_text(encoding="utf-8")
    kernels = analyse(source, args.taint)

    total_bufs   = sum(len(k["shared_buffers"]) for k in kernels)
    flagged_bufs = total_bufs  # conservative: all are flagged
    if args.taint:
        flagged_bufs = sum(
            1 for k in kernels
            for b in k["shared_buffers"]
            if b["sensitivity"] in ("conservative", "confidential")
        )

    result = {
        "file":          args.file,
        "pass":          "taint" if args.taint else "conservative",
        "kernels_found": len(kernels),
        "total_shared_buffers": total_bufs,
        "flagged_buffers": flagged_bufs,
        "kernels":       kernels,
    }

    output_json = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"Discovery results written to {args.output}")
    else:
        print(output_json)

    # Summary to stderr
    print(f"\n[discover] {args.file}: {len(kernels)} kernel(s), "
          f"{total_bufs} __shared__ buffer(s), "
          f"{flagged_bufs} flagged ({'taint' if args.taint else 'conservative'} pass)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
