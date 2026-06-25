#!/usr/bin/env python3
"""
witness_gen.py — Module 4: LMHC Witness Generator

Workflow
--------
1. Module 1 (discover.py, taint pass) on the input .cu:
     → sensitive_regions (confidential __shared__ buffers)

2. Module 2 (lmhc_rewriter.py, taint-only) on the same .cu:
     → rewritten source with volatile+fence scrubs inserted
     → scrub_sites (which buffer(s) scrubbed in which kernel)
     → all_exits_protected (True iff no kernel was skipped/errored)

3. Compile the rewritten source to a .cubin (nvcc --cubin).
   The cubin is the COMPILED DEVICE BINARY — not the .cu source.

4. Build policy_doc = canonical JSON of {policy_version, arch,
   compile_flags, sensitive_regions, scrub_sites, all_exits_protected}.

5. kernel_hash = SHA-384(cubin_bytes || policy_json_bytes)
   Both inputs are deterministic; Module 5 re-derives the hash independently.

6. Emit witness.json.

Usage
-----
    # From the lmhc-forge root:
    python3 src/witness_gen.py src/test_kernels.cu \\
        [--arch sm_89] [--nvcc /usr/local/cuda-12.8/bin/nvcc] \\
        [--ccbin /usr/bin/gcc-14] \\
        [--output results/witness.json] \\
        [--keep-rewritten src/test_kernels_m4_lmhc.cu]
"""

import hashlib
import json
import subprocess
import sys
import tempfile
import argparse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from discover import analyse
from lmhc_rewriter import rewrite_file


POLICY_VERSION = "lmhc-v1.0"


# ──────────────────────────────────────────────────────────────────────────────
# Cubin compiler
# ──────────────────────────────────────────────────────────────────────────────

def compile_to_cubin(source_path: Path, arch: str,
                     nvcc: str, ccbin: str) -> bytes:
    """
    Compile source_path to a device-only .cubin using nvcc --cubin.
    Returns the raw cubin bytes. The .cubin is removed after reading.
    Raises RuntimeError if nvcc exits non-zero.
    """
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as tf:
        cubin_out = Path(tf.name)

    cmd = [
        nvcc, "--cubin",
        f"-arch={arch}",
        "-O3",
        "-ccbin", ccbin,
        "-o", str(cubin_out),
        str(source_path),
    ]
    print(f"[M4]   {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr.strip()
    if stderr:
        print(f"[M4]   nvcc stderr: {stderr}", file=sys.stderr)
    if result.returncode != 0:
        cubin_out.unlink(missing_ok=True)
        raise RuntimeError(f"nvcc --cubin failed (exit {result.returncode})")

    data = cubin_out.read_bytes()
    cubin_out.unlink(missing_ok=True)
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Hash
# ──────────────────────────────────────────────────────────────────────────────

def make_policy_doc(arch: str, compile_flags: str,
                    sensitive_regions: list, scrub_sites: list,
                    all_exits_protected: bool,
                    source_file_hash: str) -> dict:
    """Canonical policy document — sort_keys=True when serialising.

    source_file_hash = SHA-384(raw bytes of the input .cu source file).
    Included in the hash so that swapping the source file to a de-annotated
    copy breaks CHECK 1 even when the cubin is identical.
    """
    return {
        "all_exits_protected": all_exits_protected,
        "arch":                arch,
        "compile_flags":       compile_flags,
        "policy_version":      POLICY_VERSION,
        "scrub_sites":         scrub_sites,
        "sensitive_regions":   sensitive_regions,
        "source_file_hash":    source_file_hash,
    }


def compute_hash(cubin_bytes: bytes, policy_doc: dict) -> str:
    """SHA-384(cubin_bytes || policy_json_bytes), policy serialised sort_keys=True."""
    policy_bytes = json.dumps(policy_doc, sort_keys=True,
                              separators=(",", ":")).encode("utf-8")
    h = hashlib.sha384()
    h.update(cubin_bytes)
    h.update(policy_bytes)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="LMHC Module 4: Witness Generator"
    )
    ap.add_argument("file",
                    help="Input .cu source file")
    ap.add_argument("--arch",   default="sm_89",
                    help="CUDA architecture (default: sm_89)")
    ap.add_argument("--nvcc",   default="/usr/local/cuda-12.8/bin/nvcc",
                    help="Path to nvcc")
    ap.add_argument("--ccbin",  default="/usr/bin/gcc-14",
                    help="Host compiler for -ccbin (default: /usr/bin/gcc-14)")
    ap.add_argument("--output", default=None,
                    help="Output witness.json (default: results/witness.json)")
    ap.add_argument("--keep-rewritten", default=None,
                    help="Save the Module-2-rewritten source to this path")
    ap.add_argument("--save-cubin", default=None,
                    help="Save the compiled .cubin to this path (required for Module 5)")
    args = ap.parse_args()

    source_path = Path(args.file).resolve()
    if not source_path.exists():
        print(f"error: {source_path} not found", file=sys.stderr)
        sys.exit(1)

    source_bytes = source_path.read_bytes()
    source = source_bytes.decode("utf-8")
    source_file_hash = hashlib.sha384(source_bytes).hexdigest()
    print(f"[M4] source_file_hash: {source_file_hash}", file=sys.stderr)

    # ── Step 1: Module 1 — taint discovery ───────────────────────────────────
    print("[M4] Step 1: Module 1 taint discovery…", file=sys.stderr)
    discovery = analyse(source, do_taint=True)

    sensitive_regions = []
    for k in discovery:
        for buf in k["shared_buffers"]:
            if buf["sensitivity"] == "confidential":
                sensitive_regions.append({
                    "kernel": k["name"],
                    "buffer": buf["name"],
                    "type":   buf["type"],
                    "nelems": buf["nelems"],
                })
    print(f"       {len(sensitive_regions)} confidential region(s): "
          + ", ".join(f"{r['kernel']}.{r['buffer']}" for r in sensitive_regions),
          file=sys.stderr)

    # ── Step 2: Module 2 — taint-only rewrite ────────────────────────────────
    print("[M4] Step 2: Module 2 taint-only rewrite…", file=sys.stderr)
    rewritten_source, summary = rewrite_file(source, discovery, taint_only=True)

    scrub_sites = []
    all_exits_protected = (len(summary["errors"]) == 0)
    for detail in summary["details"]:
        if detail["status"] == "rewritten" and detail["scrubbed_buffers"]:
            scrub_sites.append({
                "kernel":           detail["kernel"],
                "scrubbed_buffers": detail["scrubbed_buffers"],
            })
        elif detail["status"] == "skipped":
            all_exits_protected = False
    print(f"       {len(scrub_sites)} kernel(s) with scrubs inserted, "
          f"all_exits_protected={all_exits_protected}", file=sys.stderr)
    if summary["errors"]:
        print(f"       rewriter errors: {summary['errors']}", file=sys.stderr)

    if args.keep_rewritten:
        Path(args.keep_rewritten).write_text(rewritten_source, encoding="utf-8")
        print(f"       rewritten source → {args.keep_rewritten}", file=sys.stderr)

    # ── Step 3: Compile rewritten source → cubin ─────────────────────────────
    print("[M4] Step 3: Compiling rewritten source → .cubin…", file=sys.stderr)
    with tempfile.NamedTemporaryFile(
            suffix="_lmhc_m4.cu", delete=False,
            mode="w", encoding="utf-8") as tf:
        tf.write(rewritten_source)
        tmp_cu = Path(tf.name)

    try:
        cubin_bytes = compile_to_cubin(tmp_cu, args.arch, args.nvcc, args.ccbin)
    finally:
        tmp_cu.unlink(missing_ok=True)

    compile_flags = (
        f"--cubin -arch={args.arch} -O3 -ccbin {args.ccbin}"
    )
    print(f"       cubin: {len(cubin_bytes)} bytes", file=sys.stderr)

    if args.save_cubin:
        Path(args.save_cubin).write_bytes(cubin_bytes)
        print(f"       cubin saved → {args.save_cubin}", file=sys.stderr)

    # ── Step 4: Policy document + SHA-384 ────────────────────────────────────
    print("[M4] Step 4: SHA-384(cubin || policy)…", file=sys.stderr)
    policy_doc = make_policy_doc(
        arch=args.arch,
        compile_flags=compile_flags,
        sensitive_regions=sensitive_regions,
        scrub_sites=scrub_sites,
        all_exits_protected=all_exits_protected,
        source_file_hash=source_file_hash,
    )
    kernel_hash = compute_hash(cubin_bytes, policy_doc)
    print(f"       kernel_hash: {kernel_hash}", file=sys.stderr)

    # ── Step 5: Emit witness.json ─────────────────────────────────────────────
    witness = {
        "kernel_hash":          kernel_hash,
        "policy_version":       POLICY_VERSION,
        "sensitive_regions":    sensitive_regions,
        "scrub_sites":          scrub_sites,
        "all_exits_protected":  all_exits_protected,
        "arch":                 args.arch,
        "compile_flags":        compile_flags,
        "source_file":          str(source_path),
        "source_file_hash":     source_file_hash,
        "cubin_size_bytes":     len(cubin_bytes),
        "hardware":             "NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace, sm_89)",
        "generated_at":         datetime.now(timezone.utc).isoformat(),
        "_hash_construction":   (
            "SHA-384(cubin_bytes "
            "|| json.dumps(policy_doc, sort_keys=True, separators=(',',':')).encode('utf-8'))"
            "; policy_doc includes source_file_hash=SHA-384(source_file_bytes)"
        ),
    }

    out_path = (Path(args.output) if args.output
                else Path("results/witness.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(witness, indent=2), encoding="utf-8")
    print(f"[M4] witness.json → {out_path}", file=sys.stderr)

    # Print to stdout so the caller can inspect without reading the file
    print(json.dumps(witness, indent=2))


if __name__ == "__main__":
    main()
