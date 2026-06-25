#!/usr/bin/env python3
"""
attester.py — Module 5: LMHC Attestation Verifier

Independence requirement (SS5.4): re-derives every property from the
.cubin and policy fields. Does NOT trust:
  - all_exits_protected
  - scrub_sites
as pre-computed ground truth — those are claims checked against the binary.
sensitive_regions is read as the CLAIM of what to check, then each listed
kernel is independently verified against the disassembled SASS.

Three checks; PASS requires ALL THREE:

  CHECK 1 -- Hash integrity
    Reconstruct policy_doc from witness.json (same keys/serialisation as
    Module 4's make_policy_doc), compute SHA-384(cubin_bytes || policy_json),
    compare to kernel_hash. Detects any modification to the cubin or to the
    policy claims stored in the witness.

  CHECK 2 -- Structural scrub verification
    For every kernel named in sensitive_regions:
      a. Disassemble the .cubin with cuobjdump -sass.
      b. Locate the kernel's SASS section by demangled name.
      c. Verify that, before the kernel's EXIT instruction, the following
         ordering holds:
           STS [addr], RZ   (>=1 volatile zero-store)
           MEMBAR.SC.CTA    (__threadfence_block)
           BAR.SYNC         (__syncthreads)
         -- all derived from SASS, not from scrub_sites in the witness.

  CHECK 3 -- Kernel inventory completeness
    Enumerate ALL kernels present in the cubin (from cuobjdump). For every
    kernel NOT nominated in sensitive_regions, run Module 1's taint pass on
    the source file named in the witness. If any non-nominated kernel has a
    confidential-derived __shared__ buffer, FAIL with "unaccounted sensitive
    kernel: <name>". This makes sensitive-region nomination a CHECKED claim,
    not a trusted one.

    Residual trust boundary: the taint result is only as strong as the source
    annotations in the file witness.source_file names. That field is NOT
    hash-protected by CHECK 1 -- a sophisticated attacker could swap it.
    The fully independent version (taint re-derived from SASS alone without
    source) remains future work.

Usage:
    python3 src/attester.py results/test_kernels.cubin results/witness.json \\
        [--cuobjdump /usr/local/cuda-12.8/bin/cuobjdump]
"""

import hashlib
import json
import re
import subprocess
import sys
import argparse
from pathlib import Path

# Import Module 1's taint logic for CHECK 3 (same directory as this file).
sys.path.insert(0, str(Path(__file__).parent))
import discover as _discover


# ──────────────────────────────────────────────────────────────────────────────
# SASS patterns — derived independently from the binary
# ──────────────────────────────────────────────────────────────────────────────

_FUNC_RE   = re.compile(r'Function\s*:\s*(\S+)')
_STS_RZ_RE = re.compile(r'\bSTS\b[^;]*\bRZ\b')   # zero-store: STS [addr], RZ
_MEMBAR_RE = re.compile(r'\bMEMBAR\.SC\.CTA\b')   # __threadfence_block()
_BAR_RE    = re.compile(r'\bBAR\.SYNC\b')          # __syncthreads()
_EXIT_RE   = re.compile(r'\bEXIT\b')               # kernel exit


def _demangle(mangled: str) -> str:
    """Itanium ABI length-prefix demangle: _Z11scrub_naivePj → scrub_naive."""
    m = re.match(r'_Z(\d+)(\w+)', mangled)
    if m:
        n = int(m.group(1))
        rest = m.group(2)
        return rest[:n] if len(rest) >= n else rest
    return mangled


def _parse_sass(sass_text: str) -> dict[str, list[str]]:
    """Split cuobjdump -sass output into {demangled_name: [sass_lines]}."""
    funcs: dict[str, list[str]] = {}
    cur: str | None = None
    buf: list[str] = []

    for line in sass_text.splitlines():
        fm = _FUNC_RE.search(line)
        if fm:
            if cur is not None:
                funcs[cur] = buf
            cur = _demangle(fm.group(1))
            buf = []
        elif cur is not None:
            buf.append(line)

    if cur is not None:
        funcs[cur] = buf
    return funcs


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 2 — structural scrub verification (derived from SASS alone)
# ──────────────────────────────────────────────────────────────────────────────

def _verify_scrub(lines: list[str], kernel: str) -> tuple[bool, str]:
    """
    Verify that the SASS for `kernel` contains, before EXIT:
      STS.RZ (≥1) → MEMBAR.SC.CTA → BAR.SYNC
    in that strict ordering.

    Returns (ok: bool, detail: str).
    Does NOT consult witness.json at all.
    """
    exit_pos = [i for i, l in enumerate(lines) if _EXIT_RE.search(l)]
    if not exit_pos:
        return False, f"{kernel}: no EXIT instruction found in SASS"

    # Use the first EXIT as the scope boundary (single-exit kernels).
    # All scrub instructions must appear before it.
    boundary = exit_pos[0]
    pre = lines[:boundary]

    sts_rz = [i for i, l in enumerate(pre) if _STS_RZ_RE.search(l)]
    membar  = [i for i, l in enumerate(pre) if _MEMBAR_RE.search(l)]
    bar     = [i for i, l in enumerate(pre) if _BAR_RE.search(l)]

    if not sts_rz:
        return False, (
            f"{kernel}: FAIL — no STS [addr], RZ before EXIT "
            f"(scrub zero-stores absent or eliminated)"
        )
    if not membar:
        return False, (
            f"{kernel}: FAIL — no MEMBAR.SC.CTA before EXIT "
            f"(__threadfence_block missing — ordering guarantee absent)"
        )
    if not bar:
        return False, (
            f"{kernel}: FAIL — no BAR.SYNC before EXIT "
            f"(__syncthreads missing after scrub)"
        )

    last_sts = max(sts_rz)
    last_mb  = max(membar)
    last_bar = max(bar)

    if last_sts >= last_mb:
        return False, (
            f"{kernel}: FAIL — ordering violated: "
            f"last STS.RZ at line {last_sts} is not before "
            f"last MEMBAR at line {last_mb}"
        )
    if last_mb >= last_bar:
        return False, (
            f"{kernel}: FAIL — ordering violated: "
            f"last MEMBAR at line {last_mb} is not before "
            f"last BAR.SYNC at line {last_bar}"
        )

    return True, (
        f"{kernel}: PASS — scrub sequence verified in SASS: "
        f"STS.RZ@{last_sts} < MEMBAR@{last_mb} < BAR.SYNC@{last_bar} "
        f"< EXIT@{boundary}  "
        f"[{len(sts_rz)} zero-store(s) found]"
    )


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 1 — hash integrity (re-derived from cubin + policy fields)
# ──────────────────────────────────────────────────────────────────────────────

def _reconstruct_policy_doc(w: dict, source_file_hash: str | None) -> dict:
    """
    Reconstruct the policy_doc that Module 4 committed to.
    Uses the same key set and value types as witness_gen.make_policy_doc.

    source_file_hash is computed INDEPENDENTLY by the caller (hashing the
    bytes at witness['source_file']), NOT read from the witness — same
    independence principle as CHECK 2 re-deriving SASS from the binary.

    If the source file is unavailable, the caller passes None; the resulting
    policy_doc will not match any legitimate kernel_hash, causing CHECK 1 to
    fail conservatively.

    NOTE: all_exits_protected and scrub_sites are from the witness for
    hash-reconstruction only — CHECK 2 and CHECK 3 verify them independently.
    """
    return {
        "all_exits_protected": w["all_exits_protected"],
        "arch":                w["arch"],
        "compile_flags":       w["compile_flags"],
        "policy_version":      w["policy_version"],
        "scrub_sites":         w["scrub_sites"],
        "sensitive_regions":   w["sensitive_regions"],
        "source_file_hash":    source_file_hash,
    }


def _recompute_hash(cubin_bytes: bytes, policy_doc: dict) -> str:
    """SHA-384(cubin_bytes || policy_json_bytes), sort_keys=True compact seps."""
    policy_bytes = json.dumps(
        policy_doc, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    h = hashlib.sha384()
    h.update(cubin_bytes)
    h.update(policy_bytes)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Main verifier
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# CHECK 3 — Kernel inventory completeness
# ──────────────────────────────────────────────────────────────────────────────

def _check_kernel_inventory(
    funcs: dict,
    sensitive_regions: list,
    witness: dict,
) -> tuple[bool, list[str]]:
    """
    For every kernel in the cubin that sensitive_regions does NOT nominate,
    run Module 1's taint pass to determine whether it handles confidential data.
    Returns (ok, [detail_line, ...]).

    Residual trust boundary: the taint result is only as strong as the
    __confidential__ annotations in the source file witness['source_file'] names.
    That field is NOT hash-protected by CHECK 1 -- a sophisticated attacker could
    point it at a modified source without the annotation. The fully independent
    version (taint from SASS alone) remains future work.
    """
    nominated     = {r["kernel"] for r in sensitive_regions}
    non_nominated = sorted(set(funcs.keys()) - nominated)

    if not non_nominated:
        return True, ["all kernels in cubin are nominated in sensitive_regions"]

    source_file = witness.get("source_file")
    if not source_file or not Path(source_file).exists():
        return False, [
            f"FAIL -- {len(non_nominated)} non-nominated kernel(s) in cubin "
            f"({', '.join(non_nominated)}) but source_file not available "
            f"for taint verification (source_file={source_file!r})"
        ]

    src_text     = Path(source_file).read_text(encoding="utf-8")
    taint_result = _discover.analyse(src_text, do_taint=True)

    conf_bufs = {
        kr["name"]: [
            b["name"]
            for b in kr["shared_buffers"]
            if b["sensitivity"] == "confidential"
        ]
        for kr in taint_result
    }

    details  = []
    all_ok   = True

    for kernel_name in non_nominated:
        if kernel_name not in conf_bufs:
            # Present in cubin but not in source -- cannot determine taint.
            all_ok = False
            details.append(
                f"FAIL -- kernel '{kernel_name}' is in the cubin but not found "
                f"in {source_file}; cannot determine taint status (conservative FAIL)"
            )
        elif conf_bufs[kernel_name]:
            all_ok = False
            bufs = ", ".join(conf_bufs[kernel_name])
            details.append(
                f"FAIL -- unaccounted sensitive kernel: {kernel_name} "
                f"(tainted __shared__ buffer(s): {bufs})"
            )
        else:
            details.append(
                f"{kernel_name}: OK -- no confidential __shared__ buffers "
                f"(taint-verified from {source_file})"
            )

    return all_ok, details


def _run_cuobjdump(cuobjdump: str, cubin: str) -> str:
    r = subprocess.run(
        [cuobjdump, "-sass", cubin],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        raise RuntimeError(f"cuobjdump failed: {r.stderr.strip()}")
    return r.stdout


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LMHC Module 5: Attestation Verifier"
    )
    ap.add_argument("cubin",   help="Compiled .cubin from Module 4")
    ap.add_argument("witness", help="witness.json from Module 4")
    ap.add_argument(
        "--cuobjdump",
        default="/usr/local/cuda-12.8/bin/cuobjdump",
        help="Path to cuobjdump"
    )
    args = ap.parse_args()

    cubin_path   = Path(args.cubin)
    witness_path = Path(args.witness)

    print("=" * 62)
    print("LMHC Attestation Verifier — Module 5")
    print("=" * 62)
    print(f"cubin   : {cubin_path}")
    print(f"witness : {witness_path}")
    print()

    # ── Load inputs ───────────────────────────────────────────────────────────
    cubin_bytes = cubin_path.read_bytes()
    witness     = json.loads(witness_path.read_text(encoding="utf-8"))

    claimed_hash       = witness["kernel_hash"]
    sensitive_regions  = witness["sensitive_regions"]   # CLAIM: kernels to verify
    policy_version     = witness.get("policy_version", "?")

    print(f"policy_version     : {policy_version}")
    print(f"claimed kernel_hash: {claimed_hash}")
    print(f"kernels to verify  : "
          + ", ".join(f"{r['kernel']}.{r['buffer']}" for r in sensitive_regions))
    print()

    results: list[tuple[str, bool, str]] = []   # (label, ok, detail)

    # ── CHECK 1: hash integrity ───────────────────────────────────────────────
    print("── CHECK 1: Hash integrity ──────────────────────────────────")

    # Independently compute source_file_hash from the actual bytes on disk.
    # NOT read from the witness -- same independence principle as CHECK 2.
    _src_path = witness.get("source_file")
    if _src_path and Path(_src_path).exists():
        _src_bytes       = Path(_src_path).read_bytes()
        _src_hash        = hashlib.sha384(_src_bytes).hexdigest()
        print(f"  source_file      : {_src_path}")
        print(f"  source_file_hash : {_src_hash}  (independently computed)")
    else:
        _src_hash = None
        print(f"  source_file      : {_src_path!r} (NOT FOUND — hash will be null → CHECK 1 will fail)")

    policy_doc     = _reconstruct_policy_doc(witness, _src_hash)
    recomputed     = _recompute_hash(cubin_bytes, policy_doc)
    hash_ok        = (recomputed == claimed_hash)

    print(f"  recomputed : {recomputed}")
    print(f"  claimed    : {claimed_hash}")
    if hash_ok:
        print("  → MATCH")
    else:
        print("  → MISMATCH — cubin or policy fields have been tampered with")
    results.append(("hash_integrity", hash_ok,
                    "SHA-384 match" if hash_ok
                    else f"hash mismatch: recomputed={recomputed[:16]}… "
                         f"claimed={claimed_hash[:16]}…"))
    print()

    # ── CHECK 2: structural scrub verification ────────────────────────────────
    print("── CHECK 2: Structural scrub verification (SASS) ───────────")
    sass_text = _run_cuobjdump(args.cuobjdump, args.cubin)
    funcs     = _parse_sass(sass_text)

    print(f"  kernels found in cubin: {', '.join(sorted(funcs.keys())) or '(none)'}")
    print()

    all_struct_ok = True
    for region in sensitive_regions:
        kernel = region["kernel"]
        buf    = region["buffer"]
        lines  = funcs.get(kernel)

        if lines is None:
            ok     = False
            detail = (f"{kernel}: FAIL — kernel not found in cubin "
                      f"(expected from sensitive_regions claim)")
        else:
            ok, detail = _verify_scrub(lines, kernel)

        print(f"  [{buf}] {detail}")
        results.append((f"struct_{kernel}_{buf}", ok, detail))
        if not ok:
            all_struct_ok = False

    print()

    # ── CHECK 3: kernel inventory completeness ────────────────────────────────
    print("── CHECK 3: Kernel inventory completeness ───────────────────")
    nominated_set = {r["kernel"] for r in sensitive_regions}
    non_nominated = sorted(set(funcs.keys()) - nominated_set)
    print(f"  kernels in cubin     : {', '.join(sorted(funcs.keys())) or '(none)'}")
    print(f"  kernels nominated    : {', '.join(sorted(nominated_set)) or '(none)'}")
    print(f"  kernels to inventory : {', '.join(non_nominated) or '(none — all accounted for)'}")

    source_file = witness.get("source_file")
    print(f"  source_file (witness): {source_file}")
    print(f"  NOTE: source_file CONTENT is now hash-protected by CHECK 1 (via")
    print(f"        source_file_hash in policy_doc). Residual gap: the source_file")
    print(f"        PATH is still witness-controlled. An attacker who regenerates")
    print(f"        the full witness (new Module 4 run, de-annotated source) can")
    print(f"        still defeat both CHECK 1 and CHECK 3 simultaneously.")
    print(f"        Taint confidence is bounded by source annotation integrity.")
    print()

    inv_ok, inv_details = _check_kernel_inventory(funcs, sensitive_regions, witness)
    for d in inv_details:
        print(f"  {d}")
    results.append(("kernel_inventory", inv_ok,
                    "; ".join(inv_details)))
    print()

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("── VERDICT ─────────────────────────────────────────────────")
    overall_ok = hash_ok and all_struct_ok and inv_ok

    if overall_ok:
        print("  PASS")
        print("  All three checks passed:")
        print("    ✓ SHA-384(cubin || policy) matches kernel_hash")
        print("    ✓ Every nominated kernel contains STS.RZ → MEMBAR.SC.CTA "
              "→ BAR.SYNC before EXIT")
        print("    ✓ Every non-nominated kernel has no confidential __shared__ "
              "buffers (taint-verified)")
    else:
        print("  FAIL")
        for label, ok, detail in results:
            if not ok:
                print(f"    ✗ {label}: {detail}")

    print("=" * 62)
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
