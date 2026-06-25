#!/usr/bin/env python3
"""
tamper_tests.py -- T1 through T6 against the unmodified attester.py per run.

T1: Stripped cubin + original witness                         -> expect FAIL (hash mismatch)
T2: Stripped cubin + regenerated hash, gemm still nominated  -> expect CHECK1 PASS, CHECK2 FAIL
T3: Original cubin + original witness                        -> expect PASS
T4: Stripped cubin + honest hash, gemm DROPPED from regions  -> CHECK3 FAIL (unaccounted kernel)
T5: Stripped cubin + de-annotated source + no source_file_hash in policy
    Before fix: PASS (hole -- CHECK3 fooled by de-annotated source, CHECK1 misses missing hash)
    After fix : FAIL on CHECK1 (verifier includes source_file_hash; old-format kernel_hash doesn't)
T6: Generate-from-scratch -- adversary runs updated witness_gen.py on de-annotated
    ORIGINAL source.  Produces fully internally-consistent new-format witness.
    Expected: PASS (KNOWN LIMITATION -- only binary-level taint closes this)
"""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT        = Path("/opt/lmhc-forge")
NVCC        = "/usr/local/cuda-12.8/bin/nvcc"
COBJD       = "/usr/local/cuda-12.8/bin/cuobjdump"
ATTESTER    = ROOT / "src/attester.py"
WITNESS_GEN = ROOT / "src/witness_gen.py"
ORIG_CU     = Path("/tmp/test_kernels_m4_lmhc.cu")    # M2-rewritten source (for T1-T5)
ORIG_SOURCE = ROOT / "src/test_kernels.cu"              # original pre-M2 source (for T6)
ORIG_CUBIN  = Path("/tmp/test_kernels.cubin")
ORIG_WIT    = Path("/tmp/witness.json")

# ── Source manipulation helpers ───────────────────────────────────────────────

def strip_gemm_scrub(source):
    """Remove the LMHC scrub block for a_tile from gemm_tile."""
    out, in_scrub = [], False
    for line in source.splitlines(keepends=True):
        if "BEGIN LMHC scrub" in line and "a_tile" in line:
            in_scrub = True
        if not in_scrub:
            out.append(line)
        if "END LMHC scrub" in line and "a_tile" in line:
            in_scrub = False
    return "".join(out)


def deannotate_gemm(source):
    """Remove __confidential__ from gemm_tile's A parameter.
    Taint pass will no longer flag a_tile as confidential.
    attention_fwd's q_proj annotation is left intact.
    """
    old = "    __confidential__ const float *A,   // confidential weight matrix"
    new = "    const float *A,"
    if old not in source:
        sys.exit("ERROR: deannotation target not found in source")
    return source.replace(old, new, 1)


# ── Compile ───────────────────────────────────────────────────────────────────

def compile_cubin(source, label):
    with tempfile.NamedTemporaryFile(
            suffix=f"_{label}.cu", delete=False, mode="w", encoding="utf-8") as tf:
        tf.write(source)
        cu = Path(tf.name)
    out = Path(tempfile.mktemp(suffix=f"_{label}.cubin"))
    try:
        r = subprocess.run(
            [NVCC, "--cubin", "-arch=sm_89", "-O3",
             "-ccbin", "/usr/bin/gcc-14", "-o", str(out), str(cu)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            sys.exit(f"nvcc failed:\n{r.stderr}")
        return out.read_bytes()
    finally:
        cu.unlink(missing_ok=True)
        out.unlink(missing_ok=True)


# ── Hash (mirrors attester.py / witness_gen.py) ───────────────────────────────

def compute_hash(cubin_bytes, policy_doc):
    """SHA-384(cubin_bytes || compact-JSON(policy_doc))."""
    pb = json.dumps(policy_doc, sort_keys=True, separators=(",", ":")).encode()
    h  = hashlib.sha384()
    h.update(cubin_bytes)
    h.update(pb)
    return h.hexdigest()


def make_policy_doc(w, include_source_hash=True):
    """Reconstruct policy_doc from witness fields.

    include_source_hash=True  (default): include source_file_hash when the
        updated Module 4 has written it to the witness. Used for T1-T4.
    include_source_hash=False: omit source_file_hash regardless. Used for T5
        to simulate a pre-fix attacker's witness that was generated without
        the source hash in the policy_doc.
    """
    d = {
        "all_exits_protected": w["all_exits_protected"],
        "arch":                w["arch"],
        "compile_flags":       w["compile_flags"],
        "policy_version":      w["policy_version"],
        "scrub_sites":         w["scrub_sites"],
        "sensitive_regions":   w["sensitive_regions"],
    }
    if include_source_hash and "source_file_hash" in w:
        d["source_file_hash"] = w["source_file_hash"]
    return d


# ── Run verifier ─────────────────────────────────────────────────────────────

def run_verifier(cubin_path, witness_path, label):
    sep = "=" * 72
    print(f"\n{sep}")
    print(label)
    print(sep)
    r = subprocess.run(
        ["python3", str(ATTESTER),
         str(cubin_path), str(witness_path),
         "--cuobjdump", COBJD],
        capture_output=True, text=True
    )
    print(r.stdout, end="")
    if r.stderr.strip():
        print(r.stderr, end="")
    print(f"[exit {r.returncode}]")
    return r.returncode


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Building stripped cubin (gemm_tile.a_tile scrub removed) ...")
    orig_src      = ORIG_CU.read_text(encoding="utf-8")
    stripped_src  = strip_gemm_scrub(orig_src)
    if stripped_src == orig_src:
        sys.exit("ERROR: stripping failed -- source unchanged")

    orig_bytes    = ORIG_CUBIN.read_bytes()
    stripped_bytes = compile_cubin(stripped_src, "stripped")
    stripped_path  = Path("/tmp/test_kernels_stripped.cubin")
    stripped_path.write_bytes(stripped_bytes)

    print(f"  original cubin : {len(orig_bytes):>6} bytes")
    print(f"  stripped cubin : {len(stripped_bytes):>6} bytes")
    print(f"  cubins differ  : {orig_bytes != stripped_bytes}")

    # De-annotated source (T5): remove __confidential__ from gemm_tile.A
    print("\nBuilding de-annotated source (gemm_tile.__confidential__ removed) ...")
    deannotated_src  = deannotate_gemm(orig_src)
    deannotated_path = Path("/tmp/test_kernels_deannotated.cu")
    deannotated_path.write_text(deannotated_src, encoding="utf-8")
    print(f"  written to {deannotated_path}")

    # Load original witness
    orig_wit = json.loads(ORIG_WIT.read_text(encoding="utf-8"))
    orig_pol = make_policy_doc(orig_wit)

    # ── T2: regenerated hash, gemm still in sensitive_regions ─────────────────
    t2_pol  = copy.deepcopy(orig_pol)
    t2_hash = compute_hash(stripped_bytes, t2_pol)
    t2_wit  = copy.deepcopy(orig_wit)
    t2_wit["kernel_hash"]      = t2_hash
    t2_wit["cubin_size_bytes"] = len(stripped_bytes)
    t2_path = Path("/tmp/witness_T2.json")
    t2_path.write_text(json.dumps(t2_wit, indent=2), encoding="utf-8")

    # ── T4: gemm DROPPED, honest hash, old source_file ────────────────────────
    t4_pol = copy.deepcopy(orig_pol)
    t4_pol["sensitive_regions"] = [
        r for r in orig_pol["sensitive_regions"] if r["kernel"] != "gemm_tile"
    ]
    t4_pol["scrub_sites"] = [
        s for s in orig_pol["scrub_sites"] if s["kernel"] != "gemm_tile"
    ]
    t4_hash = compute_hash(stripped_bytes, t4_pol)
    t4_wit  = copy.deepcopy(orig_wit)
    t4_wit["kernel_hash"]         = t4_hash
    t4_wit["cubin_size_bytes"]    = len(stripped_bytes)
    t4_wit["sensitive_regions"]   = t4_pol["sensitive_regions"]
    t4_wit["scrub_sites"]         = t4_pol["scrub_sites"]
    t4_wit["all_exits_protected"] = True
    t4_path = Path("/tmp/witness_T4.json")
    t4_path.write_text(json.dumps(t4_wit, indent=2), encoding="utf-8")

    # ── T5: stripped cubin + de-annotated source, old-format kernel_hash ──────
    # policy_doc DELIBERATELY excludes source_file_hash -- simulates a witness
    # generated before the fix (when witness_gen.py did not hash the source).
    # CHECK 3 reads the de-annotated source and clears gemm_tile.
    # Before fix: PASS (hole). After fix: FAIL (verifier adds source_file_hash
    # to CHECK 1 reconstruction; old-format hash doesn't include it -> mismatch).
    t5_pol = {
        "all_exits_protected": True,
        "arch":           orig_pol["arch"],
        "compile_flags":  orig_pol["compile_flags"],
        "policy_version": orig_pol["policy_version"],
        "scrub_sites": [
            s for s in orig_pol["scrub_sites"] if s["kernel"] != "gemm_tile"
        ],
        "sensitive_regions": [
            r for r in orig_pol["sensitive_regions"] if r["kernel"] != "gemm_tile"
        ],
        # NO source_file_hash -- this is the pre-fix omission being demonstrated
    }
    t5_hash = compute_hash(stripped_bytes, t5_pol)
    t5_wit  = copy.deepcopy(orig_wit)
    t5_wit["kernel_hash"]         = t5_hash
    t5_wit["cubin_size_bytes"]    = len(stripped_bytes)
    t5_wit["sensitive_regions"]   = t5_pol["sensitive_regions"]
    t5_wit["scrub_sites"]         = t5_pol["scrub_sites"]
    t5_wit["all_exits_protected"] = True
    t5_wit["source_file"]         = str(deannotated_path)
    # Remove source_file_hash from witness top-level too (not in old format)
    t5_wit.pop("source_file_hash", None)
    t5_path = Path("/tmp/witness_T5.json")
    t5_path.write_text(json.dumps(t5_wit, indent=2), encoding="utf-8")

    print(f"\nT2 hash : {t2_hash[:32]}...")
    sr2 = [r["kernel"] + "." + r["buffer"] for r in t2_wit["sensitive_regions"]]
    print(f"  sensitive_regions : {sr2}")

    print(f"\nT4 hash : {t4_hash[:32]}...")
    sr4 = [r["kernel"] + "." + r["buffer"] for r in t4_wit["sensitive_regions"]]
    print(f"  sensitive_regions : {sr4}")

    print(f"\nT5 hash : {t5_hash[:32]}... (old-format: NO source_file_hash in policy)")
    sr5 = [r["kernel"] + "." + r["buffer"] for r in t5_wit["sensitive_regions"]]
    print(f"  sensitive_regions : {sr5}")
    print(f"  source_file       : {t5_wit['source_file']}")
    has_sfh = "source_file_hash" in t5_wit
    print(f"  source_file_hash in witness : {has_sfh}")

    # ── T6: generate-from-scratch using updated witness_gen.py ───────────────
    # Adversary reads the ORIGINAL (pre-M2) source, removes __confidential__,
    # then runs the full witness generation pipeline on it.
    # witness_gen.py will:
    #   1. taint: find only attention_fwd.smem_qk (gemm_tile.a_tile no longer tainted)
    #   2. rewrite: insert scrub only for attention_fwd (gemm_tile gets none)
    #   3. compile: cubin has no gemm_tile scrub -- naturally stripped
    #   4. hash = SHA-384(T6_cubin || policy_with_deannotated_source_hash)
    # The verifier reads the same de-annotated source, independently recomputes
    # the source_file_hash, and all three checks pass.
    print("\nT6: Running witness_gen.py on de-annotated original source ...")
    orig_source_text = ORIG_SOURCE.read_text(encoding="utf-8")
    deannotated_orig = deannotate_gemm(orig_source_text)
    t6_input_path    = Path("/tmp/test_kernels_t6_input.cu")
    t6_input_path.write_text(deannotated_orig, encoding="utf-8")
    t6_cubin_path    = Path("/tmp/test_kernels_T6.cubin")
    t6_wit_path      = Path("/tmp/witness_T6.json")

    r6 = subprocess.run(
        ["python3", str(WITNESS_GEN),
         str(t6_input_path),
         "--arch", "sm_89",
         "--nvcc", NVCC,
         "--ccbin", "/usr/bin/gcc-14",
         "--output", str(t6_wit_path),
         "--save-cubin", str(t6_cubin_path)],
        capture_output=True, text=True
    )
    if r6.returncode != 0:
        print(f"[T6] witness_gen.py FAILED:\n{r6.stderr}")
        sys.exit(1)
    print("[T6] witness_gen.py output (stderr):")
    for line in r6.stderr.strip().splitlines():
        print(f"  {line}")
    t6_wit = json.loads(t6_wit_path.read_text(encoding="utf-8"))
    print(f"[T6] cubin size   : {t6_cubin_path.stat().st_size} bytes")
    print(f"[T6] kernel_hash  : {t6_wit['kernel_hash'][:32]}...")
    sr6 = [r["kernel"] + "." + r["buffer"] for r in t6_wit["sensitive_regions"]]
    print(f"[T6] sensitive_regions : {sr6}")
    print(f"[T6] source_file  : {t6_wit['source_file']}")
    print(f"[T6] source_file_hash : {t6_wit.get('source_file_hash','<missing>')[:32]}...")

    # ── Run all six tests ─────────────────────────────────────────────────────
    run_verifier(stripped_path, ORIG_WIT,
        "T1 -- Stripped cubin + ORIGINAL witness  (expect FAIL: hash mismatch)")
    run_verifier(stripped_path, t2_path,
        "T2 -- Stripped cubin + REGENERATED witness / gemm still nominated"
        "  (expect CHECK1 PASS, CHECK2 FAIL)")
    run_verifier(ORIG_CUBIN, ORIG_WIT,
        "T3 -- Original cubin + original witness  (expect PASS)")
    run_verifier(stripped_path, t4_path,
        "T4 -- Stripped cubin + witness DROPS gemm  (expect CHECK3 FAIL)")
    run_verifier(stripped_path, t5_path,
        "T5 -- Stripped cubin + de-annotated source + old-format witness"
        "  (before fix: PASS=hole; after fix: FAIL on CHECK1)")
    run_verifier(t6_cubin_path, t6_wit_path,
        "T6 -- Generate-from-scratch: updated witness_gen.py on de-annotated source"
        "  (KNOWN LIMITATION: expect PASS -- binary-level taint required to close)")


if __name__ == "__main__":
    main()
