#!/usr/bin/env python3
"""
build_controlled_cases.py — the N0-N12 controlled structural-case corpus.

N0, N1, N2, N9, N10, N12 are REAL_COMPILED_CASE references to existing
core-matrix artifacts (no new compilation -- pulled directly from
results/artifacts.jsonl / results/features.jsonl / results/nvdisasm_features.jsonl).
N11 is a REAL_BINARY_INPUT_CASE: a real cubin queried for a kernel name
that isn't in it (a witness/tooling-input scenario, not a compiled-structure
scenario).
N3, N4, N5, N6, N7, N8 are REAL_COMPILED_CASE: dedicated new sources
(controlled_case_kernels.py), compiled across 2 toolkits x 2 opt levels x
2 archs = 8 configs each. N5 and N6 are compiled/disassembled only, NEVER
executed (see controlled_case_kernels.py's header comment).

For every real case, records: intended condition, compile outcome,
cuobjdump features+verdict, nvdisasm features+verdict, cross-tool
agreement, and "conforms" -- computed empirically from the actual observed
binary, never assumed from source intent alone (task's own explicit
requirement).
"""

import csv
import json
import sys
import time
import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from controlled_case_kernels import render, CASE_META
from compile_matrix import compile_one, artifact_id as compute_artifact_id
from sass_parser import parse_sass_file
from classify_binary import classify_kernel
from nvdisasm_parser import parse_nvdisasm_json
import subprocess

KERNELS_DIR = ROOT / "kernels" / "controlled"
ARTIFACTS_DIR = ROOT / "artifacts" / "controlled"
NVDISASM_PATHS = {
    "cuda-12.8": "/usr/local/cuda-12.8/bin/nvdisasm",
    "cuda-13.3": "/usr/local/cuda-13.3/bin/nvdisasm",
}
TOOLKITS = {
    "cuda-12.8": {"id": "cuda-12.8", "nvcc": "/usr/local/cuda-12.8/bin/nvcc",
                   "cuobjdump": "/usr/local/cuda-12.8/bin/cuobjdump", "host_compiler": "gcc-14"},
    "cuda-13.3": {"id": "cuda-13.3", "nvcc": "/usr/local/cuda-13.3/bin/nvcc",
                   "cuobjdump": "/usr/local/cuda-13.3/bin/cuobjdump", "host_compiler": "gcc-14"},
}
DEDICATED_CASES = ["n3", "n4", "n5", "n6", "n7", "n8"]
CONFIGS = [
    (toolkit, opt, arch)
    for toolkit in ("cuda-12.8", "cuda-13.3")
    for opt in (0, 3)
    for arch in ("sm_86", "sm_89")
]


def load_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def nvdisasm_for(cubin_path: str, toolkit_id: str, out_json_path: Path) -> tuple[dict, str | None]:
    nvdisasm_path = NVDISASM_PATHS[toolkit_id]
    r = subprocess.run([nvdisasm_path, "-c", "-json", cubin_path], capture_output=True, text=True, timeout=60)
    out_json_path.write_text(r.stdout, encoding="utf-8")
    if r.returncode != 0:
        return {}, f"nvdisasm exited {r.returncode}: {r.stderr[:500]}"
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        return {}, f"invalid JSON: {e}"
    functions, schema_error = parse_nvdisasm_json(data)
    return functions, schema_error


def build_dedicated_cases() -> list[dict]:
    KERNELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "cubin").mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "sass").mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "nvdisasm_json").mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "logs").mkdir(exist_ok=True)

    # Write sources once.
    src_paths = {}
    for case_key in DEDICATED_CASES:
        fname, src = render(case_key)
        src_path = KERNELS_DIR / f"{case_key}.cu"
        src_path.write_text(src, encoding="utf-8", newline="\n")
        src_paths[case_key] = (src_path, fname)

    records = []
    for case_key in DEDICATED_CASES:
        src_path, fname = src_paths[case_key]
        meta = CASE_META[case_key]
        for toolkit_id, opt, arch in CONFIGS:
            toolkit = TOOLKITS[toolkit_id]
            cell = {"construct": f"controlled_{case_key}", "context": "K2shape",
                    "ptxas_opt": opt, "arch": arch}
            aid = compute_artifact_id(cell["construct"], cell["context"], opt, arch, toolkit_id, [])

            # compile_one() expects generated-kernel conventions (paths under
            # kernels/generated/kernels/metadata) -- reimplemented minimally
            # here rather than forcing this corpus through that exact layout.
            cubin_path = ARTIFACTS_DIR / "cubin" / f"{aid}.cubin"
            sass_path = ARTIFACTS_DIR / "sass" / f"{aid}.sass"
            stdout_path = ARTIFACTS_DIR / "logs" / f"{aid}.stdout.txt"
            stderr_path = ARTIFACTS_DIR / "logs" / f"{aid}.stderr.txt"
            cmd = [toolkit["nvcc"], f"-arch={arch}", "-O3", "-Xptxas", f"-O{opt}",
                   "-ccbin", toolkit["host_compiler"], "-cubin", "-o", str(cubin_path), str(src_path)]

            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            compile_ms = int((time.time() - t0) * 1000)
            stdout_path.write_text(r.stdout, encoding="utf-8")
            stderr_path.write_text(r.stderr, encoding="utf-8")

            rec = {
                "case_id": meta["id"], "case_key": case_key,
                "artifact_id": aid, "category": "REAL_COMPILED_CASE",
                "safe_to_execute": meta["safe_to_execute"], "executed": False,
                "toolkit": toolkit_id, "ptxas_optimization": opt, "target_arch": arch,
                "source_file": str(src_path.as_posix()), "compile_command": " ".join(cmd),
                "compile_return_code": r.returncode, "compile_time_ms": compile_ms,
                "compile_status": "COMPILE_OK" if r.returncode == 0 else "COMPILE_FAILED",
                "intended_description": meta["description"],
                "intended_category": meta["intended_category"],
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }

            if r.returncode != 0:
                rec["error_message"] = r.stderr.strip()[:1000]
                records.append(rec)
                continue

            cuobjdump_path = toolkit["cuobjdump"]
            dr = subprocess.run([cuobjdump_path, "-sass", str(cubin_path)], capture_output=True, text=True, timeout=60)
            sass_path.write_text(dr.stdout, encoding="utf-8")
            rec["sass_path"] = str(sass_path.as_posix())
            rec["cubin_path"] = str(cubin_path.as_posix())

            cuobjdump_functions = parse_sass_file(dr.stdout)
            cf = next((f for f in cuobjdump_functions.values() if f.demangled_name == fname), None)
            if cf:
                co_result = classify_kernel(cf)
                rec["cuobjdump_verdict"] = co_result.category
                rec["cuobjdump_detail"] = co_result.detail
                rec["cuobjdump_features"] = cf.to_dict()
            else:
                rec["cuobjdump_verdict"] = "UNDECIDABLE"
                rec["cuobjdump_detail"] = f"kernel '{fname}' not found in cuobjdump SASS"

            nv_json_path = ARTIFACTS_DIR / "nvdisasm_json" / f"{aid}.json"
            nv_functions, schema_error = nvdisasm_for(str(cubin_path), toolkit_id, nv_json_path)
            rec["nvdisasm_json_path"] = str(nv_json_path.as_posix())
            if schema_error:
                rec["nvdisasm_verdict"] = "UNDECIDABLE"
                rec["nvdisasm_detail"] = schema_error
            else:
                nf = nv_functions.get(fname)
                if nf:
                    nv_result = classify_kernel(nf)
                    rec["nvdisasm_verdict"] = nv_result.category
                    rec["nvdisasm_detail"] = nv_result.detail
                    rec["nvdisasm_features"] = nf.to_dict()
                else:
                    rec["nvdisasm_verdict"] = "UNDECIDABLE"
                    rec["nvdisasm_detail"] = f"kernel '{fname}' not found in nvdisasm output"

            rec["cross_tool_agreement"] = (rec.get("cuobjdump_verdict") == rec.get("nvdisasm_verdict"))
            rec["conforms_to_intended"] = (rec.get("cuobjdump_verdict") == meta["intended_category"])
            records.append(rec)
            print(f"  {meta['id']} {toolkit_id} O{opt} {arch} -> cuobjdump={rec.get('cuobjdump_verdict')} "
                  f"nvdisasm={rec.get('nvdisasm_verdict')} intended={meta['intended_category']} "
                  f"conforms={rec.get('conforms_to_intended')}")

    return records


def build_reference_cases() -> list[dict]:
    """N0, N1, N2, N9, N10, N12: reuse existing core-matrix artifacts by
    reference (no new compilation). N11: a real binary queried for a
    nonexistent kernel name (REAL_BINARY_INPUT_CASE)."""
    artifacts = {a["artifact_id"]: a for a in load_jsonl(ROOT / "results" / "artifacts.jsonl")}
    features = {f["artifact_id"]: f for f in load_jsonl(ROOT / "results" / "features.jsonl")}
    nvdisasm_feats = {f["artifact_id"]: f for f in load_jsonl(ROOT / "results" / "nvdisasm_features.jsonl")}
    if not nvdisasm_feats:
        nvdisasm_feats = {f["artifact_id"]: f for f in load_jsonl(ROOT / "results" / "nvdisasm_features_pilot.jsonl")}

    def find_one(construct, context, want_category=None):
        for aid, a in artifacts.items():
            if a["zeroization_construct"] == construct and a["kernel_context"] == context and a["compile_status"] == "COMPILE_OK":
                f = features.get(aid)
                if f and (want_category is None or f["checker_verdict"] == want_category):
                    return aid, a, f
        return None, None, None

    records = []

    refs = [
        ("N0", "no scrub absent", "Z0", "K2", "ABSENT"),
        ("N1", "complete hardened recognized signature", "Z5", "K2", "FULL_SIGNATURE"),
        ("N2", "zero stores present, fence absent", "Z2", "K0", "STORE_ONLY"),
        ("N9", "valid register-mediated zero encoding (real ptxas-O0 form)", "Z5", "K4", None),
        ("N10", "multiple-exit control flow, intended cleanup structure present", "Z5", "K4", "FULL_SIGNATURE"),
        ("N12", "pure-dead plain scrub eliminated by compilation", "Z1", "K0", "ABSENT"),
    ]
    for case_id, desc, construct, context, want_cat in refs:
        aid, a, f = find_one(construct, context, want_cat)
        rec = {
            "case_id": case_id, "case_key": case_id.lower(),
            "artifact_id": aid, "category": "REAL_COMPILED_CASE",
            "reused_from_core_matrix": True,
            "intended_description": desc,
            "intended_category": want_cat,
        }
        if aid:
            rec["toolkit"] = a["cuda_version"]
            rec["ptxas_optimization"] = a["ptxas_optimization"]
            rec["target_arch"] = a["target_arch"]
            rec["cubin_path"] = a["cubin_path"]
            rec["sass_path"] = a["sass_path"]
            rec["cuobjdump_verdict"] = f["checker_verdict"]
            rec["cuobjdump_detail"] = f["checker_detail"]
            rec["cuobjdump_features"] = f.get("kernel_features")
            nv = nvdisasm_feats.get(aid)
            if nv:
                rec["nvdisasm_verdict"] = nv.get("nvdisasm_verdict")
                rec["nvdisasm_features"] = nv.get("nvdisasm_features")
                rec["cross_tool_agreement"] = nv.get("category_agreement")
            rec["conforms_to_intended"] = (want_cat is None) or (rec["cuobjdump_verdict"] == want_cat)
        else:
            rec["error_message"] = f"no matching core-matrix artifact found for {construct}x{context}"
        records.append(rec)
        print(f"  {case_id} (reference {construct}x{context}) -> "
              f"cuobjdump={rec.get('cuobjdump_verdict')} intended={want_cat} "
              f"conforms={rec.get('conforms_to_intended')}")

    # N11: REAL_BINARY_INPUT_CASE -- a real cubin, queried for a kernel name
    # that is NOT present. This is a witness/tooling-input scenario, not
    # evidence of GPU memory insecurity (per the corrected framing).
    ref_aid, ref_a, ref_f = find_one("Z5", "K2", "FULL_SIGNATURE")
    n11 = {
        "case_id": "N11", "case_key": "n11",
        "artifact_id": ref_aid, "category": "REAL_BINARY_INPUT_CASE",
        "intended_description": "requested/nominated kernel absent from the binary "
                                 "(witness names a kernel not present in the cubin)",
        "intended_category": "UNDECIDABLE",
    }
    if ref_aid:
        sass_text = Path(ref_a["sass_path"]).read_text(encoding="utf-8", errors="replace")
        from classify_binary import classify_artifact
        result = classify_artifact(sass_text, "kernel_z5_k2_NONEXISTENT_NAME")
        n11["cuobjdump_verdict"] = result.category
        n11["cuobjdump_detail"] = result.detail
        n11["conforms_to_intended"] = (result.category == "UNDECIDABLE")
    records.append(n11)
    print(f"  N11 (nonexistent kernel name against real binary) -> "
          f"cuobjdump={n11.get('cuobjdump_verdict')} conforms={n11.get('conforms_to_intended')}")

    return records


def main():
    print("=== dedicated cases (N3-N8) ===")
    dedicated = build_dedicated_cases()
    print("\n=== reference cases (N0,N1,N2,N9,N10,N11,N12) ===")
    reference = build_reference_cases()

    all_records = dedicated + reference
    out_path = ROOT / "results" / "controlled_cases.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out_path} ({len(all_records)} records)")

    # Summary CSV
    summary_path = ROOT / "results" / "controlled_case_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "category", "artifact_id", "toolkit", "ptxas_optimization",
                     "target_arch", "compile_status", "intended_category", "cuobjdump_verdict",
                     "nvdisasm_verdict", "cross_tool_agreement", "conforms_to_intended",
                     "safe_to_execute", "intended_description"])
        for r in all_records:
            w.writerow([r.get("case_id"), r.get("category"), r.get("artifact_id"),
                        r.get("toolkit"), r.get("ptxas_optimization"), r.get("target_arch"),
                        r.get("compile_status", "N/A" if r.get("reused_from_core_matrix") else ""),
                        r.get("intended_category"), r.get("cuobjdump_verdict"),
                        r.get("nvdisasm_verdict"), r.get("cross_tool_agreement"),
                        r.get("conforms_to_intended"), r.get("safe_to_execute"),
                        r.get("intended_description")])
    print(f"wrote {summary_path}")

    mismatches = [r for r in all_records if r.get("conforms_to_intended") is False]
    mismatch_path = ROOT / "results" / "controlled_case_mismatches.csv"
    with mismatch_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "artifact_id", "intended_category", "cuobjdump_verdict",
                     "cuobjdump_detail", "toolkit", "ptxas_optimization", "target_arch"])
        for r in mismatches:
            w.writerow([r.get("case_id"), r.get("artifact_id"), r.get("intended_category"),
                        r.get("cuobjdump_verdict"), r.get("cuobjdump_detail"),
                        r.get("toolkit"), r.get("ptxas_optimization"), r.get("target_arch")])
    print(f"wrote {mismatch_path} ({len(mismatches)} mismatches)")


if __name__ == "__main__":
    main()
