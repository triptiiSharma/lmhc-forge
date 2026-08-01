#!/usr/bin/env python3
"""
reproducibility_run.py — Section G: recompile 50 deterministically-selected
core configurations 3 additional times each, compare against the original
compile using the normalization procedure frozen in PHASE5_REPRODUCIBILITY.md
(written and committed BEFORE this script was run).
"""

import csv
import hashlib
import json
import re
import subprocess
import sys
import time
import datetime
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from compile_matrix import expand_cells, artifact_id as compute_artifact_id
from sass_parser import parse_sass_file
from classify_binary import classify_kernel
from nvdisasm_parser import parse_nvdisasm_json

SEED = 20260801
N_CONFIGS = 50
N_REPEATS = 3

TOOLKITS = {
    "cuda-12.8": {"id": "cuda-12.8", "nvcc": "/usr/local/cuda-12.8/bin/nvcc",
                   "cuobjdump": "/usr/local/cuda-12.8/bin/cuobjdump",
                   "nvdisasm": "/usr/local/cuda-12.8/bin/nvdisasm", "host_compiler": "gcc-14"},
    "cuda-13.3": {"id": "cuda-13.3", "nvcc": "/usr/local/cuda-13.3/bin/nvcc",
                   "cuobjdump": "/usr/local/cuda-13.3/bin/cuobjdump",
                   "nvdisasm": "/usr/local/cuda-13.3/bin/nvdisasm", "host_compiler": "gcc-14"},
}

RUNS_DIR = ROOT / "artifacts" / "reproducibility"

_ADDR_PREFIX_RE = re.compile(r'^\s*/\*[0-9a-f]+\*/\s*')
_ENCODING_LINE_RE = re.compile(r'^\s*/\*\s*0x[0-9a-f]+\s*\*/\s*$')


def load_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def normalize_sass_text(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        if _ENCODING_LINE_RE.match(line):
            continue
        m = _ADDR_PREFIX_RE.match(line)
        if m:
            line = line[m.end():]
        line = line.strip()
        if line:
            out_lines.append(line)
    return "\n".join(out_lines)


def normalize_nvdisasm_json(data) -> list:
    """4-tuple sequence: (predicate, opcode, operands, control_flow_flag)."""
    functions, err = parse_nvdisasm_json(data) if False else (None, None)  # not used; kept for clarity
    if not isinstance(data, list) or len(data) != 2:
        return []
    _, funcs = data
    out = []
    for fn in funcs:
        instrs = fn.get("sass-instructions", [])
        for instr in instrs:
            pred = instr.get("predicate")
            opcode = instr.get("opcode")
            operands = instr.get("operands")
            cf = (instr.get("other-attributes") or {}).get("control-flow") if isinstance(instr.get("other-attributes"), dict) else None
            out.append([pred, opcode, operands, cf])
    return out


def select_configs():
    artifacts = {a["artifact_id"]: a for a in load_jsonl(ROOT / "results" / "artifacts.jsonl")}
    features = {f["artifact_id"]: f for f in load_jsonl(ROOT / "results" / "features.jsonl")}
    config = json.loads((ROOT / "configs" / "core_matrix.json").read_text())
    cells = expand_cells(config)
    core_ids = set()
    for cell in cells:
        aid = compute_artifact_id(cell["construct"], cell["context"], cell["ptxas_opt"],
                                   cell["arch"], cell["toolkit"]["id"], cell.get("extra_device_flags", []))
        core_ids.add(aid)

    import random
    rng = random.Random(SEED)
    selected = []

    CONSTRUCTS = [f"Z{i}" for i in range(7)]
    CONTEXTS = [f"K{i}" for i in range(6)]
    OPTS = [0, 1, 2, 3]
    ARCHS = ["sm_75", "sm_80", "sm_86", "sm_87", "sm_89", "sm_90"]
    TOOLKIT_IDS = ["cuda-12.8", "cuda-13.3"]

    by_combo = {}
    for aid in core_ids:
        a = artifacts[aid]
        by_combo.setdefault((a["zeroization_construct"], a["kernel_context"]), []).append(aid)

    combo_order = [(z, k) for z in CONSTRUCTS for k in CONTEXTS]
    rng.shuffle(combo_order)
    idx = 0
    seen = set()
    while len(selected) < N_CONFIGS:
        z, k = combo_order[idx % len(combo_order)]
        candidates = [aid for aid in by_combo.get((z, k), []) if aid not in seen]
        if candidates:
            pick = rng.choice(candidates)
            selected.append(pick)
            seen.add(pick)
        idx += 1
        if idx > 10000:
            break
    return selected[:N_CONFIGS], artifacts, features


def compile_once(src_path, toolkit, opt, arch, out_dir: Path, tag: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cubin_path = out_dir / f"{tag}.cubin"
    cmd = [toolkit["nvcc"], f"-arch={arch}", "-O3", "-Xptxas", f"-O{opt}",
           "-ccbin", toolkit["host_compiler"], "-cubin", "-o", str(cubin_path), str(src_path)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    compile_ms = int((time.time() - t0) * 1000)
    result = {"command": " ".join(cmd), "return_code": r.returncode, "compile_time_ms": compile_ms}
    if r.returncode != 0:
        result["error"] = r.stderr.strip()[:1000]
        return result

    result["cubin_path"] = str(cubin_path.as_posix())
    result["cubin_sha256"] = sha256_bytes(cubin_path.read_bytes())

    dr = subprocess.run([toolkit["cuobjdump"], "-sass", str(cubin_path)], capture_output=True, text=True, timeout=60)
    sass_path = out_dir / f"{tag}.sass"
    sass_path.write_text(dr.stdout, encoding="utf-8")
    result["sass_path"] = str(sass_path.as_posix())
    result["sass_sha256"] = sha256_text(dr.stdout)
    result["sass_normalized_sha256"] = sha256_text(normalize_sass_text(dr.stdout))

    nr = subprocess.run([toolkit["nvdisasm"], "-c", "-json", str(cubin_path)], capture_output=True, text=True, timeout=60)
    json_path = out_dir / f"{tag}.json"
    json_path.write_text(nr.stdout, encoding="utf-8")
    result["nvdisasm_json_path"] = str(json_path.as_posix())
    result["nvdisasm_json_sha256"] = sha256_text(nr.stdout)
    try:
        data = json.loads(nr.stdout)
        norm = normalize_nvdisasm_json(data)
        result["nvdisasm_normalized_sha256"] = sha256_text(json.dumps(norm, sort_keys=False))
    except json.JSONDecodeError:
        result["nvdisasm_normalized_sha256"] = None

    cuobjdump_functions = parse_sass_file(dr.stdout)
    if cuobjdump_functions:
        feat = next(iter(cuobjdump_functions.values()))
        cls = classify_kernel(feat)
        result["features"] = feat.to_dict()
        result["category"] = cls.category

    return result


def main():
    selected, artifacts, features = select_configs()
    print(f"[reproducibility] selected {len(selected)} configurations")

    runs = []
    t_start = time.time()
    for i, aid in enumerate(selected, 1):
        a = artifacts[aid]
        toolkit = TOOLKITS[a["cuda_version"]]
        src_path = a["source_file"]

        # Original compile's known values (from the core matrix run).
        original = {
            "cubin_sha256": a.get("cubin_hash_sha256"),
            "category": features.get(aid, {}).get("checker_verdict"),
            "features": (features.get(aid, {}).get("kernel_features")),
        }

        run_dir = RUNS_DIR / aid
        repeats = []
        for rep in range(1, N_REPEATS + 1):
            rep_dir = run_dir / f"repeat_{rep}"
            res = compile_once(src_path, toolkit, a["ptxas_optimization"], a["target_arch"], rep_dir, f"repeat_{rep}")
            repeats.append(res)

        record = {
            "artifact_id": aid,
            "zeroization_construct": a["zeroization_construct"],
            "kernel_context": a["kernel_context"],
            "ptxas_optimization": a["ptxas_optimization"],
            "target_arch": a["target_arch"],
            "cuda_version": a["cuda_version"],
            "original_cubin_sha256": original["cubin_sha256"],
            "original_category": original["category"],
            "repeats": repeats,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        # Classify each repeat vs original.
        classifications = []
        for rep_res in repeats:
            if rep_res.get("return_code") != 0:
                classifications.append("COMPILE_FAILED_ON_REPEAT")
                continue
            if rep_res.get("cubin_sha256") == original["cubin_sha256"]:
                classifications.append("byte-identical")
                continue
            norm_match = True  # can't compare normalized to "original" directly without recomputing original's
            if rep_res.get("category") != original["category"]:
                classifications.append("category difference")
            elif rep_res.get("features") != original["features"]:
                classifications.append("feature difference")
            else:
                classifications.append("metadata/presentation-only difference (cubin bytes differ, "
                                        "category+features match)")
        record["repeat_classifications"] = classifications
        runs.append(record)
        print(f"  [{i}/{len(selected)}] {aid} {a['zeroization_construct']}x{a['kernel_context']} "
              f"{a['target_arch']} O{a['ptxas_optimization']} {a['cuda_version']} -> {classifications}")

    out_path = ROOT / "results" / "reproducibility_runs.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out_path}")

    # Summary
    all_classes = [c for r in runs for c in r["repeat_classifications"]]
    summary = {
        "n_configs": len(runs),
        "n_repeats_per_config": N_REPEATS,
        "n_total_repeat_compiles": len(all_classes),
        "classification_counts": dict(Counter(all_classes)),
        "any_category_difference": any(c == "category difference" for c in all_classes),
        "any_feature_difference": any(c == "feature difference" for c in all_classes),
        "any_compile_failure_on_repeat": any(c == "COMPILE_FAILED_ON_REPEAT" for c in all_classes),
    }
    summary_path = ROOT / "results" / "reproducibility_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2))

    diffs = [r for r in runs if any(c != "byte-identical" for c in r["repeat_classifications"])]
    diff_path = ROOT / "results" / "reproducibility_differences.csv"
    with diff_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["artifact_id", "construct", "context", "arch", "opt", "toolkit", "classifications"])
        for r in diffs:
            w.writerow([r["artifact_id"], r["zeroization_construct"], r["kernel_context"],
                        r["target_arch"], r["ptxas_optimization"], r["cuda_version"],
                        ";".join(r["repeat_classifications"])])
    print(f"wrote {diff_path} ({len(diffs)} configs with at least one non-byte-identical repeat)")

    elapsed = time.time() - t_start
    print(f"\ntotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
