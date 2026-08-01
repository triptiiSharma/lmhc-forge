#!/usr/bin/env python3
"""
run_cross_tool_validation.py — runs the matching toolkit's nvdisasm over
selected core artifacts, extracts independent features via
nvdisasm_parser.py, classifies with the SAME shared classification policy
used for cuobjdump (classify_binary.classify_kernel -- the policy is the
study's codebook, PLAN.md §5, deliberately identical across both parsing
paths so a category comparison is meaningful; only the feature EXTRACTION
is independent), and compares against the existing cuobjdump-derived
results in results/features.jsonl.

Terminology (frozen per the approved plan): this produces "cross-tool
category/feature agreement", never "accuracy" -- nvdisasm and cuobjdump
both interpret the same NVIDIA cubin, so agreement between them is
evidence the primary parser isn't idiosyncratically wrong, not independent
semantic ground truth.

Usage:
    python3 run_cross_tool_validation.py --ids-file pilot_ids.txt --out-suffix pilot
    python3 run_cross_tool_validation.py --all-core --out-suffix full
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from nvdisasm_parser import parse_nvdisasm_json, NvdisasmKernelFeatures
from classify_binary import classify_kernel
from compile_matrix import expand_cells, artifact_id as compute_artifact_id

NVDISASM_PATHS = {
    "cuda-12.8": "/usr/local/cuda-12.8/bin/nvdisasm",
    "cuda-13.3": "/usr/local/cuda-13.3/bin/nvdisasm",
}
JSON_DIR = ROOT / "artifacts" / "nvdisasm_json"


def load_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def get_nvdisasm_version(nvdisasm_path: str) -> str:
    r = subprocess.run([nvdisasm_path, "--version"], capture_output=True, text=True, timeout=15)
    return r.stdout.strip().replace("\n", " | ")


def run_one(artifact: dict, cuobjdump_feat: dict | None, out_dir: Path, timeout_s: int = 60) -> dict:
    aid = artifact["artifact_id"]
    toolkit_id = artifact["cuda_version"]
    nvdisasm_path = NVDISASM_PATHS.get(toolkit_id)
    cubin_path = artifact.get("cubin_path")

    record = {
        "artifact_id": aid,
        "cuda_version": toolkit_id,
        "nvdisasm_path": nvdisasm_path,
        "nvdisasm_version": None,
        "command": None,
        "return_code": None,
        "stderr": None,
        "raw_json_path": None,
        "schema_ok": None,
        "schema_error": None,
        "nvdisasm_features": None,
        "nvdisasm_verdict": None,
        "nvdisasm_detail": None,
        "cuobjdump_verdict": cuobjdump_feat.get("checker_verdict") if cuobjdump_feat else None,
        "category_agreement": None,
        "feature_agreement": {},
        "analysis_time_ms": None,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    if not nvdisasm_path:
        record["schema_error"] = f"no nvdisasm mapping for toolkit '{toolkit_id}'"
        return record
    if not cubin_path or not Path(cubin_path).exists():
        record["schema_error"] = "cubin_path missing or does not exist"
        return record

    record["nvdisasm_version"] = get_nvdisasm_version(nvdisasm_path)

    cmd = [nvdisasm_path, "-c", "-json", cubin_path]
    record["command"] = " ".join(cmd)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        record["return_code"] = None
        record["stderr"] = f"TIMEOUT after {timeout_s}s"
        return record

    record["return_code"] = r.returncode
    record["stderr"] = r.stderr.strip()[:2000] if r.stderr else None

    json_path = out_dir / f"{aid}.json"
    json_path.write_text(r.stdout, encoding="utf-8")
    record["raw_json_path"] = str(json_path.as_posix())

    if r.returncode != 0:
        record["schema_error"] = f"nvdisasm exited {r.returncode}"
        return record

    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        record["schema_error"] = f"invalid JSON: {e}"
        return record

    functions, schema_error = parse_nvdisasm_json(data)
    if schema_error:
        record["schema_ok"] = False
        record["schema_error"] = schema_error
        return record
    record["schema_ok"] = True

    target_name = artifact["kernel_demangled_name"]
    match = functions.get(target_name)
    if match is None:
        record["nvdisasm_verdict"] = "UNDECIDABLE"
        record["nvdisasm_detail"] = (
            f"kernel '{target_name}' not found in nvdisasm output "
            f"(found: {sorted(functions.keys())})"
        )
    else:
        result = classify_kernel(match)
        record["nvdisasm_verdict"] = result.category
        record["nvdisasm_detail"] = result.detail
        record["nvdisasm_features"] = match.to_dict()

    record["analysis_time_ms"] = int((time.time() - t0) * 1000)

    if cuobjdump_feat and record["nvdisasm_verdict"]:
        record["category_agreement"] = (record["nvdisasm_verdict"] == cuobjdump_feat.get("checker_verdict"))
        cf = cuobjdump_feat.get("kernel_features") or {}
        nf = record["nvdisasm_features"] or {}
        for key in ("zero_shared_store_direct_count", "zero_shared_store_indirect_count",
                    "zero_shared_store_count", "membar_count", "membar_cta_count",
                    "bar_sync_count", "exit_count", "branch_count"):
            record["feature_agreement"][key] = (cf.get(key) == nf.get(key))

    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", help="file with one artifact_id per line")
    ap.add_argument("--all-core", action="store_true", help="run over all 2016 core artifacts")
    ap.add_argument("--out", default=str(ROOT / "results" / "nvdisasm_features.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    JSON_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = {a["artifact_id"]: a for a in load_jsonl(ROOT / "results" / "artifacts.jsonl")}
    cuobjdump_features = {f["artifact_id"]: f for f in load_jsonl(ROOT / "results" / "features.jsonl")}

    if args.all_core:
        config = json.loads((ROOT / "configs" / "core_matrix.json").read_text())
        cells = expand_cells(config)
        target_ids = set()
        for cell in cells:
            aid = compute_artifact_id(cell["construct"], cell["context"], cell["ptxas_opt"],
                                       cell["arch"], cell["toolkit"]["id"], cell.get("extra_device_flags", []))
            target_ids.add(aid)
        target_ids = sorted(target_ids)
    elif args.ids_file:
        target_ids = [l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()]
    else:
        print("ERROR: specify --ids-file or --all-core", file=sys.stderr)
        sys.exit(1)

    if args.limit:
        target_ids = target_ids[: args.limit]

    out_path = Path(args.out)
    done_ids = {r["artifact_id"] for r in load_jsonl(out_path)}
    pending = [i for i in target_ids if i not in done_ids]
    print(f"[cross_tool] {len(target_ids)} target, {len(done_ids)} already done, {len(pending)} pending")

    t_start = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for i, aid in enumerate(pending, 1):
            artifact = artifacts.get(aid)
            if not artifact:
                print(f"  [{i}/{len(pending)}] {aid}: NOT FOUND in artifacts.jsonl, skipping")
                continue
            cf = cuobjdump_features.get(aid)
            rec = run_one(artifact, cf, JSON_DIR)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"  [{i}/{len(pending)}] {aid} {artifact['zeroization_construct']}x"
                  f"{artifact['kernel_context']} {artifact['target_arch']} "
                  f"O{artifact['ptxas_optimization']} {artifact['cuda_version']} -> "
                  f"nvdisasm={rec['nvdisasm_verdict']} cuobjdump={rec['cuobjdump_verdict']} "
                  f"agree={rec['category_agreement']}")

    elapsed = time.time() - t_start
    print(f"\n[cross_tool] {len(pending)} processed in {elapsed:.1f}s, wrote {out_path}")


if __name__ == "__main__":
    main()
