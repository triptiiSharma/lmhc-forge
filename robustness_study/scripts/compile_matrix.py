#!/usr/bin/env python3
"""
compile_matrix.py — expand a matrix config (configs/*.json) into compile
cells, invoke nvcc/ptxas for each, disassemble with the matching cuobjdump,
and append one JSONL record per cell to results/artifacts.jsonl.

Resumable: a cell already present in artifacts.jsonl with a terminal status
(COMPILE_OK / COMPILE_FAILED / COMPILE_UNSUPPORTED) and whose referenced
files still exist on disk is skipped, not recompiled (task §23).

Must be run inside WSL2 (needs nvcc/ptxas/cuobjdump on PATH via absolute
paths from the config) — see AUDIT.md §4.5 for why native Windows can't do
this.

Usage:
    python3 compile_matrix.py --config configs/pilot_matrix.json --workers 1
    python3 compile_matrix.py --config configs/core_matrix.json --workers 4 --limit 50
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from kernel_lib import CONSTRUCTS, CONTEXTS

ROOT = Path(__file__).parent.parent  # robustness_study/
CUBIN_DIR = ROOT / "artifacts" / "cubin"
SASS_DIR = ROOT / "artifacts" / "sass"
LOG_DIR = ROOT / "artifacts" / "logs"
ARTIFACTS_JSONL = ROOT / "results" / "artifacts.jsonl"

# Recognized ptxas/nvcc unsupported-target error substrings (WSL2 CUDA 12.8/13.3,
# observed via AUDIT.md/PLAN.md probes; extended if the pilot surfaces others).
UNSUPPORTED_MARKERS = [
    "is not defined for option",
    "Unsupported gpu architecture",
    "unrecognized option",
    "unknown option",
]


def artifact_id(construct: str, context: str, opt: int, arch: str, toolkit_id: str, extra_flags: list[str]) -> str:
    key = f"{construct}|{context}|{opt}|{arch}|{toolkit_id}|{','.join(extra_flags or [])}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_done_ids() -> set[str]:
    done = set()
    if not ARTIFACTS_JSONL.exists():
        return done
    with ARTIFACTS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("compile_status") in ("COMPILE_OK", "COMPILE_FAILED", "COMPILE_UNSUPPORTED"):
                # Only trust it as done if referenced files still exist for OK records.
                if rec["compile_status"] == "COMPILE_OK":
                    cubin_ok = rec.get("cubin_path") and Path(rec["cubin_path"]).exists()
                    sass_ok = rec.get("sass_path") and Path(rec["sass_path"]).exists()
                    if not (cubin_ok and sass_ok):
                        continue
                done.add(rec["artifact_id"])
    return done


def append_record(record: dict) -> None:
    """Append one JSONL line. Single write() call is effectively atomic for
    lines well under the OS pipe buffer size, which every record here is."""
    line = json.dumps(record, sort_keys=True) + "\n"
    with ARTIFACTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()


def get_git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
                            capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


GIT_COMMIT = get_git_commit()


def compile_one(cell: dict, toolkit: dict, host_opt: str, timeout_s: int) -> dict:
    construct, context = cell["construct"], cell["context"]
    opt, arch = cell["ptxas_opt"], cell["arch"]
    extra_flags = cell.get("extra_device_flags", [])
    aid = artifact_id(construct, context, opt, arch, toolkit["id"], extra_flags)

    src_path = ROOT / "kernels" / "generated" / f"{construct.lower()}_{context.lower()}.cu"
    meta_path = ROOT / "kernels" / "metadata" / f"{construct.lower()}_{context.lower()}.json"
    src_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    cubin_path = CUBIN_DIR / f"{aid}.cubin"
    sass_path = SASS_DIR / f"{aid}.sass"
    stdout_path = LOG_DIR / f"{aid}.stdout.txt"
    stderr_path = LOG_DIR / f"{aid}.stderr.txt"

    cmd = [
        toolkit["nvcc"], f"-arch={arch}", host_opt,
        "-Xptxas", f"-O{opt}",
        "-ccbin", toolkit["host_compiler"],
        *extra_flags,
        "-cubin", "-o", str(cubin_path), str(src_path),
    ]

    record = {
        "artifact_id": aid,
        "study_phase": cell.get("study_phase", "unspecified"),
        "source_template": f"{construct}x{context}",
        "source_file": str(src_path.as_posix()),
        "source_hash_sha256": src_meta["source_hash_sha256"],
        "zeroization_construct": construct,
        "kernel_context": context,
        "buffer_words": src_meta["sbuf_nwords"],
        "block_threads": src_meta["block_threads"],
        "toolkit_path": toolkit["nvcc"],
        "cuda_version": toolkit["id"],
        "host_compiler": toolkit["host_compiler"],
        "host_optimization": host_opt,
        "ptxas_optimization": opt,
        "target_arch": arch,
        "maxrregcount": next((f for f in extra_flags if "maxrregcount" in f), None),
        "extra_device_flags": extra_flags,
        "compile_command": " ".join(cmd),
        "kernel_mangled_name": None,
        "kernel_demangled_name": src_meta["kernel_function_name"],
        "source_intent_label": construct,
        "binary_ground_truth": None,
        "checker_verdict": None,
        "manual_review_status": "not_reviewed",
        "manual_reviewer_1": None,
        "manual_reviewer_2": None,
        "manual_disagreement": None,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": GIT_COMMIT,
        "notes": [],
    }

    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        record["compile_status"] = "COMPILE_FAILED"
        record["compile_return_code"] = None
        record["compile_time_ms"] = int((time.time() - t0) * 1000)
        record["error_category"] = "timeout"
        record["error_message"] = f"compile exceeded {timeout_s}s timeout"
        stdout_path.write_text(e.stdout or "", encoding="utf-8")
        stderr_path.write_text(e.stderr or "", encoding="utf-8")
        record["compile_stdout_path"] = str(stdout_path.as_posix())
        record["compile_stderr_path"] = str(stderr_path.as_posix())
        return record

    elapsed_ms = int((time.time() - t0) * 1000)
    stdout_path.write_text(r.stdout, encoding="utf-8")
    stderr_path.write_text(r.stderr, encoding="utf-8")
    record["compile_return_code"] = r.returncode
    record["compile_time_ms"] = elapsed_ms
    record["compile_stdout_path"] = str(stdout_path.as_posix())
    record["compile_stderr_path"] = str(stderr_path.as_posix())

    if r.returncode != 0:
        unsupported = any(m in r.stderr for m in UNSUPPORTED_MARKERS)
        record["compile_status"] = "COMPILE_UNSUPPORTED" if unsupported else "COMPILE_FAILED"
        record["error_category"] = "unsupported_target" if unsupported else "compile_error"
        record["error_message"] = r.stderr.strip()[:2000]
        return record

    record["compile_status"] = "COMPILE_OK"
    record["error_category"] = None
    record["error_message"] = None
    if r.stderr.strip():
        record["notes"].append(f"nvcc warnings: {r.stderr.strip()[:500]}")

    record["cubin_path"] = str(cubin_path.as_posix())
    record["cubin_hash_sha256"] = sha256_file(cubin_path)
    record["cubin_size_bytes"] = cubin_path.stat().st_size

    # Disassemble with the matching toolkit's cuobjdump.
    dis_cmd = [toolkit["cuobjdump"], "-sass", str(cubin_path)]
    try:
        dr = subprocess.run(dis_cmd, capture_output=True, text=True, timeout=60)
        sass_text = dr.stdout
        sass_path.write_text(sass_text, encoding="utf-8")
        record["sass_path"] = str(sass_path.as_posix())
        record["sass_size_bytes"] = sass_path.stat().st_size
        if dr.returncode != 0:
            record["notes"].append(f"cuobjdump stderr: {dr.stderr.strip()[:500]}")
    except subprocess.TimeoutExpired:
        record["sass_path"] = None
        record["sass_size_bytes"] = None
        record["notes"].append("cuobjdump timed out")

    return record


def expand_cells(config: dict) -> list[dict]:
    cells = []
    for construct, context in config["pairs"]:
        for toolkit in config["toolkits"]:
            for arch in config["archs"]:
                for opt in config["ptxas_opt_levels"]:
                    cells.append({
                        "construct": construct, "context": context,
                        "toolkit": toolkit, "arch": arch, "ptxas_opt": opt,
                        "study_phase": config["name"],
                    })
    for extra in config.get("extra_flag_recheck_cells", []):
        toolkit = next(t for t in config["toolkits"] if t["id"] == extra["toolkit"])
        cells.append({
            "construct": extra["construct"], "context": extra["context"],
            "toolkit": toolkit, "arch": extra["arch"], "ptxas_opt": extra["ptxas_opt"],
            "extra_device_flags": extra["extra_device_flags"],
            "study_phase": config["name"] + "_flag_recheck",
        })
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=90, help="per-compile timeout, seconds")
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for d in (CUBIN_DIR, SASS_DIR, LOG_DIR, ARTIFACTS_JSONL.parent):
        d.mkdir(parents=True, exist_ok=True)

    cells = expand_cells(config)
    if args.limit:
        cells = cells[: args.limit]

    done_ids = load_done_ids()
    host_opt = config["host_optimization"]

    pending = []
    for cell in cells:
        aid = artifact_id(cell["construct"], cell["context"], cell["ptxas_opt"],
                           cell["arch"], cell["toolkit"]["id"], cell.get("extra_device_flags", []))
        if aid in done_ids:
            continue
        pending.append(cell)

    print(f"[compile_matrix] {config['name']}: {len(cells)} total cells, "
          f"{len(cells) - len(pending)} already done, {len(pending)} pending, "
          f"{args.workers} worker(s)")

    results = {"COMPILE_OK": 0, "COMPILE_FAILED": 0, "COMPILE_UNSUPPORTED": 0}
    t_start = time.time()

    def run(cell):
        return compile_one(cell, cell["toolkit"], host_opt, args.timeout)

    if args.workers == 1:
        for i, cell in enumerate(pending, 1):
            rec = run(cell)
            append_record(rec)
            results[rec["compile_status"]] += 1
            print(f"  [{i}/{len(pending)}] {rec['zeroization_construct']}x{rec['kernel_context']} "
                  f"{rec['target_arch']} O{rec['ptxas_optimization']} {rec['cuda_version']} "
                  f"-> {rec['compile_status']} ({rec['compile_time_ms']}ms)")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(run, cell): cell for cell in pending}
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                append_record(rec)
                results[rec["compile_status"]] += 1
                print(f"  [{i}/{len(pending)}] {rec['zeroization_construct']}x{rec['kernel_context']} "
                      f"{rec['target_arch']} O{rec['ptxas_optimization']} {rec['cuda_version']} "
                      f"-> {rec['compile_status']} ({rec['compile_time_ms']}ms)")

    elapsed = time.time() - t_start
    print(f"\n[compile_matrix] done in {elapsed:.1f}s: {results}")
    print(f"[compile_matrix] records appended to {ARTIFACTS_JSONL}")


if __name__ == "__main__":
    main()
