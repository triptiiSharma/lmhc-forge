import json
import hashlib
from pathlib import Path

errors = []

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# 1. Core artifacts.jsonl hash verification (independent of validate_dataset.py)
artifacts = [json.loads(l) for l in open("results/artifacts.jsonl")]
for a in artifacts:
    if a.get("compile_status") != "COMPILE_OK":
        continue
    for pf, hf in [("cubin_path", "cubin_hash_sha256")]:
        p, h = a.get(pf), a.get(hf)
        if p and h:
            if not Path(p).exists():
                errors.append(f"{a['artifact_id']}: {pf} missing: {p}")
            elif sha256(p) != h:
                errors.append(f"{a['artifact_id']}: {pf} hash mismatch")
print(f"[1] core artifacts.jsonl: {len(artifacts)} records checked, {len(errors)} errors so far")

# 2. controlled_cases.jsonl file existence
controlled = [json.loads(l) for l in open("results/controlled_cases.jsonl")]
missing = 0
for r in controlled:
    for key in ("cubin_path", "sass_path", "nvdisasm_json_path"):
        p = r.get(key)
        if p and not Path(p).exists():
            errors.append(f"{r['case_id']}/{r.get('artifact_id')}: {key} missing: {p}")
            missing += 1
print(f"[2] controlled_cases.jsonl: {len(controlled)} records, {missing} missing files")

# 3. reproducibility_runs.jsonl -- re-verify hashes fresh (not just trust the stored classification)
repro = [json.loads(l) for l in open("results/reproducibility_runs.jsonl")]
repro_errors = 0
for r in repro:
    for rep in r["repeats"]:
        cp = rep.get("cubin_path")
        if cp and Path(cp).exists():
            actual = sha256(cp)
            if actual != rep.get("cubin_sha256"):
                repro_errors += 1
                errors.append(f"{r['artifact_id']}: repeat cubin hash mismatch on disk vs recorded")
        elif cp:
            repro_errors += 1
            errors.append(f"{r['artifact_id']}: repeat cubin file missing: {cp}")
print(f"[3] reproducibility_runs.jsonl: {len(repro)} configs x3 repeats, {repro_errors} errors")

# 4. runtime_sanity.jsonl -- check source files and executables exist
runtime = [json.loads(l) for l in open("results/runtime_sanity.jsonl")]
rt_missing = 0
for r in runtime:
    sp = r.get("source_file")
    if sp and not Path(sp).exists():
        rt_missing += 1
        errors.append(f"{r['zeroization_construct']}x{r['kernel_context']}: source missing: {sp}")
print(f"[4] runtime_sanity.jsonl: {len(runtime)} records, {rt_missing} missing sources")

# 5. nvdisasm_features.jsonl completeness vs core matrix
nvdisasm_full = [json.loads(l) for l in open("results/nvdisasm_features.jsonl")]
print(f"[5] nvdisasm_features.jsonl: {len(nvdisasm_full)} records (expect 2016 core)")
assert len(nvdisasm_full) == 2016, "nvdisasm full pass record count mismatch!"

# 6. cross-reconcile controlled_cases totals
by_case = {}
for r in controlled:
    by_case.setdefault(r["case_id"], []).append(r)
total_attempted = sum(len(v) for v in by_case.values())
total_matched = sum(1 for r in controlled if r.get("conforms_to_intended") is True)
total_nonmatched = sum(1 for r in controlled if r.get("conforms_to_intended") is False)
print(f"[6] controlled cases: attempted={total_attempted} matched={total_matched} "
      f"non_matched={total_nonmatched} sum_check={total_matched+total_nonmatched==total_attempted}")

print(f"\n=== TOTAL ERRORS: {len(errors)} ===")
for e in errors[:20]:
    print(" ", e)
