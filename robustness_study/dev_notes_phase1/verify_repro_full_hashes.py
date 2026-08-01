import json

recs = [json.loads(l) for l in open("results/reproducibility_runs.jsonl")]
mismatches = []
for r in recs:
    repeats = r["repeats"]
    cubin_hashes = {rep.get("cubin_sha256") for rep in repeats}
    sass_hashes = {rep.get("sass_sha256") for rep in repeats}
    json_hashes = {rep.get("nvdisasm_json_sha256") for rep in repeats}
    if len(cubin_hashes) > 1 or len(sass_hashes) > 1 or len(json_hashes) > 1:
        mismatches.append((r["artifact_id"], len(cubin_hashes), len(sass_hashes), len(json_hashes)))
    if r["original_cubin_sha256"] not in cubin_hashes:
        mismatches.append((r["artifact_id"], "ORIGINAL_MISMATCH", cubin_hashes, r["original_cubin_sha256"]))

print("total configs:", len(recs))
print("mismatches (cubin/sass/json hash not uniform across 3 repeats, or repeats != original):", len(mismatches))
for m in mismatches:
    print(" ", m)
