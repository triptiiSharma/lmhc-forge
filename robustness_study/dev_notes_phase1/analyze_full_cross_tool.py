import json
from collections import Counter, defaultdict

recs = [json.loads(l) for l in open("results/nvdisasm_features.jsonl")]
artifacts = {json.loads(l)["artifact_id"]: json.loads(l) for l in open("results/artifacts.jsonl")}

print("total records:", len(recs))
print("schema_ok:", Counter(r["schema_ok"] for r in recs))
print("return_code:", Counter(r["return_code"] for r in recs))
print("category_agreement:", Counter(r["category_agreement"] for r in recs))
print("nvdisasm_verdict:", Counter(r["nvdisasm_verdict"] for r in recs))
print("cuobjdump_verdict:", Counter(r["cuobjdump_verdict"] for r in recs))

disagreements = [r for r in recs if r["category_agreement"] is False]
print("\ndisagreements:", len(disagreements))
for d in disagreements[:20]:
    a = artifacts.get(d["artifact_id"], {})
    print(" ", d["artifact_id"], a.get("zeroization_construct"), a.get("kernel_context"),
          a.get("target_arch"), a.get("ptxas_optimization"), a.get("cuda_version"),
          "nvdisasm=", d["nvdisasm_verdict"], "cuobjdump=", d["cuobjdump_verdict"])

feature_keys = ["zero_shared_store_direct_count", "zero_shared_store_indirect_count",
                "zero_shared_store_count", "membar_count", "membar_cta_count",
                "bar_sync_count", "exit_count", "branch_count"]
feat_agree = {k: 0 for k in feature_keys}
feat_total = {k: 0 for k in feature_keys}
feat_dis = {k: [] for k in feature_keys}
for r in recs:
    fa = r.get("feature_agreement") or {}
    for k in feature_keys:
        if k in fa:
            feat_total[k] += 1
            if fa[k]:
                feat_agree[k] += 1
            else:
                feat_dis[k].append(r["artifact_id"])

print("\nfeature-level agreement:")
for k in feature_keys:
    print(f"  {k}: {feat_agree[k]}/{feat_total[k]}")
    if feat_dis[k]:
        print(f"    disagreements ({len(feat_dis[k])}):", feat_dis[k][:10])

# by dimension
def by_dim(key_fn, label):
    counts = defaultdict(lambda: [0, 0])  # [agree, total]
    for r in recs:
        a = artifacts.get(r["artifact_id"], {})
        k = key_fn(a)
        counts[k][1] += 1
        if r["category_agreement"]:
            counts[k][0] += 1
    print(f"\nby {label}:")
    for k, (agree, total) in sorted(counts.items(), key=lambda x: str(x[0])):
        print(f"  {k}: {agree}/{total}")

by_dim(lambda a: a.get("zeroization_construct"), "construct")
by_dim(lambda a: a.get("kernel_context"), "context")
by_dim(lambda a: a.get("ptxas_optimization"), "opt level")
by_dim(lambda a: a.get("target_arch"), "architecture")
by_dim(lambda a: a.get("cuda_version"), "toolkit")

unsupported = [r for r in recs if r.get("schema_error")]
print("\nunsupported/schema_error count:", len(unsupported))
for u in unsupported[:10]:
    print(" ", u["artifact_id"], u.get("schema_error"))

nv_undecidable = sum(1 for r in recs if r["nvdisasm_verdict"] == "UNDECIDABLE")
co_undecidable = sum(1 for r in recs if r["cuobjdump_verdict"] == "UNDECIDABLE")
print("\nnvdisasm UNDECIDABLE count:", nv_undecidable)
print("cuobjdump UNDECIDABLE count:", co_undecidable)

times = [r["analysis_time_ms"] for r in recs if r.get("analysis_time_ms") is not None]
times_sorted = sorted(times)
def pctl(vals, p):
    k = (len(vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[f]
    return vals[f] + (vals[c] - vals[f]) * (k - f)
print("\nnvdisasm analysis_time_ms: median", times_sorted[len(times_sorted)//2], "p95", round(pctl(times_sorted, 0.95), 1))
