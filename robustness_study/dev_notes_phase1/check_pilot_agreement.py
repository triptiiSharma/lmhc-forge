import json
from collections import Counter

recs = [json.loads(l) for l in open("results/nvdisasm_features_pilot.jsonl")]
print("total records:", len(recs))
print("schema_ok distribution:", Counter(r["schema_ok"] for r in recs))
print("schema_error present:", sum(1 for r in recs if r.get("schema_error")))
print("return_code distribution:", Counter(r["return_code"] for r in recs))
print("category_agreement:", Counter(r["category_agreement"] for r in recs))
print("nvdisasm_verdict distribution:", Counter(r["nvdisasm_verdict"] for r in recs))
print("cuobjdump_verdict distribution:", Counter(r["cuobjdump_verdict"] for r in recs))

disagreements = [r for r in recs if r["category_agreement"] is False]
print("\ndisagreements:", len(disagreements))
for d in disagreements:
    print(" ", d["artifact_id"], d["nvdisasm_verdict"], "vs", d["cuobjdump_verdict"])

# feature-level agreement
feature_keys = ["zero_shared_store_direct_count", "zero_shared_store_indirect_count",
                "zero_shared_store_count", "membar_count", "membar_cta_count",
                "bar_sync_count", "exit_count", "branch_count"]
feat_agree_counts = {k: 0 for k in feature_keys}
feat_total_counts = {k: 0 for k in feature_keys}
feat_disagreements = {k: [] for k in feature_keys}
for r in recs:
    fa = r.get("feature_agreement") or {}
    for k in feature_keys:
        if k in fa:
            feat_total_counts[k] += 1
            if fa[k]:
                feat_agree_counts[k] += 1
            else:
                feat_disagreements[k].append(r["artifact_id"])

print("\nfeature-level agreement:")
for k in feature_keys:
    print(f"  {k}: {feat_agree_counts[k]}/{feat_total_counts[k]}")
    if feat_disagreements[k]:
        print(f"    disagreements: {feat_disagreements[k]}")

analysis_times = [r["analysis_time_ms"] for r in recs if r.get("analysis_time_ms") is not None]
print("\nanalysis_time_ms: min", min(analysis_times), "max", max(analysis_times))
