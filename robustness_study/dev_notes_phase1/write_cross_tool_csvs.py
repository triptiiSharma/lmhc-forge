import csv
import json

recs = [json.loads(l) for l in open("results/nvdisasm_features.jsonl")]
artifacts = {json.loads(l)["artifact_id"]: json.loads(l) for l in open("results/artifacts.jsonl")}

feature_keys = ["zero_shared_store_direct_count", "zero_shared_store_indirect_count",
                "zero_shared_store_count", "membar_count", "membar_cta_count",
                "bar_sync_count", "exit_count", "branch_count"]

with open("results/cross_tool_comparison.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["artifact_id", "zeroization_construct", "kernel_context", "target_arch",
                "ptxas_optimization", "cuda_version", "cuobjdump_verdict", "nvdisasm_verdict",
                "category_agreement"] + [f"cuobjdump_{k}" for k in feature_keys]
               + [f"nvdisasm_{k}" for k in feature_keys]
               + [f"agree_{k}" for k in feature_keys])
    cuobjdump_features = {json.loads(l)["artifact_id"]: json.loads(l) for l in open("results/features.jsonl")}
    for r in recs:
        a = artifacts.get(r["artifact_id"], {})
        cf = (cuobjdump_features.get(r["artifact_id"], {}).get("kernel_features")) or {}
        nf = r.get("nvdisasm_features") or {}
        fa = r.get("feature_agreement") or {}
        row = [r["artifact_id"], a.get("zeroization_construct"), a.get("kernel_context"),
               a.get("target_arch"), a.get("ptxas_optimization"), a.get("cuda_version"),
               r["cuobjdump_verdict"], r["nvdisasm_verdict"], r["category_agreement"]]
        row += [cf.get(k) for k in feature_keys]
        row += [nf.get(k) for k in feature_keys]
        row += [fa.get(k) for k in feature_keys]
        w.writerow(row)

mismatches = [r for r in recs if r["category_agreement"] is False]
with open("results/cross_tool_mismatches.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["artifact_id", "construct", "context", "arch", "opt", "toolkit",
                "cuobjdump_verdict", "nvdisasm_verdict", "cause"])
    for r in mismatches:
        a = artifacts.get(r["artifact_id"], {})
        w.writerow([r["artifact_id"], a.get("zeroization_construct"), a.get("kernel_context"),
                    a.get("target_arch"), a.get("ptxas_optimization"), a.get("cuda_version"),
                    r["cuobjdump_verdict"], r["nvdisasm_verdict"], "UNRESOLVED"])

summary = {
    "total_artifacts": len(recs),
    "category_agreement_count": sum(1 for r in recs if r["category_agreement"]),
    "category_agreement_pct": round(100 * sum(1 for r in recs if r["category_agreement"]) / len(recs), 2),
    "disagreement_count": len(mismatches),
    "unsupported_count": sum(1 for r in recs if r.get("schema_error")),
    "nvdisasm_undecidable_count": sum(1 for r in recs if r["nvdisasm_verdict"] == "UNDECIDABLE"),
    "cuobjdump_undecidable_count": sum(1 for r in recs if r["cuobjdump_verdict"] == "UNDECIDABLE"),
    "feature_agreement": {},
}
for k in feature_keys:
    total = sum(1 for r in recs if k in (r.get("feature_agreement") or {}))
    agree = sum(1 for r in recs if (r.get("feature_agreement") or {}).get(k))
    summary["feature_agreement"][k] = {"agree": agree, "total": total}

with open("results/cross_tool_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print("wrote results/cross_tool_comparison.csv, results/cross_tool_mismatches.csv, results/cross_tool_summary.json")
