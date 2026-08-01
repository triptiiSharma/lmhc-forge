import json
from collections import defaultdict

recs = [json.loads(l) for l in open("results/controlled_cases.jsonl")]
by_case = defaultdict(list)
for r in recs:
    by_case[r["case_id"]].append(r)

order = [f"N{i}" for i in range(13)]
total_attempted = 0
total_compiled = 0
total_matched = 0
for cid in order:
    rs = by_case[cid]
    attempted = len(rs)
    compiled = sum(1 for r in rs if r.get("compile_status") == "COMPILE_OK" or r.get("cuobjdump_verdict") is not None)
    matched = sum(1 for r in rs if r.get("conforms_to_intended") is True)
    cuobjdump_verdicts = sorted(set(str(r.get("cuobjdump_verdict")) for r in rs))
    nvdisasm_verdicts = sorted(set(str(r.get("nvdisasm_verdict")) for r in rs if "nvdisasm_verdict" in r))
    agreements = [r.get("cross_tool_agreement") for r in rs if "cross_tool_agreement" in r]
    agree_str = f"{sum(1 for a in agreements if a)}/{len(agreements)}" if agreements else "N/A (no nvdisasm run for this case)"
    total_attempted += attempted
    total_compiled += compiled
    total_matched += matched
    print(f"{cid} | attempted={attempted} compiled={compiled} matched={matched} "
          f"cuobjdump={cuobjdump_verdicts} nvdisasm={nvdisasm_verdicts} agree={agree_str}")

print(f"\nTOTALS: attempted={total_attempted} compiled={total_compiled} matched={total_matched} "
      f"non_matched={total_attempted - total_matched}")
