import json
recs = [json.loads(l) for l in open("results/controlled_cases.jsonl")]

n6 = [r for r in recs if r["case_id"] == "N6"][0]
print("N6 cuobjdump_features:", json.dumps(n6.get("cuobjdump_features"), indent=2))
print("N6 cuobjdump_detail:", n6.get("cuobjdump_detail"))

n9 = [r for r in recs if r["case_id"] == "N9"][0]
print("\nN9 cuobjdump_features:", json.dumps(n9.get("cuobjdump_features"), indent=2))
