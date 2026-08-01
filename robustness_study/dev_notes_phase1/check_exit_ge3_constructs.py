import json
from collections import Counter

artifacts = {json.loads(l)["artifact_id"]: json.loads(l) for l in open("results/artifacts.jsonl")}
features = [json.loads(l) for l in open("results/features.jsonl")]

ge3 = [f for f in features
       if f.get("kernel_features") and f["kernel_features"].get("exit_count", 0) >= 3]
constructs = Counter(artifacts[f["artifact_id"]]["zeroization_construct"] for f in ge3)
print("total exit_count>=3 records (whole corpus):", len(ge3))
print("by construct:", dict(constructs))
