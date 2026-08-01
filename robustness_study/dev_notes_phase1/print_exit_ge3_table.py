import csv
rows = list(csv.DictReader(open("results/phase5_private_mapping.csv")))
sel = [r for r in rows if r["selection_reason"] == "tier3_multi_exit_ge3"]
for r in sel:
    print(f"| {r['review_id']} | {r['original_artifact_id']} | "
          f"{r['zeroization_construct']}x{r['kernel_context']} | {r['target_arch']} | "
          f"O{r['ptxas_optimization']} | {r['cuda_version']} | {r['checker_verdict']} |")
