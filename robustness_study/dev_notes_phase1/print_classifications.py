import csv
with open("results/classifications.csv") as f:
    rows = list(csv.DictReader(f))
rows.sort(key=lambda r: (r["zeroization_construct"], r["kernel_context"], r["target_arch"], r["ptxas_optimization"]))
for r in rows:
    print(f'{r["zeroization_construct"]:4} {r["kernel_context"]:4} {r["target_arch"]:6} O{r["ptxas_optimization"]}  {r["cuda_version"]:10} -> {r["checker_verdict"]:18} {r["checker_detail"][:75]}')
