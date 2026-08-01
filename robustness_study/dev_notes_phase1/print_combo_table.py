import csv
from collections import defaultdict

rows = list(csv.DictReader(open("results/classifications.csv")))
constructs = [f"Z{i}" for i in range(7)]
contexts = [f"K{i}" for i in range(6)]
combo_verdicts = defaultdict(lambda: defaultdict(int))
for r in rows:
    combo_verdicts[(r["zeroization_construct"], r["kernel_context"])][r["checker_verdict"]] += 1

print(f'{"construct":4} {"context":4} verdicts (count)')
for z in constructs:
    for k in contexts:
        v = combo_verdicts[(z, k)]
        print(f'{z:4} {k:4} {dict(v)}')
