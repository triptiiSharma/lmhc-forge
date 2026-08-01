# Paper Tables

Generated from the frozen evidence commit; every cell traces to FINAL_RESULTS_LEDGER.md.

## Table I. Experimental design and configuration dimensions

| Dimension | Values | Count |
|---|---|---|
| Zeroization constructs | Z0-Z6 (no-scrub, plain, volatile, plain+barrier, volatile+barrier, hardened, inline-PTX hardened) | 7 |
| Kernel contexts | K0-K5 (pure-dead, fill-then-scrub, fill/use/scrub, reduction, tiled-GEMM, attention-style) | 6 |
| ptxas optimization levels | O0, O1, O2, O3 | 4 |
| Architecture targets | sm_75, sm_80, sm_86, sm_87, sm_89, sm_90 | 6 |
| CUDA toolkits | 12.8 (V12.8.90/93), 13.3 (V13.3.29/33) | 2 |
| Core matrix size | 7 x 6 x 4 x 6 x 2 | 2016 |
| Physical GPU hardware | RTX 4050 Laptop (Ada, sm_89) | 1 |
| Controlled cases | N0-N12 | 13 (55 configurations) |
| Reproducibility subset | stratified core configurations x 3 repeats | 50 x 3 = 150 |
| Runtime sanity subset | 6 constructs x 3 contexts x 5 repeats, sm_89 only | 18 x 5 = 90 |


## Table II. Structural-category outcomes by zeroization construct

| Construct | Total | FULL_SIGNATURE | PARTIAL_SIGNATURE | STORE_ONLY | ABSENT | UNDECIDABLE |
|---|---|---|---|---|---|---|
| Z0 | 288 | 0 | 0 | 0 | 288 | 0 |
| Z1 | 289 | 0 | 192 | 0 | 97 | 0 |
| Z2 | 289 | 0 | 193 | 96 | 0 | 0 |
| Z3 | 288 | 0 | 192 | 0 | 96 | 0 |
| Z4 | 288 | 0 | 288 | 0 | 0 | 0 |
| Z5 | 288 | 288 | 0 | 0 | 0 | 0 |
| Z6 | 288 | 288 | 0 | 0 | 0 | 0 |


## Table III. Controlled N0-N12 case results

| Case | Type | Attempted | Compiled | Matched intended | Cross-tool agreement |
|---|---|---|---|---|---|
| N0 | REAL_COMPILED_CASE | 1 | 1 | 1 | 1/1 |
| N1 | REAL_COMPILED_CASE | 1 | 1 | 1 | 1/1 |
| N2 | REAL_COMPILED_CASE | 1 | 1 | 1 | 1/1 |
| N3 | REAL_COMPILED_CASE | 8 | 8 | 8 | 8/8 |
| N4 | REAL_COMPILED_CASE | 8 | 8 | 8 | 8/8 |
| N5 | REAL_COMPILED_CASE | 8 | 8 | 8 | 8/8 |
| N6 | REAL_COMPILED_CASE | 8 | 8 | 0 | 8/8 |
| N7 | REAL_COMPILED_CASE | 8 | 8 | 8 | 8/8 |
| N8 | REAL_COMPILED_CASE | 8 | 8 | 8 | 8/8 |
| N9 | REAL_COMPILED_CASE | 1 | 1 | 1 | 1/1 |
| N10 | REAL_COMPILED_CASE | 1 | 1 | 1 | 1/1 |
| N11 | REAL_BINARY_INPUT_CASE | 1 | 1 | 1 | N/A |
| N12 | REAL_COMPILED_CASE | 1 | 1 | 1 | 1/1 |
| TOTAL |  | 55 | 55 | 47 | 54/54 (N11 excluded) |


## Table IV. Cross-tool agreement, reproducibility, and analysis-cost summary

| Metric | Value |
|---|---|
| Cross-tool category agreement | 2016/2016 (100.0%) |
| Cross-tool feature agreement (8 features) | 2016/2016 each (100%) |
| Cross-tool disagreements | 0 |
| nvdisasm analysis time (median / p95) | 248ms / 379ms |
| cuobjdump-path analysis time (median / p95) | 6ms / 13ms |
| Compile time (median / p95) | 313ms / 368ms |
| Reproducibility: configurations x repeats | 50 x 3 = 150 |
| Reproducibility: byte-identical rate | 150/150 (100%) |
| Total core-corpus storage | 89 MB |
