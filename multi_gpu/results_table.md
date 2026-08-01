## LMHC-Forge — Multi-GPU Replication Results

| GPU                                | Architecture | Driver        | CUDA Runtime | Same-SM trials | EI (no scrub) | Control EI | EI (LMHC scrub) | Valid |
| ---------------------------------- | ------------ | ------------- | ------------ | -------------- | ------------- | ---------- | --------------- | ----- |
| NVIDIA GeForce RTX 3050 Laptop GPU | Ampere       | 610.43.02     | 12.4         | 200            | 1.0000        | 0.0000     | 0.0000          | Y     |
| NVIDIA GeForce RTX 3060 Ti         | Ampere       | cuda_api_13.2 | 12.8         | 200            | 0.9950        | 0.0000     | 0.0000          | Y     |
| NVIDIA GeForce RTX 4050 Laptop GPU | Ada Lovelace | cuda_api_12.8 | 12.8         | 200            | 1.0000        | 0.0000     | 0.0000          | Y     |

NOTE: Co-residency is forced on every device (single-block __launch_bounds__(32,1) + %smid verification). These are controlled-placement results consistent with the adversarial threat model — not natural-scheduling frequencies, and not a claim about production prevalence or about non-tested hardware. See ../docs/INTERPRETATION.md for pre-committed outcome readings.

### Notes
- **NVIDIA GeForce RTX 3050 Laptop GPU** (NVIDIA_GeForce_RTX_3050_Laptop_GPU_sm_86_cuda_api_13.3.json): VALID: control clean; LMHC scrub drives EI from 1.0 to 0.0 on this device.
- **NVIDIA GeForce RTX 3060 Ti** (NVIDIA_GeForce_RTX_3060_Ti_sm_86_cuda_api_13.2.json): VALID: control clean; LMHC scrub drives EI to 0.0 on this device.
- **NVIDIA GeForce RTX 4050 Laptop GPU** (NVIDIA_GeForce_RTX_4050_Laptop_GPU_sm_89_cuda_api_12.8.json): VALID: control clean; LMHC scrub drives EI to 0.0 on this device.

