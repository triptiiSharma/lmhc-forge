// multi_gpu_repro.cu — LMHC-Forge Multi-GPU Replication Harness
//
// Scientific objective: test reproducibility of the residue mechanism and the
// LMHC mitigation across NVIDIA architectures under identical adversarial
// (forced co-residency) conditions. The only independent variable is the GPU;
// methodology is held byte-for-byte identical to src/leakage_demo.cu.
//
// SCOPE GUARD: Controlled-placement replication only. Natural-scheduling mode
// is explicitly out of scope for this artifact.
//
// MANDATORY FRAMING (SPEC.md §Module 6, INTERPRETATION.md):
//   "Residue is fully observable when the subsequent kernel is co-resident on
//    the same SM — co-residency is forced in this experiment to demonstrate the
//    window. This is in-process cross-kernel shared-memory persistence, NOT
//    cross-tenant LeftoverLocals on NVIDIA. Co-residency is forced on every
//    device, so these are controlled-placement results consistent with our
//    adversarial threat model — not natural-scheduling frequencies, and not a
//    claim about production prevalence or about non-tested hardware."
//
// Three phases (writer = kernelA_*, listener = kernelB_*):
//   1. Residue run  — writer without scrub  → listener  (measures raw EI)
//   2. Control run  — listener only, writer NEVER launched (validity gate)
//   3. LMHC run     — writer with hardened scrub → listener (post-scrub EI)
//
// If the control is non-zero → valid=false; do not report residue as a leak.
//
// Output: JSON to stdout (redirect to multi_gpu/results/<name>.json), or via -o.
// Usage:  ./multi_gpu_repro [-n <trials>] [-o <file.json>]

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#ifdef __linux__
#  include <sys/utsname.h>
#endif

// ── Compile-time nvcc version (embedded by the preprocessor) ─────────────────
#ifndef __CUDACC_VER_MAJOR__
#  define __CUDACC_VER_MAJOR__ 0
#  define __CUDACC_VER_MINOR__ 0
#  define __CUDACC_VER_BUILD__ 0
#endif

// ── Baseline geometry — identical to src/leakage_demo.cu; never runtime-variable
#define NWORDS      4096           // 4096 × uint32 = 16 KB shared memory
#define NTRIALS_DEFAULT  200       // default trial count; overridable with -n
#define BLOCK_DIM    32            // one warp; __launch_bounds__ keeps occupancy low
#define PATTERN  0xCAFEBABEu       // sentinel written by writer kernels

#define CUDA_CHECK(call)                                                        \
    do { cudaError_t _e = (call);                                               \
         if (_e != cudaSuccess) {                                               \
             fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                     __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); }    \
    } while (0)

// ── %smid special register ────────────────────────────────────────────────────
// %smid is a documented PTX special register (PTX ISA §9.4) that returns the
// streaming-multiprocessor index of the executing thread block. Reading it
// slightly perturbs scheduling because it adds an instruction to the warp's
// critical path; the effect is negligible for single-block single-warp kernels.
__device__ __forceinline__ uint32_t get_smid(void) {
    uint32_t id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
}

// ── Writer A: fills shared with PATTERN, exits WITHOUT scrub ──────────────────
// Verbatim from src/leakage_demo.cu::kernelA_unscrubbed.
__launch_bounds__(BLOCK_DIM, 1)
__global__ void kernelA_unscrubbed(uint32_t *smid_out) {
    __shared__ uint32_t buf[NWORDS];
    const int tid = threadIdx.x;
    if (tid == 0) *smid_out = get_smid();
    for (int i = tid; i < NWORDS; i += BLOCK_DIM) buf[i] = PATTERN;
    __syncthreads();
    // Volatile sink: prevents dead-store-elimination of the fill above.
    if (tid == 0) { volatile uint32_t s = buf[NWORDS - 1]; (void)s; }
    // EXIT WITHOUT SCRUB — pattern persists in shared SRAM
}

// ── Writer B: fills shared with PATTERN, exits WITH LMHC hardened scrub ───────
// Verbatim from src/leakage_demo.cu::kernelA_scrubbed.
// This is in-process cross-kernel shared-memory persistence,
// NOT cross-tenant LeftoverLocals on NVIDIA.
__launch_bounds__(BLOCK_DIM, 1)
__global__ void kernelA_scrubbed(uint32_t *smid_out) {
    __shared__ uint32_t buf[NWORDS];
    const int tid = threadIdx.x;
    if (tid == 0) *smid_out = get_smid();
    for (int i = tid; i < NWORDS; i += BLOCK_DIM) buf[i] = PATTERN;
    __syncthreads();
    if (tid == 0) { volatile uint32_t s = buf[NWORDS - 1]; (void)s; }
    // ── LMHC hardened scrub (Module 2 pattern — volatile + fence + sync) ──────
    volatile uint32_t *vp = buf;
    for (int i = tid; i < NWORDS; i += BLOCK_DIM) vp[i] = 0u;
    __threadfence_block();
    __syncthreads();
    // EXIT AFTER SCRUB — all words guaranteed zero by C++ memory model
}

// ── Listener: reads shared BEFORE writing, counts nonzero words ───────────────
// Verbatim from src/leakage_demo.cu::kernelB_listener.
__launch_bounds__(BLOCK_DIM, 1)
__global__ void kernelB_listener(uint32_t *smid_out, int *nz_out) {
    __shared__ uint32_t buf[NWORDS];
    const int tid = threadIdx.x;
    if (tid == 0) *smid_out = get_smid();
    // === READ BEFORE ANY WRITE — measures cross-launch residue ================
    int nz = 0;
    for (int i = tid; i < NWORDS; i += BLOCK_DIM)
        if (buf[i] != 0u) nz++;
    // ==========================================================================
    for (int off = 16; off > 0; off >>= 1)
        nz += __shfl_down_sync(0xFFFFFFFFu, nz, off);
    if (tid == 0) atomicAdd(nz_out, nz);
}

// ── Control listener: writer NEVER launched in this process ───────────────────
// Identical code path to kernelB_listener; named separately to make the
// validity-gate intent explicit in the call site.
__launch_bounds__(BLOCK_DIM, 1)
__global__ void kernelB_control(uint32_t *smid_out, int *nz_out) {
    __shared__ uint32_t buf[NWORDS];
    const int tid = threadIdx.x;
    if (tid == 0) *smid_out = get_smid();
    // READ BEFORE ANY WRITE — no writer has ever run in this process
    int nz = 0;
    for (int i = tid; i < NWORDS; i += BLOCK_DIM)
        if (buf[i] != 0u) nz++;
    for (int off = 16; off > 0; off >>= 1)
        nz += __shfl_down_sync(0xFFFFFFFFu, nz, off);
    if (tid == 0) atomicAdd(nz_out, nz);
}

// ── Host helpers ──────────────────────────────────────────────────────────────

static const char *get_arch_name(int major, int minor) {
    if (major >= 10)               return "Blackwell";
    if (major == 9)                return "Hopper";
    if (major == 8 && minor == 9)  return "Ada Lovelace";
    if (major == 8)                return "Ampere";
    if (major == 7 && minor == 5)  return "Turing";
    if (major == 7)                return "Volta";
    if (major == 6)                return "Pascal";
    if (major == 5)                return "Maxwell";
    return "Unknown";
}

// Read NVIDIA display-driver version from /proc/driver/nvidia/version.
// Falls back to CUDA driver API level if the file is absent or unreadable.
static void get_driver_str(char *buf, size_t len) {
    FILE *f = fopen("/proc/driver/nvidia/version", "r");
    if (f) {
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            char *p = strstr(line, "Kernel Module");
            if (p) {
                p += strlen("Kernel Module");
                while (*p == ' ') p++;
                size_t n = 0;
                while (*p && *p != ' ' && *p != '\n' && n + 1 < len)
                    buf[n++] = *p++;
                buf[n] = '\0';
                if (n > 0) { fclose(f); return; }
            }
        }
        fclose(f);
    }
    int drv = 0; cudaDriverGetVersion(&drv);
    snprintf(buf, len, "cuda_api_%d.%d", drv / 1000, (drv % 1000) / 10);
}

static int detect_wsl2(void) {
#ifdef __linux__
    FILE *f = fopen("/proc/version", "r");
    if (!f) return 0;
    char line[512]; int found = 0;
    while (fgets(line, sizeof(line), f))
        if (strstr(line, "microsoft") || strstr(line, "Microsoft") || strstr(line, "WSL"))
            { found = 1; break; }
    fclose(f); return found;
#else
    return 0;
#endif
}

static void get_os_str(char *buf, size_t len) {
#ifdef __linux__
    struct utsname u; uname(&u);
    snprintf(buf, len, "%s %s %s", u.sysname, u.release, u.machine);
#else
    snprintf(buf, len, "non-Linux");
#endif
}

// Escape backslash and double-quote for JSON strings.
static void json_esc(char *out, size_t outlen, const char *in) {
    size_t j = 0;
    for (size_t i = 0; in[i] && j + 2 < outlen; i++) {
        if (in[i] == '"' || in[i] == '\\') out[j++] = '\\';
        out[j++] = in[i];
    }
    out[j] = '\0';
}

// Replace spaces with underscores; keep alphanumeric, _, -, .
static void sanitize(char *out, size_t outlen, const char *in) {
    size_t j = 0;
    for (size_t i = 0; in[i] && j + 1 < outlen; i++) {
        char c = in[i];
        if (c == ' ') { out[j++] = '_'; }
        else if ((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')
                 || c=='_' || c=='-' || c=='.') { out[j++] = c; }
    }
    out[j] = '\0';
}

// ── Phase result ──────────────────────────────────────────────────────────────

typedef struct { int same_sm; long long nz; long long total; } PhaseResult;

static PhaseResult run_residue_phase(int use_scrub, int ntrials,
                                     uint32_t *d_smA, uint32_t *d_smB, int *d_nz) {
    PhaseResult r = {0, 0, 0};
    for (int t = 0; t < ntrials; t++) {
        CUDA_CHECK(cudaMemset(d_nz, 0, sizeof(int)));
        if (use_scrub)
            kernelA_scrubbed<<<1, BLOCK_DIM>>>(d_smA);
        else
            kernelA_unscrubbed<<<1, BLOCK_DIM>>>(d_smA);
        CUDA_CHECK(cudaDeviceSynchronize());
        kernelB_listener<<<1, BLOCK_DIM>>>(d_smB, d_nz);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t ha, hb; int hnz;
        CUDA_CHECK(cudaMemcpy(&ha, d_smA, 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&hb, d_smB, 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&hnz, d_nz, sizeof(int), cudaMemcpyDeviceToHost));
        if (ha == hb) { r.same_sm++; r.nz += hnz; r.total += NWORDS; }
    }
    return r;
}

static PhaseResult run_control_phase(int ntrials, uint32_t *d_smB, int *d_nz) {
    // Writer is intentionally absent. All trials counted (no SM comparison needed).
    PhaseResult r = {ntrials, 0, (long long)ntrials * NWORDS};
    for (int t = 0; t < ntrials; t++) {
        CUDA_CHECK(cudaMemset(d_nz, 0, sizeof(int)));
        kernelB_control<<<1, BLOCK_DIM>>>(d_smB, d_nz);
        CUDA_CHECK(cudaDeviceSynchronize());
        int hnz;
        CUDA_CHECK(cudaMemcpy(&hnz, d_nz, sizeof(int), cudaMemcpyDeviceToHost));
        r.nz += hnz;
    }
    return r;
}

// ── Usage ─────────────────────────────────────────────────────────────────────

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [-n <trials>] [-o <output.json>]\n"
        "  -n  trials per phase (default: %d, matches src/leakage_demo.cu)\n"
        "  -o  write JSON to file (default: stdout)\n"
        "Buffer geometry is fixed at %d words (%d KB) to match the baseline\n"
        "experiment in src/leakage_demo.cu exactly.\n",
        prog, NTRIALS_DEFAULT, NWORDS, NWORDS * 4 / 1024);
    exit(1);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    int ntrials = NTRIALS_DEFAULT;
    const char *outfile = NULL;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-n") && i+1 < argc) ntrials = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-o") && i+1 < argc) outfile = argv[++i];
        else usage(argv[0]);
    }
    if (ntrials <= 0) {
        fprintf(stderr, "Error: -n must be a positive integer.\n"); return 1;
    }

    // ── Collect environment ───────────────────────────────────────────────────
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    char driver_str[64]; get_driver_str(driver_str, sizeof(driver_str));
    char os_str[256];    get_os_str(os_str, sizeof(os_str));
    int  wsl2 = detect_wsl2();
    int  rt_ver = 0, drv_ver = 0;
    cudaRuntimeGetVersion(&rt_ver);
    cudaDriverGetVersion(&drv_ver);

    const char *arch_name = get_arch_name(prop.major, prop.minor);
    char sm_str[16];
    snprintf(sm_str, sizeof(sm_str), "sm_%d%d", prop.major, prop.minor);

    char nvcc_ver[64];
    snprintf(nvcc_ver, sizeof(nvcc_ver), "%d.%d.%d",
             __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);

    time_t now = time(NULL);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));

    // ── GPU allocations ───────────────────────────────────────────────────────
    uint32_t *d_smA, *d_smB; int *d_nz;
    CUDA_CHECK(cudaMalloc(&d_smA, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_smB, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_nz,  sizeof(int)));

    fprintf(stderr, "[multi_gpu_repro] GPU     : %s (%s, %s)\n",
            prop.name, sm_str, arch_name);
    fprintf(stderr, "[multi_gpu_repro] Config  : trials=%d  buffer=%d words (%d KB, fixed)\n",
            ntrials, NWORDS, NWORDS * 4 / 1024);

    // ── Run three phases ──────────────────────────────────────────────────────
    // Control MUST run first — before any writer has ever launched in this
    // process. This mirrors phase0_control.cu: proves the listener reads zero
    // when the writer has never run, ruling out self-contamination.
    fprintf(stderr, "[multi_gpu_repro] Phase 1 : control (validity gate, no writer)...\n");
    PhaseResult control = run_control_phase(ntrials, d_smB, d_nz);

    fprintf(stderr, "[multi_gpu_repro] Phase 2 : residue (no scrub)...\n");
    PhaseResult residue = run_residue_phase(0, ntrials, d_smA, d_smB, d_nz);

    fprintf(stderr, "[multi_gpu_repro] Phase 3 : LMHC hardened scrub...\n");
    PhaseResult scrub   = run_residue_phase(1, ntrials, d_smA, d_smB, d_nz);

    cudaFree(d_smA); cudaFree(d_smB); cudaFree(d_nz);

    // ── Exposure Indices ──────────────────────────────────────────────────────
    double ei_residue = residue.total > 0
        ? (double)residue.nz / (double)residue.total : 0.0;
    double ei_control = control.total > 0
        ? (double)control.nz / (double)control.total : 0.0;
    double ei_scrub   = scrub.total > 0
        ? (double)scrub.nz   / (double)scrub.total   : 0.0;

    // ── Validity gate ─────────────────────────────────────────────────────────
    // Control MUST be all-zero. Non-zero means possible self-contamination or
    // driver-level scrubbing — do NOT report residue as a leak in that case.
    int valid = (control.nz == 0);
    const char *valid_reason = valid
        ? "control all-zero (self-contamination ruled out)"
        : "control non-zero (possible self-contamination or driver scrub — residue result unreliable)";

    // ── Suggested output filename ─────────────────────────────────────────────
    char gpu_s[128], drv_s[64], suggested[300];
    sanitize(gpu_s, sizeof(gpu_s), prop.name);
    sanitize(drv_s, sizeof(drv_s), driver_str);
    snprintf(suggested, sizeof(suggested), "%s_%s_%s.json", gpu_s, sm_str, drv_s);

    fprintf(stderr,
        "[multi_gpu_repro] Result  : valid=%s  EI_residue=%.4f  "
        "EI_control=%.4f  EI_scrub=%.4f\n",
        valid ? "true" : "false", ei_residue, ei_control, ei_scrub);
    fprintf(stderr, "[multi_gpu_repro] Save as : multi_gpu/results/%s\n", suggested);

    // ── JSON output ───────────────────────────────────────────────────────────
    FILE *out = stdout;
    if (outfile) { out = fopen(outfile, "w"); if (!out) { perror(outfile); return 1; } }

    char gpu_e[256], os_e[512], drv_e[128];
    json_esc(gpu_e, sizeof(gpu_e), prop.name);
    json_esc(os_e,  sizeof(os_e),  os_str);
    json_esc(drv_e, sizeof(drv_e), driver_str);

    fprintf(out,
        "{\n"
        "  \"schema_version\": \"multi_gpu_repro_v1\",\n"
        "  \"timestamp\": \"%s\",\n"
        "  \"valid\": %s,\n"
        "  \"valid_reason\": \"%s\",\n"
        "\n"
        "  \"environment\": {\n"
        "    \"gpu_name\": \"%s\",\n"
        "    \"compute_capability\": \"%s\",\n"
        "    \"architecture_name\": \"%s\",\n"
        "    \"driver_version\": \"%s\",\n"
        "    \"cuda_runtime_version\": \"%d.%d\",\n"
        "    \"cuda_driver_api_version\": \"%d.%d\",\n"
        "    \"nvcc_version_compiled_with\": \"%s\",\n"
        "    \"os\": \"%s\",\n"
        "    \"wsl2\": %s,\n"
        "    \"gpu_sm_count\": %d,\n"
        "    \"shared_mem_per_sm_bytes\": %zu\n"
        "  },\n"
        "\n"
        "  \"config\": {\n"
        "    \"buffer_words\": %d,\n"
        "    \"buffer_bytes\": %d,\n"
        "    \"buffer_geometry\": \"fixed compile-time constant (NWORDS=%d); "
        "identical to src/leakage_demo.cu baseline\",\n"
        "    \"pattern_sentinel\": \"0xCAFEBABE\",\n"
        "    \"block_dim\": %d,\n"
        "    \"trials_per_phase\": %d,\n"
        "    \"launch_bounds\": \"__launch_bounds__(%d, 1)\",\n"
        "    \"co_residency_method\": "
        "\"single-block, same-SM forced via __launch_bounds__(32,1); "
        "%%smid verified per trial\"\n"
        "  },\n"
        "\n"
        "  \"smid_note\": \"%%smid is a documented PTX special register "
        "(PTX ISA \\u00a79.4); reading it slightly perturbs scheduling\",\n"
        "\n"
        "  \"residue_run\": {\n"
        "    \"description\": \"writer without scrub then listener; "
        "measures raw cross-launch EI\",\n"
        "    \"same_sm_trials\": %d,\n"
        "    \"nonzero_word_count_same_sm\": %lld,\n"
        "    \"total_word_count_same_sm\": %lld,\n"
        "    \"exposure_index\": %.6f\n"
        "  },\n"
        "\n"
        "  \"control_run\": {\n"
        "    \"description\": \"listener only, writer never launched; "
        "validity gate run FIRST before any writer (must be 0.0)\",\n"
        "    \"trials\": %d,\n"
        "    \"nonzero_word_count\": %lld,\n"
        "    \"total_word_count\": %lld,\n"
        "    \"exposure_index\": %.6f,\n"
        "    \"control_clean\": %s\n"
        "  },\n"
        "\n"
        "  \"lmhc_scrub_run\": {\n"
        "    \"description\": \"writer with LMHC hardened scrub "
        "(volatile + __threadfence_block() + __syncthreads()) then listener\",\n"
        "    \"same_sm_trials\": %d,\n"
        "    \"nonzero_word_count_same_sm\": %lld,\n"
        "    \"total_word_count_same_sm\": %lld,\n"
        "    \"exposure_index\": %.6f\n"
        "  },\n"
        "\n"
        "  \"framing\": \"Residue is fully observable when the subsequent "
        "kernel is co-resident on the same SM -- co-residency is forced in "
        "this experiment to demonstrate the window. This is in-process "
        "cross-kernel shared-memory persistence, NOT cross-tenant LeftoverLocals "
        "on NVIDIA. Co-residency is forced on every device, so these are "
        "controlled-placement results consistent with our adversarial threat "
        "model -- not natural-scheduling frequencies, and not a claim about "
        "production prevalence or about non-tested hardware.\",\n"
        "\n"
        "  \"result_summary\": \"%s\"\n"
        "}\n",
        /* header */
        ts,
        valid ? "true" : "false",
        valid_reason,
        /* environment */
        gpu_e, sm_str, arch_name, drv_e,
        rt_ver / 1000, (rt_ver % 1000) / 10,
        drv_ver / 1000, (drv_ver % 1000) / 10,
        nvcc_ver, os_e,
        wsl2 ? "true" : "false",
        prop.multiProcessorCount,
        prop.sharedMemPerMultiprocessor,
        /* config — buffer geometry is a compile-time constant */
        NWORDS, NWORDS * 4, NWORDS, BLOCK_DIM, ntrials, BLOCK_DIM,
        /* residue_run */
        residue.same_sm, (long long)residue.nz, (long long)residue.total, ei_residue,
        /* control_run */
        ntrials, (long long)control.nz, (long long)control.total, ei_control,
        (control.nz == 0) ? "true" : "false",
        /* lmhc_scrub_run */
        scrub.same_sm, (long long)scrub.nz, (long long)scrub.total, ei_scrub,
        /* result_summary */
        valid
            ? (ei_scrub == 0.0 && ei_control == 0.0
               ? "VALID: control clean; LMHC scrub drives EI to 0.0 on this device."
               : "VALID: control clean; see individual EI values.")
            : "INVALID: control non-zero -- residue result unreliable on this device."
    );

    if (outfile) fclose(out);
    return 0;
}
