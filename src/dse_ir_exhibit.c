/* dse_ir_exhibit.c -- RP Section VI.C "DSE-possibility exhibit"
 *
 * NOT a CUDA/NVVM-specific test (nvcc's internal device-side LLVM pipeline is
 * not independently invokable with `opt` outside the closed-source ptxas
 * back end). This is a generic C / standard-LLVM-IR analog of the IDENTICAL
 * semantic hazard Module 3 characterizes on ptxas: a zeroing store to a
 * buffer never read again is a dead store an optimizing compiler is
 * PERMITTED to delete (Theorem 1's abstract-machine argument; Yang et al.,
 * "Dead store elimination (still) considered harmful", USENIX Security 2017,
 * RP ref [15]) -- independent of whether the specific back end (ptxas today,
 * any future one) chooses to exercise that permission.
 *
 * fill_read_scrub_naive: plain pointer loop, buffer never read after the
 *   scrub loop -> a standard DSE pass is free to delete the scrub stores.
 * fill_read_scrub_hardened: volatile-qualified pointer -> the C abstract
 *   machine treats these as observable side effects; DSE MUST NOT remove
 *   them.
 *
 * Build (see RESULTS.md Module 3 / DSE-IR-exhibit section for the exact
 * commands and machine-checked store counts):
 *   clang -O0 -S -emit-llvm -o dse_ir_exhibit.ll dse_ir_exhibit.c
 *   opt -passes=dse -S dse_ir_exhibit.ll -o dse_ir_exhibit.dse.ll
 *   grep -c "store " dse_ir_exhibit.dse.ll   (per function, isolated)
 */

#define NWORDS 1024

/* sink is a global the compiler cannot prove unread across translation
 * units; it is where Phase 2's "output" read goes, mirroring dse_test.cu's
 * `*output = buf[0] ^ buf[NWORDS-1]`. */
volatile unsigned int sink;

/* buf is declared INSIDE each function (stack-local, non-escaping) rather
 * than passed as a pointer parameter -- this is the correct analog of CUDA
 * __shared__ buf[NWORDS]: memory that does not escape the function/kernel,
 * so a whole-function optimizer CAN legally prove no read follows the scrub.
 * A pointer PARAMETER would be caller-visible/escaping and is not a fair
 * analog (mismatched escape analysis would make the comparison meaningless).
 */

/* ---- naive: scrub is a plain store loop, buffer dead after ---- */
__attribute__((noinline)) void fill_read_scrub_naive(void) {
    unsigned int buf[NWORDS];
    int i;
    /* Phase 1: fill (simulated sensitive computation) */
    for (i = 0; i < NWORDS; i++)
        buf[i] = (unsigned int)(i * 1234567u) ^ 0xDEADBEEFu;
    /* Phase 2: produce output (reads from buf) */
    sink = buf[0] ^ buf[NWORDS - 1];
    /* Phase 3: NAIVE SCRUB -- buf is never read again after this loop, and
     * buf does not escape this function (no return, no pointer stored
     * anywhere externally). A standard DSE pass is free to delete every
     * store below. */
    for (i = 0; i < NWORDS; i++)
        buf[i] = 0u;
    /* <- function returns; buf's storage (stack) is reclaimed. */
}

/* ---- hardened: scrub through a volatile-qualified pointer ---- */
__attribute__((noinline)) void fill_read_scrub_hardened(void) {
    unsigned int buf[NWORDS];
    volatile unsigned int *vbuf = (volatile unsigned int *)buf;
    int i;
    for (i = 0; i < NWORDS; i++)
        buf[i] = (unsigned int)(i * 1234567u) ^ 0xDEADBEEFu;
    sink = buf[0] ^ buf[NWORDS - 1];
    /* HARDENED SCRUB: volatile stores are observable side effects under the
     * C abstract machine; a conforming optimizer may not remove them. */
    for (i = 0; i < NWORDS; i++)
        vbuf[i] = 0u;
}
