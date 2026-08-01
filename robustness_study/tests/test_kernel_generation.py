#!/usr/bin/env python3
"""
test_kernel_generation.py — task §22 requirements 1, 2, 3.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from kernel_lib import CONSTRUCTS, CONTEXTS, Context, render_kernel
import generate_kernels
from compile_matrix import expand_cells, artifact_id

CONFIGS = Path(__file__).parent.parent / "configs"


class TestConstructsAndContexts(unittest.TestCase):
    def test_seven_constructs_six_contexts(self):
        self.assertEqual(len(CONSTRUCTS), 7)
        self.assertEqual(len(CONTEXTS), 6)
        self.assertEqual(set(CONSTRUCTS), {"Z0", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"})
        self.assertEqual(set(CONTEXTS), {"K0", "K1", "K2", "K3", "K4", "K5"})

    def test_context_nwords_divisible_by_block(self):
        for kid, ctx in CONTEXTS.items():
            with self.subTest(context=kid):
                self.assertEqual(ctx.nwords % ctx.block, 0)

    def test_bad_context_definition_rejected(self):
        with self.assertRaises(ValueError):
            Context(id="KBAD", name="bad", block=32, nwords=100, elem="uint32_t",
                    based_on="test", params="", preamble="", postamble="")

    def test_z0_produces_no_scrub_code(self):
        code = CONSTRUCTS["Z0"].scrub_code("sbuf", 1024, 32, "uint32_t")
        self.assertEqual(code, "")

    def test_z6_uses_inline_ptx(self):
        code = CONSTRUCTS["Z6"].scrub_code("sbuf", 1024, 32, "uint32_t")
        self.assertIn("asm volatile", code)
        self.assertIn("st.shared.u32", code)

    def test_z5_has_fence_and_barrier_in_order(self):
        code = CONSTRUCTS["Z5"].scrub_code("sbuf", 1024, 32, "uint32_t")
        fence_idx = code.index("__threadfence_block()")
        bar_idx = code.index("__syncthreads()")
        self.assertLess(fence_idx, bar_idx)

    def test_z2_has_no_fence_or_barrier(self):
        code = CONSTRUCTS["Z2"].scrub_code("sbuf", 1024, 32, "uint32_t")
        self.assertNotIn("__threadfence_block", code)
        self.assertNotIn("__syncthreads", code)


class TestKernelRendering(unittest.TestCase):
    def test_all_42_pairs_render_with_unique_names(self):
        names = set()
        for zid in CONSTRUCTS:
            for kid in CONTEXTS:
                fname, src = render_kernel(CONSTRUCTS[zid], CONTEXTS[kid])
                self.assertNotIn(fname, names, f"duplicate function name {fname}")
                names.add(fname)
                self.assertIn("__global__", src)
                self.assertEqual(src.count("{"), src.count("}"))
        self.assertEqual(len(names), 42)

    def test_generate_kernels_writes_unique_files(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "generated"
            meta_dir = Path(td) / "metadata"
            out_dir.mkdir(parents=True, exist_ok=True)
            meta_dir.mkdir(parents=True, exist_ok=True)
            generated = []
            for zid in CONSTRUCTS:
                for kid in CONTEXTS:
                    generated.append(generate_kernels.generate_one(zid, kid, out_dir, meta_dir))
            self.assertEqual(len(generated), 42)
            hashes = [g["source_hash_sha256"] for g in generated]
            self.assertEqual(len(hashes), len(set(hashes)), "generated sources are not all unique")
            for g in generated:
                self.assertTrue(Path(g["source_file"]).exists())


class TestMatrixExpansion(unittest.TestCase):
    def test_pilot_matrix_expands_to_42(self):
        import json
        config = json.loads((CONFIGS / "pilot_matrix.json").read_text())
        cells = expand_cells(config)
        self.assertEqual(len(cells), 42)

    def test_core_matrix_expands_to_2016(self):
        import json
        config = json.loads((CONFIGS / "core_matrix.json").read_text())
        cells = expand_cells(config)
        self.assertEqual(len(cells), 2016)
        # 7 constructs x 6 contexts x 4 opt levels x 6 archs x 2 toolkits
        self.assertEqual(len(config["pairs"]), 42)
        self.assertEqual(len(config["ptxas_opt_levels"]), 4)
        self.assertEqual(len(config["archs"]), 6)
        self.assertEqual(len(config["toolkits"]), 2)


class TestArtifactIdDeterminism(unittest.TestCase):
    def test_same_inputs_same_id(self):
        a = artifact_id("Z5", "K2", 3, "sm_89", "cuda-12.8", [])
        b = artifact_id("Z5", "K2", 3, "sm_89", "cuda-12.8", [])
        self.assertEqual(a, b)

    def test_different_inputs_different_id(self):
        a = artifact_id("Z5", "K2", 3, "sm_89", "cuda-12.8", [])
        b = artifact_id("Z5", "K2", 0, "sm_89", "cuda-12.8", [])
        c = artifact_id("Z5", "K2", 3, "sm_86", "cuda-12.8", [])
        d = artifact_id("Z5", "K2", 3, "sm_89", "cuda-12.8", ["--use_fast_math"])
        ids = {a, b, c, d}
        self.assertEqual(len(ids), 4)


if __name__ == "__main__":
    unittest.main()
