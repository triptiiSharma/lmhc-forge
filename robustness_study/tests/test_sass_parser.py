#!/usr/bin/env python3
"""
test_sass_parser.py — task §22 requirements 4, 5, 6, 7.

Uses real SASS: the original repo's dse_test.sass (independently published
ground truth in docs/RESULTS.md / results/dse_differential.json), the
Phase-1 multi-EXIT feasibility probe, and one real pilot artifact that
exposed the -O0 indirect-zero-store bug (PLAN.md §9 / this session's Phase 3
finding) — all copied into tests/fixtures/, none synthetic.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from sass_parser import demangle_simple, split_functions, parse_sass_file

FIXTURES = Path(__file__).parent / "fixtures"


class TestDemangle(unittest.TestCase):
    def test_known_names(self):
        self.assertEqual(demangle_simple("_Z11scrub_naivePj"), "scrub_naive")
        self.assertEqual(demangle_simple("_Z14scrub_hardenedPj"), "scrub_hardened")
        self.assertEqual(demangle_simple("_Z18scrub_ptx_fallbackPj"), "scrub_ptx_fallback")
        self.assertEqual(demangle_simple("_Z12z2_k3_kernelPKfPfi"), "z2_k3_kernel")


class TestOriginalRepoFixture(unittest.TestCase):
    """Cross-check against docs/RESULTS.md / results/dse_differential.json,
    which this study must not silently contradict (AUDIT.md §3)."""

    @classmethod
    def setUpClass(cls):
        text = (FIXTURES / "dse_test_original_repo.sass").read_text(encoding="utf-8")
        cls.functions = parse_sass_file(text)
        cls.by_name = {f.demangled_name: f for f in cls.functions.values()}

    def test_three_functions_found(self):
        self.assertEqual(
            set(self.by_name.keys()),
            {"scrub_naive", "scrub_hardened", "scrub_ptx_fallback"},
        )

    def test_total_sts_counts_match_published_evidence(self):
        # dse_differential.json: 58 STS each (total, zero + nonzero).
        for name in ("scrub_naive", "scrub_hardened", "scrub_ptx_fallback"):
            with self.subTest(name=name):
                self.assertEqual(self.by_name[name].shared_store_count, 58)

    def test_zero_store_counts_match_published_evidence(self):
        # RESULTS.md: "the 29 scrub zero-stores survive... in ALL THREE versions".
        for name in ("scrub_naive", "scrub_hardened", "scrub_ptx_fallback"):
            with self.subTest(name=name):
                self.assertEqual(self.by_name[name].zero_shared_store_count, 29)

    def test_membar_bar_ordering_differential_matches_published_evidence(self):
        naive = self.by_name["scrub_naive"]
        hardened = self.by_name["scrub_hardened"]
        ptx_fb = self.by_name["scrub_ptx_fallback"]
        self.assertEqual(naive.membar_cta_count, 0)
        self.assertEqual(naive.bar_sync_count, 2)
        self.assertEqual(hardened.membar_cta_count, 1)
        self.assertEqual(hardened.bar_sync_count, 3)
        self.assertEqual(ptx_fb.membar_cta_count, 1)
        self.assertEqual(ptx_fb.bar_sync_count, 3)


class TestMultiExitFixture(unittest.TestCase):
    """PLAN.md §9: a single-source-exit kernel that compiles to multiple
    SASS EXIT instructions, empirically confirmed correct on real hardware."""

    @classmethod
    def setUpClass(cls):
        text = (FIXTURES / "multi_exit_probe.sass").read_text(encoding="utf-8")
        cls.functions = parse_sass_file(text)
        cls.by_name = {f.demangled_name: f for f in cls.functions.values()}

    def test_both_kernels_found(self):
        self.assertIn("z2_k3_kernel", self.by_name)
        self.assertIn("z6_k4_kernel", self.by_name)

    def test_z2_k3_has_multiple_exits_and_no_fence(self):
        f = self.by_name["z2_k3_kernel"]
        self.assertGreater(f.exit_count, 1)
        self.assertGreater(f.zero_shared_store_count, 0)
        self.assertEqual(f.membar_cta_count, 0)   # Z2 has no threadfence by design
        self.assertGreater(f.bar_sync_count, 0)   # from K3's own reduction-tree syncs

    def test_z6_k4_has_multiple_exits_and_full_fence(self):
        f = self.by_name["z6_k4_kernel"]
        self.assertGreater(f.exit_count, 1)
        self.assertGreater(f.zero_shared_store_count, 0)
        self.assertEqual(f.membar_cta_count, 1)
        self.assertGreater(f.bar_sync_count, 0)


class TestIndirectZeroStoreO0(unittest.TestCase):
    """The real bug this session found and fixed: at -O0, ptxas emits
    `MOV Rn, RZ` + `STS [...], Rn` instead of `STS [...], RZ` directly.
    Manually verified against this exact fixture's raw SASS (all 5 stores
    traced back to an un-redefined MOV Rn,RZ before their use)."""

    def test_five_indirect_zero_stores_recognized(self):
        text = (FIXTURES / "z2_k0_sm89_O0_indirect_zero_store.sass").read_text(encoding="utf-8")
        functions = parse_sass_file(text)
        f = next(iter(functions.values()))
        self.assertEqual(f.zero_shared_store_count, 5)
        self.assertEqual(f.zero_shared_store_direct_count, 0)
        self.assertEqual(f.zero_shared_store_indirect_count, 5)


class TestSplitFunctionsBoundaries(unittest.TestCase):
    def test_no_cross_contamination_between_kernels(self):
        text = (FIXTURES / "dse_test_original_repo.sass").read_text(encoding="utf-8")
        functions = split_functions(text)
        self.assertEqual(len(functions), 3)
        for name, lines in functions.items():
            for line in lines:
                self.assertNotIn("Function :", line)


if __name__ == "__main__":
    unittest.main()
