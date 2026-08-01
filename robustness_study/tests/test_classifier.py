#!/usr/bin/env python3
"""
test_classifier.py — task §22 requirements 6, 7, 8.

Mixes small synthetic SASS snippets (to isolate exact boundary conditions
that don't reliably occur in real compiled output, e.g. "kernel entirely
missing") with real fixtures for the cases that matter most: a genuine
FULL_SIGNATURE pilot artifact and the real multi-EXIT probe.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from sass_parser import parse_sass_file
from classify_binary import classify_kernel, classify_artifact

FIXTURES = Path(__file__).parent / "fixtures"


def make_sass(func_name: str, body_lines: list[str]) -> str:
    header = f"\t\tFunction : {func_name}\n"
    lines = "\n".join(f"        /*{i*0x10:04x}*/                   {l}" for i, l in enumerate(body_lines))
    return header + lines + "\n"


class TestSyntheticBoundaryConditions(unittest.TestCase):
    def _classify(self, func_name, body_lines):
        text = make_sass(func_name, body_lines)
        functions = parse_sass_file(text)
        return classify_kernel(next(iter(functions.values())))

    def test_absent_no_stores_at_all(self):
        r = self._classify("_Z4testv", ["EXIT ;"])
        self.assertEqual(r.category, "ABSENT")

    def test_store_only_no_fence_no_bar(self):
        r = self._classify("_Z4testv", ["STS [R0], RZ ;", "EXIT ;"])
        self.assertEqual(r.category, "STORE_ONLY")

    def test_partial_signature_fence_missing(self):
        r = self._classify("_Z4testv", ["STS [R0], RZ ;", "BAR.SYNC ;", "EXIT ;"])
        self.assertEqual(r.category, "PARTIAL_SIGNATURE")

    def test_partial_signature_bar_missing(self):
        r = self._classify("_Z4testv", ["STS [R0], RZ ;", "MEMBAR.SC.CTA ;", "EXIT ;"])
        self.assertEqual(r.category, "PARTIAL_SIGNATURE")

    def test_full_signature_correct_order(self):
        r = self._classify("_Z4testv", [
            "STS [R0], RZ ;", "MEMBAR.SC.CTA ;", "BAR.SYNC ;", "EXIT ;",
        ])
        self.assertEqual(r.category, "FULL_SIGNATURE")

    def test_partial_signature_wrong_order(self):
        # BAR.SYNC and MEMBAR before the store — order violated (N5-style case).
        r = self._classify("_Z4testv", [
            "BAR.SYNC ;", "MEMBAR.SC.CTA ;", "STS [R0], RZ ;", "EXIT ;",
        ])
        self.assertEqual(r.category, "PARTIAL_SIGNATURE")

    def test_full_signature_holds_across_three_exits(self):
        """Phase 4 finding: in the real 2,018-artifact core corpus, every
        multi-EXIT FULL_SIGNATURE case that occurred had exit_count==2 —
        the >=3-exit branch of classify_kernel's ordering check was never
        exercised by real compiled data. This synthetic case closes that
        gap: three predicated EXITs, all after the signature."""
        r = self._classify("_Z4testv", [
            "STS [R0], RZ ;", "MEMBAR.SC.CTA ;", "BAR.SYNC ;",
            "@P0 EXIT ;", "@P1 EXIT ;", "EXIT ;",
        ])
        self.assertEqual(r.category, "FULL_SIGNATURE")
        self.assertTrue(r.multi_exit_scope_assumed)

    def test_undecidable_when_signature_precedes_only_some_relevant_exits(self):
        """A relevant exit (after the last zero-store) sits BEFORE the
        barrier, while another relevant exit sits after it -> a textual
        checker cannot resolve which thread path is actually taken, so
        this must be UNDECIDABLE, not guessed either way (task §13)."""
        r = self._classify("_Z4testv", [
            "STS [R0], RZ ;", "MEMBAR.SC.CTA ;",
            "@P0 EXIT ;",      # relevant (after store), but BEFORE the barrier below
            "BAR.SYNC ;",
            "@P1 EXIT ;",      # relevant and after the barrier
        ])
        self.assertEqual(r.category, "UNDECIDABLE")

    def test_undecidable_no_exit_never_coerced_to_pass(self):
        """Task §22 requirement 8: a parser failure (no EXIT found -> scope
        cannot be bounded) must never be silently reported as a pass."""
        r = self._classify("_Z4testv", [
            "STS [R0], RZ ;", "MEMBAR.SC.CTA ;", "BAR.SYNC ;",   # looks perfect...
            # ...but there is no EXIT at all, so scope can't be bounded.
        ])
        self.assertEqual(r.category, "UNDECIDABLE")
        self.assertNotEqual(r.category, "FULL_SIGNATURE")

    def test_nonzero_store_is_not_absent_but_not_zero_store_either(self):
        # N7-style: a real (nonzero) store should not be misread as a zero-store,
        # and the kernel should read as ABSENT (no zeroization evidence), not FULL.
        r = self._classify("_Z4testv", ["STS [R0], R5 ;", "EXIT ;"])
        self.assertEqual(r.category, "ABSENT")

    def test_missing_kernel_is_undecidable_not_absent(self):
        text = make_sass("_Z4realv", ["EXIT ;"])
        r = classify_artifact(text, "kernel_that_does_not_exist")
        self.assertEqual(r.category, "UNDECIDABLE")


class TestRealFixtures(unittest.TestCase):
    def test_real_full_signature_pilot_artifact(self):
        """Z5xK2, sm_89, ptxas -O3, a real compiled+disassembled pilot artifact."""
        text = (FIXTURES / "z5_k2_sm89_O3_full_signature.sass").read_text(encoding="utf-8")
        r = classify_artifact(text, "kernel_z5_k2")
        self.assertEqual(r.category, "FULL_SIGNATURE")
        self.assertFalse(r.multi_exit_scope_assumed)

    def test_real_multi_exit_full_signature(self):
        """z6_k4_kernel: multiple SASS EXITs from single-source-exit code;
        signature must hold relative to every relevant exit."""
        text = (FIXTURES / "multi_exit_probe.sass").read_text(encoding="utf-8")
        r = classify_artifact(text, "z6_k4_kernel")
        self.assertEqual(r.category, "FULL_SIGNATURE")
        self.assertTrue(r.multi_exit_scope_assumed)

    def test_real_multi_exit_partial_signature_no_fence(self):
        """z2_k3_kernel: multiple SASS EXITs, no MEMBAR.SC.CTA by design (Z2)."""
        text = (FIXTURES / "multi_exit_probe.sass").read_text(encoding="utf-8")
        r = classify_artifact(text, "z2_k3_kernel")
        self.assertEqual(r.category, "PARTIAL_SIGNATURE")

    def test_only_valid_categories_ever_returned(self):
        from classify_binary import VALID_CATEGORIES
        text = (FIXTURES / "dse_test_original_repo.sass").read_text(encoding="utf-8")
        functions = parse_sass_file(text)
        for f in functions.values():
            r = classify_kernel(f)
            self.assertIn(r.category, VALID_CATEGORIES)


if __name__ == "__main__":
    unittest.main()
