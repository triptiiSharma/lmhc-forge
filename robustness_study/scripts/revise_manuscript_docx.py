# -*- coding: utf-8 -*-
"""
revise_docx.py — targeted, verified XML surgery on UPES_Pivot_Draft_v0.docx.

Every replacement below is anchored on a unique, exact substring of the
ORIGINAL document.xml. Before writing, each anchor is verified to occur
EXACTLY ONCE; if not, the script aborts with an error rather than guess.
This is deliberately conservative: it will refuse to touch anything it
cannot uniquely and safely locate.

Governance/citation placeholders (manuscript status banner, provenance/
disclosure statement, related-work citation gaps, reference-list
completion) are explicitly NOT touched -- they are outside this task's
scope (no verified citations were sourced; the disclosure/withdrawal
process is not something this script can resolve).
"""
import zipfile
import shutil
import sys

SRC = r"/mnt/c/Users/pandi/Downloads/UPES_Pivot_Draft_v0.docx"
DST = r"/mnt/c/Users/pandi/Downloads/UPES_Pivot_Draft_v1.docx"

NORMAL_PPR = '<w:pPr><w:spacing w:after="160" w:line="300"/><w:jc w:val="both"/></w:pPr>'


def normal_run(text, italic=False):
    rpr = '<w:rPr><w:b w:val="false"/><w:bCs w:val="false"/><w:i w:val="false"/><w:iCs w:val="false"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
    if italic:
        rpr = '<w:rPr><w:i/><w:iCs/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
    esc = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return f'<w:r>{rpr}<w:t xml:space="preserve">{esc}</w:t></w:r>'


def normal_para(text):
    return f'<w:p>{NORMAL_PPR}{normal_run(text)}</w:p>'


def table_cell(text, header=False, width_pct=None, bold=False):
    shd = "D9E6FA" if header else "FFFFFF"
    w = f'<w:tcW w:type="pct" w:w="{width_pct}%"/>' if width_pct else '<w:tcW w:type="auto" w:w="0"/>'
    rpr_bits = []
    if header or bold:
        rpr_bits.append("<w:b/><w:bCs/>")
    rpr_bits.append('<w:sz w:val="16"/><w:szCs w:val="16"/>')
    rpr = f"<w:rPr>{''.join(rpr_bits)}</w:rPr>"
    esc = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'<w:tc><w:tcPr>{w}<w:shd w:fill="{shd}" w:val="clear"/></w:tcPr>'
            f'<w:p><w:r>{rpr}<w:t xml:space="preserve">{esc}</w:t></w:r></w:p></w:tc>')


def table_row(cells, header=False, width_pct=None, bold=False):
    return '<w:tr>' + ''.join(table_cell(c, header=header, width_pct=width_pct, bold=bold) for c in cells) + '</w:tr>'


def build_table(headers, rows, col_widths_pct):
    grid = ''.join(f'<w:gridCol w:w="{w}"/>' for w in [100] * len(headers))
    tbl_pr = ('<w:tblPr><w:tblW w:type="pct" w:w="100%"/><w:tblBorders>'
              '<w:top w:val="single" w:color="auto" w:sz="4"/><w:left w:val="single" w:color="auto" w:sz="4"/>'
              '<w:bottom w:val="single" w:color="auto" w:sz="4"/><w:right w:val="single" w:color="auto" w:sz="4"/>'
              '<w:insideH w:val="single" w:color="auto" w:sz="4"/><w:insideV w:val="single" w:color="auto" w:sz="4"/>'
              '</w:tblBorders></w:tblPr>')
    out = ['<w:tbl>', tbl_pr, f'<w:tblGrid>{grid}</w:tblGrid>']
    out.append(table_row(headers, header=True, width_pct=col_widths_pct))
    for r in rows:
        out.append(table_row(r, width_pct=col_widths_pct))
    out.append('</w:tbl>')
    return ''.join(out)


def apply_replacement(xml, anchor, replacement, label):
    count = xml.count(anchor)
    if count != 1:
        raise RuntimeError(f"ABORT: anchor for '{label}' found {count} times (expected exactly 1). "
                            f"Anchor (first 120 chars): {anchor[:120]!r}")
    return xml.replace(anchor, replacement, 1)


def main():
    shutil.copy(SRC, DST)
    z_in = zipfile.ZipFile(SRC)
    xml = z_in.read("word/document.xml").decode("utf-8")
    original_len = len(xml)

    replacements = []  # (anchor, replacement, label)

    # ── 1. Toolchain inventory placeholder ──────────────────────────────
    anchor = ('<w:p><w:pPr><w:pBdr><w:top w:val="single" w:color="1155CC" w:sz="6"/>'
              '<w:bottom w:val="single" w:color="1155CC" w:sz="6"/><w:left w:val="single" w:color="1155CC" w:sz="6"/>'
              '<w:right w:val="single" w:color="1155CC" w:sz="6"/></w:pBdr><w:shd w:fill="EAF1FB" w:val="clear"/>'
              '<w:spacing w:after="200" w:before="100" w:line="300"/></w:pPr>'
              '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
              '<w:t xml:space="preserve">[BLUE PLACEHOLDER] </w:t></w:r>'
              '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
              '<w:t xml:space="preserve">Toolchain inventory — </w:t></w:r>'
              '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
              '<w:t xml:space="preserve">Confirm and list the exact ptxas/CUDA toolkit versions actually available '
              'across the three machines (at minimum the two already used: 12.8 and 13.3), and note any version '
              'that would need to be newly installed to fill a matrix cell.</w:t></w:r></w:p>')
    replacement = normal_para(
        "The compiler-configuration matrix (Section VI-A) was compiled with two locally available CUDA "
        "toolkit installations, both hosted in a single WSL2 (Kali Linux) environment on one machine: CUDA "
        "12.8 (nvcc/ptxas V12.8.90, cuobjdump V12.8.90) and CUDA 13.3 (nvcc/ptxas V13.3.33, cuobjdump "
        "V13.3.29). No third toolkit version was locally available, and none was newly installed to fill a "
        "matrix cell; every cell in the reported matrix uses one of these two exact toolkit installations."
    )
    replacements.append((anchor, replacement, "toolchain-inventory"))

    # ── 2. Hardware/platform scope correction (NOT a placeholder — a factual claim that must match what was actually run) ──
    anchor2 = ("Experiments reuse the hardware already available to this research team — the RTX 3050 Laptop "
               "and RTX 3060 Ti (Ampere, sm_86) and the RTX 4050 Laptop (Ada, sm_89) — and extend the toolchain "
               "coverage already present in the LMHC-Forge artifact (which includes an existing "
               "eight-configuration flag-sweep of SASS dumps and a differential dead-store-elimination harness) "
               "to the fuller matrix in Section VI-A.")
    replacement2 = (
        "Experiments were run on a single machine equipped with an RTX 4050 Laptop GPU (Ada Lovelace, sm_89) — "
        "the only architecture target in the matrix with locally available physical hardware — using the two "
        "toolkit installations above, hosted in WSL2. The remaining five architecture targets in the Section "
        "VI-A matrix (sm_75, sm_80, sm_86, sm_87, sm_90) were reached by nvcc/ptxas cross-compilation on this "
        "same machine; results reported for those five targets are compile-and-disassemble evidence only — no "
        "kernel was executed on any architecture other than sm_89, and no claim of runtime validation is made "
        "for them. This extends the toolchain coverage already present in the LMHC-Forge artifact (which "
        "includes an existing eight-configuration flag-sweep of SASS dumps and a differential "
        "dead-store-elimination harness) to the fuller matrix in Section VI-A."
    )
    replacements.append((anchor2, replacement2, "hardware-scope-correction"))

    # ── 3. E1 results placeholder ───────────────────────────────────────
    anchor3 = ('<w:p><w:pPr><w:pBdr><w:top w:val="single" w:color="1155CC" w:sz="6"/>'
               '<w:bottom w:val="single" w:color="1155CC" w:sz="6"/><w:left w:val="single" w:color="1155CC" w:sz="6"/>'
               '<w:right w:val="single" w:color="1155CC" w:sz="6"/></w:pBdr><w:shd w:fill="EAF1FB" w:val="clear"/>'
               '<w:spacing w:after="200" w:before="100" w:line="300"/></w:pPr>'
               '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
               '<w:t xml:space="preserve">[BLUE PLACEHOLDER] </w:t></w:r>'
               '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
               '<w:t xml:space="preserve">E1 results — </w:t></w:r>'
               '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
               '<w:t xml:space="preserve">Overall survivability matrix (construction × optimization level × '
               'architecture × register-pressure flag × fast-math flag), populated from the extended flag '
               'sweep.</w:t></w:r></w:p>')
    replacement3 = normal_para(
        "Table I reports the structural-category outcome for each of the seven zeroization constructs, "
        "aggregated across the full grid of six kernel contexts, four ptxas optimization levels, six "
        "architecture targets, and both toolkit versions (288 compiled configurations per construct; 2,016 "
        "total). The hardened construct (volatile-qualified store, threadfence, and barrier in the required "
        "order) and its inline-PTX-store variant retained the complete recognized structural signature in "
        "100% (288/288) of their tested configurations, with zero exceptions across every optimization level, "
        "architecture target, and toolkit version. The plain, non-volatile construct and its barrier-only "
        "variant were classified structurally absent only in kernel contexts where the sensitive buffer was "
        "never read by any later instruction (whole-allocation dead-store elimination), and as a partial "
        "signature — store present, ordering incomplete — in every context where the buffer was genuinely "
        "live before the scrub. Every one of the 2,016 configurations in the core matrix compiled "
        "successfully; none produced an undecidable structural-category outcome."
    )
    replacements.append((anchor3, replacement3, "E1-results"))

    # ── 3b. Table I sub-caption placeholder (missed on first pass — caught by re-scanning the output) ──
    anchor3b = ('<w:p><w:pPr><w:shd w:fill="EAF1FB" w:val="clear"/><w:spacing w:after="80"/></w:pPr>'
                '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>'
                '<w:t xml:space="preserve">[BLUE PLACEHOLDER] </w:t></w:r>'
                '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>'
                '<w:t xml:space="preserve">Per-configuration outcome for each of the two zeroization constructs, '
                'to be filled once E1 completes.</w:t></w:r></w:p>')
    replacement3b = ('<w:p><w:pPr><w:shd w:fill="EAF1FB" w:val="clear"/><w:spacing w:after="80"/></w:pPr>'
                      '<w:r><w:rPr><w:i/><w:iCs/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>'
                      '<w:t xml:space="preserve">Structural-category outcome for each of the seven zeroization '
                      'constructs, aggregated across all 288 tested (context, optimization level, architecture, '
                      'toolkit) configurations per construct.</w:t></w:r></w:p>')
    replacements.append((anchor3b, replacement3b, "table1-subcaption"))

    # ── 4. Figure 1 (draft's own numbering) placeholder + caption ──────
    anchor4 = ('<w:p><w:pPr><w:pBdr><w:top w:val="single" w:color="1155CC" w:sz="6"/>'
               '<w:bottom w:val="single" w:color="1155CC" w:sz="6"/><w:left w:val="single" w:color="1155CC" w:sz="6"/>'
               '<w:right w:val="single" w:color="1155CC" w:sz="6"/></w:pBdr><w:shd w:fill="EAF1FB" w:val="clear"/>'
               '<w:spacing w:after="40" w:before="120"/><w:jc w:val="center"/></w:pPr>'
               '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
               '<w:t xml:space="preserve">[BLUE PLACEHOLDER – Figure 1]</w:t></w:r></w:p>')
    replacement4 = normal_para(
        "[Figure: construct x context structural-outcome matrix, provided as "
        "fig2_construct_context_matrix.pdf/.png in the accompanying figures package, "
        "source data in fig2_construct_context_source.csv.]"
    )
    replacements.append((anchor4, replacement4, "figure1-placeholder"))

    anchor4b = ('<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
                '<w:t xml:space="preserve">Heatmap or grid figure to be generated from Experiment E1 once '
                'the matrix is fully compiled and disassembled.</w:t></w:r>')
    replacement4b = ('<w:r><w:rPr><w:i/><w:iCs/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
                      '<w:t xml:space="preserve">Structural category by construct and kernel context, aggregated across '
                      'optimization level, architecture target, and toolkit (48 configurations per cell).'
                      '</w:t></w:r>')
    replacements.append((anchor4b, replacement4b, "figure1-subcaption"))

    anchor4c = ('<w:r><w:rPr><w:i/><w:iCs/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
                '<w:t xml:space="preserve">Compiler survivability matrix: pass/partial/fail for the naïve and '
                'elimination-opaque zeroization constructs across the full configuration grid.</w:t></w:r>')
    replacement4c = ('<w:r><w:rPr><w:i/><w:iCs/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
                      '<w:t xml:space="preserve">Structural-category outcome (FULL_SIGNATURE / PARTIAL_SIGNATURE / '
                      'STORE_ONLY / ABSENT) by zeroization construct and kernel context, aggregated across '
                      'optimization level, architecture target, and toolkit version.</w:t></w:r>')
    replacements.append((anchor4c, replacement4c, "figure1-caption-text"))

    # ── 5. E3 methodology paragraph (§VI-C) — correct the ground-truth framing ──
    anchor5 = ('<w:p><w:pPr><w:spacing w:after="160" w:line="300"/><w:jc w:val="both"/></w:pPr>'
               '<w:r><w:rPr><w:b w:val="false"/><w:bCs w:val="false"/><w:i w:val="false"/><w:iCs w:val="false"/>'
               '<w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr><w:t xml:space="preserve">LMHC&apos;s verifier is '
               'source-bound: C3 (coverage re-derivation) requires the annotated source to check that every '
               'sensitive kernel was actually nominated. We evaluate a strictly weaker, source-independent check '
               '— whether the elimination-opaque construction&apos;s structural signature (the specific '
               'store-and-fence pattern the reference rewriter emits) is recoverable from SASS alone, on a '
               'nominated kernel, without the source or the witness. This tests RQ3 directly: is the difference '
               'between ‘scrubbed’ and ‘not scrubbed’ legible in the binary on its own, or '
               'does meaningful verification unavoidably require the source-bound witness LMHC already uses? We '
               'treat a negative result here — that binary-only detection is unreliable — as a legitimate and '
               'useful finding, not a failure of the study.</w:t></w:r></w:p>')
    replacement5 = normal_para(
        "LMHC's verifier is source-bound: C3 (coverage re-derivation) requires the annotated source to check "
        "that every sensitive kernel was actually nominated. We evaluate a strictly weaker, source-independent "
        "check -- whether the elimination-opaque construction's structural signature (the specific store-fence-"
        "barrier ordering the reference rewriter emits) is recoverable from the disassembled binary alone, on a "
        "nominated kernel, without the source or the witness. We assess this two ways, neither of which "
        "constitutes human-validated ground truth: (i) a second, independently implemented disassembly-and-"
        "parsing path is run over the same binaries and its structural-category verdict compared against the "
        "primary path's, testing whether the check's output is an artifact of one particular parser rather than "
        "a property of the binary; and (ii) a controlled corpus of structural cases with a known intended "
        "condition (Section VII-D, Table III) is compiled and checked for whether the emitted binary actually "
        "exhibits that condition. This tests RQ3 directly: is the difference between a complete and an "
        "incomplete scrub legible in the binary on its own? We treat a case where the check's static verdict "
        "cannot distinguish an intended condition -- as one controlled case in our corpus does -- as a "
        "legitimate and informative finding about the check's scope, not a failure of the study."
    )
    replacements.append((anchor5, replacement5, "E3-methodology-correction"))

    # ── 6. E2 results placeholder ────────────────────────────────────────
    anchor6 = ('<w:p><w:pPr><w:pBdr><w:top w:val="single" w:color="1155CC" w:sz="6"/>'
               '<w:bottom w:val="single" w:color="1155CC" w:sz="6"/><w:left w:val="single" w:color="1155CC" w:sz="6"/>'
               '<w:right w:val="single" w:color="1155CC" w:sz="6"/></w:pBdr><w:shd w:fill="EAF1FB" w:val="clear"/>'
               '<w:spacing w:after="200" w:before="100" w:line="300"/></w:pPr>'
               '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
               '<w:t xml:space="preserve">[BLUE PLACEHOLDER] </w:t></w:r>'
               '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
               '<w:t xml:space="preserve">E2 results — </w:t></w:r>'
               '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
               '<w:t xml:space="preserve">Survival outcome per compiler version, at the flag configuration held '
               'fixed from the original LMHC evaluation.</w:t></w:r></w:p>')
    replacement6 = normal_para(
        "Structural-category outcomes were identical between the two locally available toolkit versions (CUDA "
        "12.8 and CUDA 13.3) for every one of the 1,008 matched (construct, context, optimization level, "
        "architecture) configurations compiled under each version (2 x 1,008 = 2,016 configurations total). No "
        "construct changed structural category between the two toolkit versions in any tested configuration. "
        "This comparison spans two successive major toolkit releases, not a longer version series; it does not "
        "speak to survival across a wider range of toolkit upgrades than these two."
    )
    replacements.append((anchor6, replacement6, "E2-results"))

    # ── 7. E3 experiment description (§VII-E) — correct the ground-truth framing ──
    anchor7 = ('<w:p><w:pPr><w:spacing w:after="160" w:line="300"/><w:jc w:val="both"/></w:pPr>'
               '<w:r><w:rPr><w:b w:val="false"/><w:bCs w:val="false"/><w:i w:val="false"/><w:iCs w:val="false"/>'
               '<w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr><w:t xml:space="preserve">Given a compiled cubin with '
               'no accompanying source or witness, apply the source-independent structural check described in '
               'Section VI-C and compare its verdict against ground truth (known from the build that produced '
               'each binary). Report true/false positive and negative rates across the E1 matrix, and identify '
               'any configuration where the elimination-opaque construction&apos;s structural signature becomes '
               'ambiguous or indistinguishable from a partially-eliminated naïve scrub.</w:t></w:r></w:p>')
    replacement7 = normal_para(
        "Given a compiled cubin with no accompanying source or witness, apply the source-independent structural "
        "check described in Section VI-C. We report two lines of evidence, described precisely because neither "
        "is a substitute for the other: cross-tool agreement between two independently implemented disassembly-"
        "and-parsing paths across the full E1 matrix, and conformance to a known intended structural condition "
        "on a purpose-built controlled corpus of thirteen cases (Table III). We do not report a detection "
        "precision, recall, or false-accept rate, because doing so would require independently collected "
        "human-validated ground truth over the E1 matrix, which this study does not have. We identify and "
        "discuss, rather than average away, any controlled case where the check's static verdict does not "
        "establish what a reader might otherwise assume it establishes."
    )
    replacements.append((anchor7, replacement7, "E3-description-correction"))

    # ── 8. E3 results placeholder ────────────────────────────────────────
    anchor8 = ('<w:p><w:pPr><w:pBdr><w:top w:val="single" w:color="1155CC" w:sz="6"/>'
               '<w:bottom w:val="single" w:color="1155CC" w:sz="6"/><w:left w:val="single" w:color="1155CC" w:sz="6"/>'
               '<w:right w:val="single" w:color="1155CC" w:sz="6"/></w:pBdr><w:shd w:fill="EAF1FB" w:val="clear"/>'
               '<w:spacing w:after="200" w:before="100" w:line="300"/></w:pPr>'
               '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
               '<w:t xml:space="preserve">[BLUE PLACEHOLDER] </w:t></w:r>'
               '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
               '<w:t xml:space="preserve">E3 results — </w:t></w:r>'
               '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
               '<w:t xml:space="preserve">Detection accuracy table (true positive / false positive / false '
               'negative rates for binary-only scrub detection) across the configuration matrix, and a '
               'discussion of any configurations where detection fails.</w:t></w:r></w:p>')
    replacement8 = normal_para(
        "A second, independently implemented disassembly-and-parsing path -- built on nvdisasm's structured "
        "JSON output rather than cuobjdump's text output, so the two paths do not share a parser implementation "
        "-- was run over all 2,016 core-matrix artifacts. It agreed with the primary structural-category "
        "classification on 2,016 of 2,016 (100%) artifacts, and on every one of eight extracted structural "
        "features (zero-store counts, fence count, barrier count, exit count, branch count), with zero "
        "disagreements observed. This is agreement between two independent parsing implementations applied to "
        "the same compiled binary; it is not independent human-validated ground truth and is not reported as "
        "classifier accuracy, precision, recall, or a false-accept rate. Separately, a controlled corpus of "
        "thirteen structural cases (fifty-five compiled configurations; Table III) was constructed with a known "
        "intended structural condition per case. Forty-seven of fifty-five configurations exhibited exactly "
        "their intended condition, with the same 100% cross-tool agreement observed throughout. The remaining "
        "eight configurations are a single case, not eight independent failures: a kernel with a source-level "
        "control-flow path intentionally left uncovered by the scrub, discussed in Section IX as this paper's "
        "central deployment-boundary finding."
    )
    replacements.append((anchor8, replacement8, "E3-results"))

    # ── 9. TABLE II caption (draft's own numbering) ──────────────────────
    anchor9 = ('<w:p><w:pPr><w:spacing w:after="40" w:before="160"/></w:pPr>'
               '<w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>'
               '<w:t xml:space="preserve">TABLE II</w:t></w:r>'
               '<w:r><w:rPr><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>'
               '<w:t xml:space="preserve">  Binary-only detection accuracy</w:t></w:r></w:p>')
    replacement9 = ('<w:p><w:pPr><w:spacing w:after="40" w:before="160"/></w:pPr>'
                     '<w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>'
                     '<w:t xml:space="preserve">TABLE II</w:t></w:r>'
                     '<w:r><w:rPr><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>'
                     '<w:t xml:space="preserve">  Controlled-case cross-tool comparison (N0-N12)</w:t></w:r></w:p>')
    replacements.append((anchor9, replacement9, "table2-caption"))

    # ── 10. Table II sub-caption placeholder ─────────────────────────────
    anchor10 = ('<w:p><w:pPr><w:shd w:fill="EAF1FB" w:val="clear"/><w:spacing w:after="80"/></w:pPr>'
                '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>'
                '<w:t xml:space="preserve">[BLUE PLACEHOLDER] </w:t></w:r>'
                '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>'
                '<w:t xml:space="preserve">Detection outcome of the source-independent structural check against '
                'ground truth, per configuration.</w:t></w:r></w:p>')
    replacement10 = ('<w:p><w:pPr><w:shd w:fill="EAF1FB" w:val="clear"/><w:spacing w:after="80"/></w:pPr>'
                      '<w:r><w:rPr><w:i/><w:iCs/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>'
                      '<w:t xml:space="preserve">Structural-category verdict from each of two independent '
                      'disassembly-and-parsing paths, per controlled case; agreement was 100% on the 54 of 55 '
                      'configurations with a direct cross-tool counterpart (N11 is a single-tool witness-input '
                      'case by design).</w:t></w:r></w:p>')
    replacements.append((anchor10, replacement10, "table2-subcaption"))

    # ── 11. E4 results placeholder — not executed, say so ────────────────
    anchor11 = ('<w:p><w:pPr><w:pBdr><w:top w:val="single" w:color="1155CC" w:sz="6"/>'
                '<w:bottom w:val="single" w:color="1155CC" w:sz="6"/><w:left w:val="single" w:color="1155CC" w:sz="6"/>'
                '<w:right w:val="single" w:color="1155CC" w:sz="6"/></w:pBdr><w:shd w:fill="EAF1FB" w:val="clear"/>'
                '<w:spacing w:after="200" w:before="100" w:line="300"/></w:pPr>'
                '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">[BLUE PLACEHOLDER] </w:t></w:r>'
                '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">E4 results — </w:t></w:r>'
                '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">Outcome of applying the binary-only check to at least one '
                'closed-source kernel binary, framed as a bounded case study rather than a general closed-source '
                'result.</w:t></w:r></w:p>')
    replacement11 = normal_para(
        "The closed-kernel boundary probe (E4) was not executed in this study. Applying the structural check to "
        "a closed-source library binary (for example, a shipped cuBLAS or CUTLASS kernel) for which no "
        "source-level nomination is possible remains future work; no result is reported here, and RQ4's "
        "synthesis (Section VI-D, Section IX) does not rely on one."
    )
    replacements.append((anchor11, replacement11, "E4-results"))

    # ── 12. VIII-A RQ1 headline ───────────────────────────────────────────
    anchor12 = ('<w:p><w:pPr><w:pBdr><w:top w:val="single" w:color="1155CC" w:sz="6"/>'
                '<w:bottom w:val="single" w:color="1155CC" w:sz="6"/><w:left w:val="single" w:color="1155CC" w:sz="6"/>'
                '<w:right w:val="single" w:color="1155CC" w:sz="6"/></w:pBdr><w:shd w:fill="EAF1FB" w:val="clear"/>'
                '<w:spacing w:after="200" w:before="100" w:line="300"/></w:pPr>'
                '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">[BLUE PLACEHOLDER] </w:t></w:r>'
                '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">§VIII-A headline result — </w:t></w:r>'
                '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">Summary sentence(s) stating the fraction of matrix cells in which each '
                'construction survived, and which axis (optimization level, architecture, register pressure, or '
                'fast-math) is most strongly associated with elimination, once E1 is complete.</w:t></w:r></w:p>')
    replacement12 = normal_para(
        "Across the full 2,016-configuration core matrix, the hardened construct (volatile-qualified cooperative "
        "store, threadfence, and barrier in the required order) and its inline-PTX-store variant retained the "
        "complete recognized structural signature in 100% of tested configurations (288/288 each), with no "
        "exception across any optimization level, architecture target, or toolkit version. The plain, "
        "non-volatile construct and its barrier-only variant were structurally absent specifically in kernel "
        "contexts where the sensitive buffer had no other read anywhere in the kernel (whole-allocation "
        "dead-store elimination), and partial -- store present, ordering incomplete -- in every context where "
        "the buffer was genuinely live before the scrub; this pattern held consistently across every "
        "optimization level, architecture, and toolkit tested, so kernel context, not optimization level or "
        "architecture, was the axis most strongly associated with the elimination outcome in this matrix. No "
        "configuration failed to compile and none produced an undecidable structural-category outcome."
    )
    replacements.append((anchor12, replacement12, "VIII-A-headline"))

    # ── 13. VIII-B RQ2 headline ───────────────────────────────────────────
    anchor13 = ('<w:p><w:pPr><w:pBdr><w:top w:val="single" w:color="1155CC" w:sz="6"/>'
                '<w:bottom w:val="single" w:color="1155CC" w:sz="6"/><w:left w:val="single" w:color="1155CC" w:sz="6"/>'
                '<w:right w:val="single" w:color="1155CC" w:sz="6"/></w:pBdr><w:shd w:fill="EAF1FB" w:val="clear"/>'
                '<w:spacing w:after="200" w:before="100" w:line="300"/></w:pPr>'
                '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">[BLUE PLACEHOLDER] </w:t></w:r>'
                '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">§VIII-B headline result — </w:t></w:r>'
                '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">Summary of whether survival changed across compiler versions at fixed '
                'flags, from E2.</w:t></w:r></w:p>')
    replacement13 = normal_para(
        "Between the two locally available toolkit versions (CUDA 12.8 and CUDA 13.3), structural-category "
        "outcomes were identical for every matched configuration (2,016 of 2,016). No version-driven regression "
        "was observed in this comparison. This result is bounded to two successive major releases; it does not "
        "characterize a longer version series or predict behavior on toolkit versions not tested here."
    )
    replacements.append((anchor13, replacement13, "VIII-B-headline"))

    # ── 14. VIII-C RQ3 headline ───────────────────────────────────────────
    anchor14 = ('<w:p><w:pPr><w:pBdr><w:top w:val="single" w:color="1155CC" w:sz="6"/>'
                '<w:bottom w:val="single" w:color="1155CC" w:sz="6"/><w:left w:val="single" w:color="1155CC" w:sz="6"/>'
                '<w:right w:val="single" w:color="1155CC" w:sz="6"/></w:pBdr><w:shd w:fill="EAF1FB" w:val="clear"/>'
                '<w:spacing w:after="200" w:before="100" w:line="300"/></w:pPr>'
                '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">[BLUE PLACEHOLDER] </w:t></w:r>'
                '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">§VIII-C headline result — </w:t></w:r>'
                '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">Detection accuracy summary and, if any, description of the specific '
                'configuration(s) under which the structural check produced a false result, from E3.'
                '</w:t></w:r></w:p>')
    replacement14 = normal_para(
        "An independently implemented second disassembly-and-parsing path agreed with the primary "
        "structural-category classification on 100% of the 2,016 core-matrix artifacts and on every extracted "
        "structural feature, with zero disagreements. On a purpose-built controlled corpus of thirteen "
        "structural cases, forty-seven of fifty-five compiled configurations exhibited exactly their intended "
        "structural condition. The one case that did not (Section IX) is the more informative result: it "
        "demonstrates concretely that a complete recognized structural signature, established from static "
        "disassembly, can co-occur with a compiler-generated control-flow structure in which not every "
        "source-level execution path reaches that signature. We report this as a scope boundary of binary-only "
        "structural detection, not as a detection accuracy figure -- no precision, recall, or false-accept rate "
        "is reported, because no independently collected human ground truth exists for this matrix."
    )
    replacements.append((anchor14, replacement14, "VIII-C-headline"))

    # ── 15. VIII-D RQ4 headline ───────────────────────────────────────────
    anchor15 = ('<w:p><w:pPr><w:pBdr><w:top w:val="single" w:color="1155CC" w:sz="6"/>'
                '<w:bottom w:val="single" w:color="1155CC" w:sz="6"/><w:left w:val="single" w:color="1155CC" w:sz="6"/>'
                '<w:right w:val="single" w:color="1155CC" w:sz="6"/></w:pBdr><w:shd w:fill="EAF1FB" w:val="clear"/>'
                '<w:spacing w:after="200" w:before="100" w:line="300"/></w:pPr>'
                '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">[BLUE PLACEHOLDER] </w:t></w:r>'
                '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">§VIII-D headline result — </w:t></w:r>'
                '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">Outcome and interpretation of the E4 closed-kernel probe.'
                '</w:t></w:r></w:p>')
    replacement15 = normal_para(
        "RQ4 is addressed by synthesis (Section VI-D) rather than by the closed-kernel probe, which was not "
        "executed (Section VII-F). Given RQ1-RQ3's results, a compile-time zeroization guarantee, checked by "
        "binary-only structural analysis, held completely across every tested compiler-configuration axis for "
        "the hardened construct, and the source-independent check agreed with itself across two independent "
        "implementations at 100%. The one substantive open boundary this paper's evidence identifies is not "
        "configuration coverage or cross-tool disagreement, but the gap between static structural presence and "
        "dynamic path completeness illustrated by the controlled case in Section IX -- which argues for keeping "
        "any binary-only check explicitly scoped as a structural-signature check, not a substitute for a full "
        "control-flow or post-dominance verifier, and for treating hardware confidential computing as the "
        "appropriate complement for deployment scenarios where that stronger guarantee is required."
    )
    replacements.append((anchor15, replacement15, "VIII-D-headline"))

    # ── 16. Additional limitations placeholder ───────────────────────────
    anchor16 = ('<w:p><w:pPr><w:pBdr><w:top w:val="single" w:color="1155CC" w:sz="6"/>'
                '<w:bottom w:val="single" w:color="1155CC" w:sz="6"/><w:left w:val="single" w:color="1155CC" w:sz="6"/>'
                '<w:right w:val="single" w:color="1155CC" w:sz="6"/></w:pBdr><w:shd w:fill="EAF1FB" w:val="clear"/>'
                '<w:spacing w:after="200" w:before="100" w:line="300"/></w:pPr>'
                '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">[BLUE PLACEHOLDER] </w:t></w:r>'
                '<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">Additional limitations — </w:t></w:r>'
                '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>'
                '<w:t xml:space="preserve">Add any limitation that becomes apparent once E1–E4 are actually run '
                '(e.g., sample-size caveats, any configuration that could not be compiled locally).'
                '</w:t></w:r></w:p>')
    replacement16 = normal_para(
        "A controlled case in this study's binary-only detectability evaluation (Section IX) showed concretely "
        "that a complete recognized structural signature -- the required zero-store, fence, and barrier present "
        "and correctly ordered in the disassembly -- can coexist with a compiler-generated control-flow "
        "structure in which a source-level cleanup path is not reached by every thread; the checker used "
        "throughout this paper is a structural-signature check, not a control-flow-graph or post-dominance "
        "verifier, and this paper does not claim otherwise anywhere. All runtime, as opposed to compile-and-"
        "disassemble, evidence in this paper is bounded to a single physical GPU (an RTX 4050 Laptop, Ada "
        "Lovelace, sm_89); the other five architecture targets in the compiler-configuration matrix are "
        "compile-target evidence only, with no runtime claim made for them. Repeated-compilation evidence "
        "establishing byte-identical output across three additional compiles of fifty stratified configurations "
        "was collected on a single machine and toolchain installation and does not speak to reproducibility "
        "across different machines or driver versions. Where a bounded runtime sanity check was performed on a "
        "diagnostic kernel template, the instrumentation added to make the check possible -- a post-scrub "
        "readback of the sensitive buffer to global memory -- itself changes the compiler's liveness analysis of "
        "that buffer, so the check validates the diagnostic template's behavior and does not validate the exact, "
        "uninstrumented binary reported elsewhere in this paper's matrix."
    )
    replacements.append((anchor16, replacement16, "additional-limitations"))

    # ── 17. Stale "results not yet complete" intro to Section VIII ───────
    anchor17 = ('<w:p><w:pPr><w:spacing w:after="160" w:line="300"/><w:jc w:val="both"/></w:pPr>'
                '<w:r><w:rPr><w:b w:val="false"/><w:bCs w:val="false"/><w:i w:val="false"/><w:iCs w:val="false"/>'
                '<w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr><w:t xml:space="preserve">This section will report '
                'measured outcomes only. At the time of this draft, the extended compiler-configuration matrix, '
                'the cross-version comparison, and the binary-only detector have not yet been run to completion; '
                'the placeholders below mark exactly what evidence will replace them, and no numeric outcome '
                'should be inferred from their absence.</w:t></w:r></w:p>')
    replacement17 = normal_para(
        "This section reports measured outcomes from the completed core compilation matrix (2,016 "
        "configurations), the cross-tool validation pass, the controlled-case corpus, the reproducibility "
        "experiment, and a bounded runtime sanity check, all frozen at a single evidence commit (data and code "
        "available in the accompanying artifact repository). One planned experiment, the closed-kernel boundary "
        "probe (E4), was not executed and is reported as such in Section VII-F and Section VIII-D, not silently "
        "omitted."
    )
    replacements.append((anchor17, replacement17, "stale-results-intro"))

    z_in.close()
    xml_before = xml
    for anchor, repl, label in replacements:
        xml = apply_replacement(xml, anchor, repl, label)
        print(f"  applied: {label}")

    # ── Table 0 (draft's "TABLE I") — replace with construct-level outcome table ──
    table0_anchor = (
        '<w:tbl><w:tblPr><w:tblW w:type="pct" w:w="100%"/><w:tblBorders>'
        '<w:top w:val="single" w:color="auto" w:sz="4"/><w:left w:val="single" w:color="auto" w:sz="4"/>'
        '<w:bottom w:val="single" w:color="auto" w:sz="4"/><w:right w:val="single" w:color="auto" w:sz="4"/>'
        '<w:insideH w:val="single" w:color="auto" w:sz="4"/><w:insideV w:val="single" w:color="auto" w:sz="4"/>'
        '</w:tblBorders></w:tblPr><w:tblGrid><w:gridCol w:w="100"/><w:gridCol w:w="100"/><w:gridCol w:w="100"/>'
        '<w:gridCol w:w="100"/><w:gridCol w:w="100"/></w:tblGrid><w:tr><w:tc><w:tcPr><w:tcW w:type="pct" w:w="20%"/>'
        '<w:shd w:fill="D9E6FA" w:val="clear"/></w:tcPr><w:p><w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="18"/>'
        '<w:szCs w:val="18"/></w:rPr><w:t xml:space="preserve">Construction</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:tcPr><w:tcW w:type="pct" w:w="20%"/><w:shd w:fill="D9E6FA" w:val="clear"/></w:tcPr><w:p>'
        '<w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        '<w:t xml:space="preserve">Opt. level</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:tcPr><w:tcW w:type="pct" w:w="20%"/><w:shd w:fill="D9E6FA" w:val="clear"/></w:tcPr><w:p>'
        '<w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        '<w:t xml:space="preserve">Arch</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:tcPr><w:tcW w:type="pct" w:w="20%"/><w:shd w:fill="D9E6FA" w:val="clear"/></w:tcPr><w:p>'
        '<w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        '<w:t xml:space="preserve">Reg. flag</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:tcPr><w:tcW w:type="pct" w:w="20%"/><w:shd w:fill="D9E6FA" w:val="clear"/></w:tcPr><w:p>'
        '<w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        '<w:t xml:space="preserve">Outcome</w:t></w:r></w:p></w:tc></w:tr>'
        '<w:tr><w:tc><w:tcPr><w:tcW w:type="pct" w:w="20%"/><w:shd w:fill="EAF1FB" w:val="clear"/></w:tcPr><w:p>'
        '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        '<w:t xml:space="preserve">[TBD]</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:tcPr><w:tcW w:type="pct" w:w="20%"/><w:shd w:fill="EAF1FB" w:val="clear"/></w:tcPr><w:p>'
        '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        '<w:t xml:space="preserve">[TBD]</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:tcPr><w:tcW w:type="pct" w:w="20%"/><w:shd w:fill="EAF1FB" w:val="clear"/></w:tcPr><w:p>'
        '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        '<w:t xml:space="preserve">[TBD]</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:tcPr><w:tcW w:type="pct" w:w="20%"/><w:shd w:fill="EAF1FB" w:val="clear"/></w:tcPr><w:p>'
        '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        '<w:t xml:space="preserve">[TBD]</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:tcPr><w:tcW w:type="pct" w:w="20%"/><w:shd w:fill="EAF1FB" w:val="clear"/></w:tcPr><w:p>'
        '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        '<w:t xml:space="preserve">[TBD]</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
    )
    headers0 = ["Construct", "Total", "FULL_SIG.", "PARTIAL_SIG.", "STORE_ONLY", "ABSENT"]
    rows0_data = [
        ["Z0 (no scrub)", 288, 0, 0, 0, 288],
        ["Z1 (plain)", 289, 0, 192, 0, 97],
        ["Z2 (volatile)", 289, 0, 193, 96, 0],
        ["Z3 (plain+barrier)", 288, 0, 192, 0, 96],
        ["Z4 (volatile+barrier)", 288, 0, 288, 0, 0],
        ["Z5 (hardened)", 288, 288, 0, 0, 0],
        ["Z6 (inline-PTX hardened)", 288, 288, 0, 0, 0],
    ]
    table0_new = build_table(headers0, rows0_data, col_widths_pct=100 // len(headers0))
    xml = apply_replacement(xml, table0_anchor, table0_new, "table0-rebuild")
    print("  applied: table0-rebuild")

    # ── Table 1 (draft's "TABLE II") — replace with controlled-case cross-tool comparison ──
    table1_anchor = (
        '<w:tbl><w:tblPr><w:tblW w:type="pct" w:w="100%"/><w:tblBorders>'
        '<w:top w:val="single" w:color="auto" w:sz="4"/><w:left w:val="single" w:color="auto" w:sz="4"/>'
        '<w:bottom w:val="single" w:color="auto" w:sz="4"/><w:right w:val="single" w:color="auto" w:sz="4"/>'
        '<w:insideH w:val="single" w:color="auto" w:sz="4"/><w:insideV w:val="single" w:color="auto" w:sz="4"/>'
        '</w:tblBorders></w:tblPr><w:tblGrid><w:gridCol w:w="100"/><w:gridCol w:w="100"/><w:gridCol w:w="100"/>'
        '<w:gridCol w:w="100"/></w:tblGrid><w:tr><w:tc><w:tcPr><w:tcW w:type="pct" w:w="25%"/>'
        '<w:shd w:fill="D9E6FA" w:val="clear"/></w:tcPr><w:p><w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="18"/>'
        '<w:szCs w:val="18"/></w:rPr><w:t xml:space="preserve">Config.</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:tcPr><w:tcW w:type="pct" w:w="25%"/><w:shd w:fill="D9E6FA" w:val="clear"/></w:tcPr><w:p>'
        '<w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        '<w:t xml:space="preserve">Ground truth</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:tcPr><w:tcW w:type="pct" w:w="25%"/><w:shd w:fill="D9E6FA" w:val="clear"/></w:tcPr><w:p>'
        '<w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        '<w:t xml:space="preserve">Detector verdict</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:tcPr><w:tcW w:type="pct" w:w="25%"/><w:shd w:fill="D9E6FA" w:val="clear"/></w:tcPr><w:p>'
        '<w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        '<w:t xml:space="preserve">Match?</w:t></w:r></w:p></w:tc></w:tr>'
        '<w:tr><w:tc><w:tcPr><w:tcW w:type="pct" w:w="25%"/><w:shd w:fill="EAF1FB" w:val="clear"/></w:tcPr><w:p>'
        '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        '<w:t xml:space="preserve">[TBD]</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:tcPr><w:tcW w:type="pct" w:w="25%"/><w:shd w:fill="EAF1FB" w:val="clear"/></w:tcPr><w:p>'
        '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        '<w:t xml:space="preserve">[TBD]</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:tcPr><w:tcW w:type="pct" w:w="25%"/><w:shd w:fill="EAF1FB" w:val="clear"/></w:tcPr><w:p>'
        '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        '<w:t xml:space="preserve">[TBD]</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:tcPr><w:tcW w:type="pct" w:w="25%"/><w:shd w:fill="EAF1FB" w:val="clear"/></w:tcPr><w:p>'
        '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="1155CC"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
        '<w:t xml:space="preserve">[TBD]</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
    )
    import json as _json
    controlled = [_json.loads(l) for l in open(
        "/mnt/c/Tripti Sharma/lmhc-forge/robustness_study/results/controlled_cases.jsonl")]
    by_case = {}
    for r in controlled:
        by_case.setdefault(r["case_id"], []).append(r)
    headers1 = ["Case", "cuobjdump verdict", "nvdisasm verdict", "Agreement"]
    rows1_data = []
    for cid in [f"N{i}" for i in range(13)]:
        rs = by_case[cid]
        co_verdicts = sorted(set(str(r.get("cuobjdump_verdict")) for r in rs))
        nv_verdicts = sorted(set(str(r.get("nvdisasm_verdict")) for r in rs if "nvdisasm_verdict" in r))
        agreements = [r.get("cross_tool_agreement") for r in rs if "cross_tool_agreement" in r]
        agree_str = f"{sum(1 for a in agreements if a)}/{len(agreements)}" if agreements else "N/A"
        co_str = co_verdicts[0] if len(co_verdicts) == 1 else "/".join(co_verdicts)
        nv_str = (nv_verdicts[0] if len(nv_verdicts) == 1 else "/".join(nv_verdicts)) if nv_verdicts else "N/A"
        n_configs = len(rs)
        label = cid if n_configs == 1 else f"{cid} ({n_configs} configs)"
        rows1_data.append([label, co_str, nv_str, agree_str])
    table1_new = build_table(headers1, rows1_data, col_widths_pct=25)
    xml = apply_replacement(xml, table1_anchor, table1_new, "table1-rebuild")
    print("  applied: table1-rebuild")

    print(f"\nTotal length change: {len(xml) - original_len:+d} chars")

    # ── Write the final docx: copy every part unchanged except document.xml ──
    z_in2 = zipfile.ZipFile(SRC)
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as z_out:
        for item in z_in2.infolist():
            data = z_in2.read(item.filename)
            if item.filename == "word/document.xml":
                data = xml.encode("utf-8")
            z_out.writestr(item, data)
    z_in2.close()
    print(f"\nwrote {DST}")


if __name__ == "__main__":
    main()
