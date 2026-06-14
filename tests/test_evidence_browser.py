"""
Tests for the evidence browser (format_evidence_browser).

After retrieval, a nested-collapsible "Retrieved evidence" block is emitted so
the user can read the retrieved content during the (longer) generation wait.
It is built whole (not token-streamed) so the nested <details> render cleanly,
and every section is collapsed by default.

Run with: python -m unittest tests.test_evidence_browser
"""

import os
import unittest
from unittest.mock import patch

from app.models import VariantEvidence
from app.retrieval_genereviews import GeneReviewsChunk
from app.router import format_evidence_browser, _snippet, RetrievalResult


def _chunk(text="Some chapter text about the condition.", nbk="NBK1250",
           cond="Cystic Fibrosis", section="Diagnosis"):
    return GeneReviewsChunk(
        id=1, nbk_id=nbk, condition_name=cond, gene_symbols=["CFTR"],
        section_title=section, section_path="p", section_type="clinical",
        chunk_text=text, chunk_index=0, token_estimate=10, similarity=0.8,
    )


def _variant():
    return VariantEvidence(
        variation_id=7105, gene_symbol="CFTR",
        variant_name="c.1521_1523del (p.Phe508del)",
        clinical_significance="Pathogenic",
        review_status="reviewed by expert panel",
        condition_names="Cystic fibrosis", similarity=1.0, document="d",
    )


class TestSnippet(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(_snippet("hello world"), "hello world")

    def test_collapses_whitespace(self):
        self.assertEqual(_snippet("a\n\n  b   c"), "a b c")

    def test_truncates_on_word_boundary_with_ellipsis(self):
        out = _snippet("word " * 200, limit=20)
        self.assertTrue(out.endswith("…"))
        self.assertLessEqual(len(out), 21)
        self.assertNotIn("  ", out)

    def test_empty(self):
        self.assertEqual(_snippet(None), "")


class TestEvidenceBrowser(unittest.TestCase):
    @patch.dict(os.environ, {"RAG_TRACE": "0"})
    def test_disabled_returns_empty(self):
        r = RetrievalResult(query_type="variant", clinvar_evidence=[_variant()])
        self.assertEqual(format_evidence_browser(r, "CFTR"), "")

    @patch.dict(os.environ, {"RAG_TRACE": "1"})
    def test_nothing_retrieved_returns_empty(self):
        r = RetrievalResult(query_type="phenotype")
        self.assertEqual(format_evidence_browser(r, "anything"), "")

    @patch.dict(os.environ, {"RAG_TRACE": "1"})
    def test_nested_collapsibles_and_content(self):
        r = RetrievalResult(
            query_type="variant",
            clinvar_evidence=[_variant()],
            genereviews_chunks=[_chunk()],
            pubmed_abstracts=[{"pmid": "34934215", "title": "CFTR review",
                               "abstract": "An abstract.",
                               "url": "https://pubmed.ncbi.nlm.nih.gov/34934215/"}],
        )
        out = format_evidence_browser(r, "CFTR c.1521_1523delCTT")

        # Outer browse wrapper + per-DB groups + per-item details.
        self.assertIn("📚 Retrieved evidence", out)
        self.assertIn("<details>", out)
        # Nesting: more than one <details> opener (outer + groups + items).
        self.assertGreaterEqual(out.count("<details>"), 5)
        self.assertEqual(out.count("<details>"), out.count("</details>"))
        # Content surfaced.
        self.assertIn("ClinVar 7105", out)
        self.assertIn("Cystic fibrosis", out)          # condition
        self.assertIn("Cystic Fibrosis [NBK1250]", out)  # GeneReviews item
        self.assertIn("PMID 34934215", out)
        # Links present.
        self.assertIn("clinvar/variation/7105", out)
        self.assertIn("books/NBK1250", out)
        self.assertIn("pubmed.ncbi.nlm.nih.gov/34934215", out)

    @patch.dict(os.environ, {"RAG_TRACE": "1"})
    def test_nephro_group(self):
        r = RetrievalResult(
            query_type="phenotype",
            nephro_results=[{"gene": "COL4A3", "title": "Alport syndrome",
                             "inheritance": "AD", "kidney": "Hematuria, CKD"}],
        )
        out = format_evidence_browser(r, "Alport kidney")
        self.assertIn("NephroGenetics — 1 entry", out)
        self.assertIn("COL4A3 — Alport syndrome", out)
        self.assertIn("Hematuria", out)


if __name__ == "__main__":
    unittest.main()
