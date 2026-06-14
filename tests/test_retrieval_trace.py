"""
Unit tests for the retrieval trace (format_retrieval_trace).

The trace is a collapsible "Retrieval trace" block streamed before the answer,
showing which databases were searched, how many hits each returned, and the top
hit. These tests cover the formatting and routing-reflection logic over a
populated RetrievalResult - no DB access (pure formatting).

Run with: python -m unittest tests.test_retrieval_trace
"""

import os
import unittest
from unittest.mock import patch

from app.models import VariantEvidence
from app.retrieval_genereviews import GeneReviewsChunk
from app.router import format_retrieval_trace, _count, RetrievalResult


def _variant(vid=31632, gene="KMT2D", sig="Pathogenic"):
    return VariantEvidence(
        variation_id=vid, gene_symbol=gene, variant_name="c.x",
        clinical_significance=sig, review_status="reviewed",
        similarity=0.9, document="d",
    )


def _chunk(nbk="NBK62111", cond="Kabuki Syndrome", genes=None):
    return GeneReviewsChunk(
        id=1, nbk_id=nbk, condition_name=cond, gene_symbols=genes or ["KMT2D"],
        section_title="s", section_path="p", section_type="clinical",
        chunk_text="t", chunk_index=0, token_estimate=10, similarity=0.8,
    )


class TestCount(unittest.TestCase):
    def test_singular_plural(self):
        self.assertEqual(_count(1, "variant"), "1 variant")
        self.assertEqual(_count(3, "variant"), "3 variants")

    def test_y_plural(self):
        self.assertEqual(_count(1, "entry"), "1 entry")
        self.assertEqual(_count(2, "entry"), "2 entries")


class TestTraceDisabled(unittest.TestCase):
    @patch.dict(os.environ, {"RAG_TRACE": "0"})
    def test_disabled_returns_empty(self):
        r = RetrievalResult(query_type="variant", genereviews_chunks=[_chunk()])
        self.assertEqual(format_retrieval_trace(r, "KMT2D gene"), "")

    @patch.dict(os.environ, {"RAG_TRACE": "off"})
    def test_off_value_disables(self):
        r = RetrievalResult(query_type="variant")
        self.assertEqual(format_retrieval_trace(r, "KMT2D gene"), "")


class TestTraceVariantQuery(unittest.TestCase):
    @patch.dict(os.environ, {"RAG_TRACE": "1"})
    def test_full_variant_trace(self):
        r = RetrievalResult(
            query_type="variant",
            clinvar_evidence=[_variant()] * 5,
            genereviews_chunks=[_chunk()] * 3,
            pubmed_abstracts=[{"pmid": "12345678", "title": "t"}] * 2,
        )
        out = format_retrieval_trace(r, "KMT2D gene")
        # Collapsible wrapper + summary.
        self.assertIn("<details>", out)
        self.assertIn("<summary>", out)
        self.assertIn("Retrieval trace", out)
        # Classification header carries the gene.
        self.assertIn("variant lookup (KMT2D)", out)
        # Per-DB lines with counts and top hits.
        self.assertIn("ClinVar — 5 variants", out)
        self.assertIn("Variation 31632", out)
        self.assertIn("GeneReviews — 3 chapters", out)
        self.assertIn("Kabuki Syndrome [NBK62111]", out)
        self.assertIn("PubMed — 2 abstracts", out)
        self.assertIn("PMID 12345678", out)
        # No renal terms -> NephroGenetics skipped.
        self.assertIn("NephroGenetics — skipped (no renal terms)", out)


class TestTracePhenotypeQuery(unittest.TestCase):
    @patch.dict(os.environ, {"RAG_TRACE": "1"})
    def test_phenotype_skips_clinvar_and_pubmed(self):
        # No gene symbol -> ClinVar and PubMed are not searched.
        r = RetrievalResult(
            query_type="phenotype",
            genereviews_chunks=[_chunk(nbk="NBK1116", cond="Some Condition", genes=[])],
        )
        out = format_retrieval_trace(r, "what causes recurrent miscarriage")
        self.assertIn("phenotype search", out)
        self.assertIn("ClinVar — skipped (phenotype-only query)", out)
        self.assertIn("PubMed — skipped (no gene symbol)", out)
        self.assertIn("GeneReviews — 1 chapter", out)  # singular


class TestTraceRenalQuery(unittest.TestCase):
    @patch.dict(os.environ, {"RAG_TRACE": "1"})
    def test_renal_terms_search_nephrogenetics(self):
        r = RetrievalResult(
            query_type="phenotype",
            nephro_results=[{"gene": "PKD1", "title": "Polycystic kidney disease"}],
        )
        out = format_retrieval_trace(r, "polycystic kidney disease inheritance")
        self.assertIn("NephroGenetics — 1 entry", out)
        self.assertIn("PKD1 - Polycystic kidney disease", out)


class TestTraceNonClinical(unittest.TestCase):
    @patch.dict(os.environ, {"RAG_TRACE": "1"})
    def test_non_clinical_no_db_lines(self):
        r = RetrievalResult(query_type="non-clinical")
        out = format_retrieval_trace(r, "hello there")
        self.assertIn("non-clinical", out)
        self.assertIn("databases not searched", out)
        self.assertNotIn("ClinVar", out)


class TestOnStepCallback(unittest.TestCase):
    """route_and_retrieve(on_step=...) reports one line per DB, in order, with
    text identical to the all-at-once trace's lines."""

    @patch.dict(os.environ, {"RAG_TRACE": "1"})
    @patch("app.router.search_pubmed")
    @patch("app.router.retrieve_nephrogenetics")
    @patch("app.router.retrieve_genereviews")
    @patch("app.router.retrieve_variants_hybrid")
    @patch("app.router.retrieve_variants_exact")
    def test_variant_query_steps_in_order(
        self, m_exact, m_hybrid, m_grx, m_nephro, m_pubmed
    ):
        from app.router import route_and_retrieve

        m_exact.return_value = [_variant()] * 5
        m_grx.return_value = [_chunk()] * 3
        m_nephro.return_value = []
        m_pubmed.return_value = [{"pmid": "12345678", "title": "t"}] * 2

        steps: list[str] = []
        route_and_retrieve("BRCA1 gene", k=5, on_step=steps.append)

        # Header first, then one line per DB in pipeline order.
        self.assertEqual(len(steps), 5)
        self.assertIn("variant lookup (BRCA1)", steps[0])
        self.assertIn("ClinVar — 5 variants", steps[1])
        self.assertIn("GeneReviews — 3 chapters", steps[2])
        self.assertIn("NephroGenetics — skipped (no renal terms)", steps[3])
        self.assertIn("PubMed — 2 abstracts", steps[4])

    @patch.dict(os.environ, {"RAG_TRACE": "1"})
    @patch("app.router.search_pubmed")
    @patch("app.router.retrieve_nephrogenetics")
    @patch("app.router.retrieve_genereviews")
    def test_no_callback_is_noop(self, m_grx, m_nephro, m_pubmed):
        # on_step=None (default) must not raise and must still return a result.
        from app.router import route_and_retrieve

        m_grx.return_value = []
        m_nephro.return_value = []
        m_pubmed.return_value = []
        result = route_and_retrieve("what causes Kabuki syndrome", k=5)
        self.assertEqual(result.query_type, "phenotype")

    @patch.dict(os.environ, {"RAG_TRACE": "1"})
    def test_non_clinical_emits_header_and_skip(self):
        from app.router import route_and_retrieve

        steps: list[str] = []
        route_and_retrieve("hello there", k=5, on_step=steps.append)
        self.assertEqual(len(steps), 2)
        self.assertIn("non-clinical", steps[0])
        self.assertIn("databases not searched", steps[1])


if __name__ == "__main__":
    unittest.main()
