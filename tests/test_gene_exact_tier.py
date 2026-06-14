"""
Tests for the gene-exact ClinVar tier (the bare-gene gap).

A query naming only a gene ("BRCA1", no HGVS) has an exact match in the `gene`
column, but the HGVS-exact tier (no variant token) and the hybrid tier (short
symbol trigram-matches full variant names below the lexical floor) both return
nothing. retrieve_variants_by_gene closes that gap with a direct gene filter,
ordered by ClinVar review status. The router invokes it between the exact and
hybrid tiers for a bare gene.

Behavior on the real corpus is verified separately against the live DB; these
are query-shape / routing-order unit tests with mocked SQL.

Run with: python -m unittest tests.test_gene_exact_tier
"""

import unittest
from unittest.mock import MagicMock, patch

from app.models import VariantEvidence


def _mock_conn(rows):
    cur = MagicMock()
    cur.fetchall.return_value = rows
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    return conn, cur


class TestRetrieveByGene(unittest.TestCase):
    @patch("app.retrieval.release_connection")
    @patch("app.retrieval.get_connection")
    def test_filters_by_gene_and_orders_by_review(self, m_get, _m_rel):
        from app.retrieval import retrieve_variants_by_gene

        conn, cur = _mock_conn([
            (17661, "BRCA1", "NM_007294.4(BRCA1):c.181T>G", "Pathogenic",
             "reviewed by expert panel", "Breast cancer", "doc"),
        ])
        m_get.return_value = conn

        out = retrieve_variants_by_gene("BRCA1", k=5)

        sql, params = cur.execute.call_args[0]
        self.assertIn("WHERE gene = %s", sql)
        self.assertIn("ORDER BY", sql)
        self.assertIn("CASE", sql)             # review-status ranking
        # Params must be in SQL textual placeholder order: gene (WHERE) first,
        # then the CASE status strings, then k (LIMIT). The gene-then-status
        # order is the bug-class that a wrong order (status-first) caused:
        # gene=%s would bind to a review-status string and match no rows.
        self.assertEqual(params[0], "BRCA1")   # WHERE gene = %s (first)
        self.assertEqual(params[-1], 5)        # LIMIT %s (last)
        # The 9 review-status strings sit between gene and k.
        self.assertIn("reviewed by expert panel", params[1:-1])
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], VariantEvidence)
        self.assertEqual(out[0].gene_symbol, "BRCA1")
        self.assertEqual(out[0].similarity, 1.0)

    def test_empty_gene_short_circuits(self):
        from app.retrieval import retrieve_variants_by_gene
        self.assertEqual(retrieve_variants_by_gene("", k=5), [])


class TestRouterTierOrder(unittest.TestCase):
    """The router uses the gene tier only for a bare gene, between exact and
    hybrid, and does not call hybrid when the gene tier returns hits."""

    @patch("app.router.search_pubmed", return_value=[])
    @patch("app.router.retrieve_genereviews", return_value=[])
    @patch("app.router.retrieve_variants_hybrid")
    @patch("app.router.retrieve_variants_by_gene")
    @patch("app.router.retrieve_variants_exact", return_value=[])
    def test_bare_gene_uses_gene_tier_not_hybrid(
        self, m_exact, m_gene, m_hybrid, _m_grx, _m_pm
    ):
        from app.router import route_and_retrieve

        m_gene.return_value = [
            VariantEvidence(
                variation_id=17661, gene_symbol="BRCA1",
                variant_name="n", clinical_significance="Pathogenic",
                review_status="reviewed by expert panel", similarity=1.0,
                document="d",
            )
        ]
        route_and_retrieve("BRCA1", k=5)

        m_gene.assert_called_once_with("BRCA1", k=5)
        m_hybrid.assert_not_called()  # gene tier satisfied the query

    @patch("app.router.search_pubmed", return_value=[])
    @patch("app.router.retrieve_genereviews", return_value=[])
    @patch("app.router.retrieve_variants_hybrid", return_value=[])
    @patch("app.router.retrieve_variants_by_gene", return_value=[])
    @patch("app.router.retrieve_variants_exact", return_value=[])
    def test_gene_tier_empty_falls_through_to_hybrid(
        self, _m_exact, m_gene, m_hybrid, _m_grx, _m_pm
    ):
        from app.router import route_and_retrieve

        route_and_retrieve("BRCA1", k=5)
        m_gene.assert_called_once()
        m_hybrid.assert_called_once()  # gene tier empty -> hybrid fallback

    @patch("app.router.search_pubmed", return_value=[])
    @patch("app.router.retrieve_genereviews", return_value=[])
    @patch("app.router.retrieve_variants_hybrid", return_value=[])
    @patch("app.router.retrieve_variants_by_gene")
    @patch("app.router.retrieve_variants_exact")
    def test_hgvs_query_skips_gene_tier(
        self, m_exact, m_gene, _m_hybrid, _m_grx, _m_pm
    ):
        from app.router import route_and_retrieve

        # An HGVS token present -> exact tier owns it; gene tier must not fire
        # (it's only for a bare gene with no variant token).
        m_exact.return_value = [
            VariantEvidence(
                variation_id=7105, gene_symbol="CFTR", variant_name="n",
                clinical_significance="Pathogenic", review_status="r",
                similarity=1.0, document="d",
            )
        ]
        route_and_retrieve("CFTR c.1521_1523delCTT", k=5)
        m_gene.assert_not_called()


if __name__ == "__main__":
    unittest.main()
