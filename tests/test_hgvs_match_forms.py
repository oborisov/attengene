"""
Tests for HGVS del/dup base-suffix matching (the CFTR F508del bug).

ClinVar stores deletions/duplications inconsistently - some with trailing
bases (c.730_731delAG), some without (c.1521_1523del, the CFTR F508del form).
A clinician typing the base-bearing form must still match the base-less record.
hgvs_match_forms returns both forms; retrieve_variants_exact ORs them.

The behavior on the real 326K-variant corpus is verified separately against the
live DB (these are query-shape / form-generation unit tests, mocked SQL).

Run with: python -m unittest tests.test_hgvs_match_forms
"""

import unittest
from unittest.mock import MagicMock, patch

from app.hgvs import hgvs_match_forms, ParsedVariant


class TestMatchForms(unittest.TestCase):
    def test_simple_del_with_bases_adds_baseless(self):
        self.assertEqual(
            hgvs_match_forms("c.1521_1523delCTT"),
            ["c.1521_1523delCTT", "c.1521_1523del"],
        )

    def test_simple_dup_with_bases_adds_baseless(self):
        self.assertEqual(
            hgvs_match_forms("c.5266dupC"), ["c.5266dupC", "c.5266dup"]
        )

    def test_substitution_unchanged(self):
        self.assertEqual(hgvs_match_forms("c.526G>A"), ["c.526G>A"])

    def test_already_baseless_unchanged(self):
        self.assertEqual(hgvs_match_forms("c.1521_1523del"), ["c.1521_1523del"])

    def test_delins_not_reduced(self):
        # Compound delins: stripping a base group would be malformed; keep one form.
        self.assertEqual(
            hgvs_match_forms("c.581_582delCCinsAA"), ["c.581_582delCCinsAA"]
        )


class TestExactLookupOrsForms(unittest.TestCase):
    """retrieve_variants_exact ORs the coding-token forms in its WHERE clause."""

    @patch("app.retrieval.release_connection")
    @patch("app.retrieval.get_connection")
    def test_del_with_bases_ors_both_forms(self, m_get_conn, _m_rel):
        from app.retrieval import retrieve_variants_exact

        cur = MagicMock()
        cur.fetchall.return_value = []
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        m_get_conn.return_value = conn

        parsed = ParsedVariant(gene="CFTR", c_hgvs="c.1521_1523delCTT")
        retrieve_variants_exact(parsed, k=5)

        sql, params = cur.execute.call_args[0]
        # Both surface forms present as ILIKE params.
        self.assertIn("%c.1521_1523delCTT%", params)
        self.assertIn("%c.1521_1523del%", params)
        # ORed (not ANDed) so either match suffices.
        self.assertIn(" OR ", sql)
        # Gene still filtered.
        self.assertIn("CFTR", params)

    @patch("app.retrieval.release_connection")
    @patch("app.retrieval.get_connection")
    def test_substitution_single_form(self, m_get_conn, _m_rel):
        from app.retrieval import retrieve_variants_exact

        cur = MagicMock()
        cur.fetchall.return_value = []
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        m_get_conn.return_value = conn

        parsed = ParsedVariant(gene="ALPL", c_hgvs="c.526G>A")
        retrieve_variants_exact(parsed, k=5)

        _sql, params = cur.execute.call_args[0]
        self.assertIn("%c.526G>A%", params)


if __name__ == "__main__":
    unittest.main()
