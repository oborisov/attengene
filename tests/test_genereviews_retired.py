"""
Regression tests for P0 #2 (ESHG 2026 poster feedback): retired GeneReviews
chapters must not surface as live clinical sources, and non-numeric/dead
Bookshelf IDs must not produce broken citation links.

Observed failure: `vcam1 gene` returned the RETIRED "VCAN-Related
Vitreoretinopathy" chapter (NBKwagner) as a cited source, with a dead URL.

These are unit-level guards on:
  - the parser's retired-marker detection (both dash variants seen live),
  - the URL builder's numeric-vs-shortname guard,
  - the retrieval query *shape* (default-excludes retired). Behavior on the
    real corpus is validated separately against the live index.

Run with: python -m unittest tests.test_genereviews_retired
"""

import unittest
from unittest.mock import MagicMock, patch

from app.genereviews import genereviews_url
from scripts.genereviews_parser import is_retired_title


class TestRetiredMarkerDetection(unittest.TestCase):
    def test_en_dash_marker(self):
        # The common form seen in the live index.
        self.assertTrue(
            is_retired_title(
                "VCAN-Related Vitreoretinopathy – RETIRED CHAPTER, "
                "FOR HISTORICAL REFERENCE ONLY"
            )
        )

    def test_box_drawing_dash_marker(self):
        # A handful of live titles use the box-drawing char "─" before RETIRED.
        self.assertTrue(
            is_retired_title(
                "Congenital Hepatic Fibrosis Overview ─ RETIRED CHAPTER, "
                "FOR HISTORICAL REFERENCE ONLY"
            )
        )

    def test_case_insensitive(self):
        self.assertTrue(is_retired_title("Foo retired chapter blah"))

    def test_historical_reference_alone(self):
        self.assertTrue(is_retired_title("Foo - For Historical Reference Only"))

    def test_live_title_not_flagged(self):
        self.assertFalse(is_retired_title("Lynch Syndrome"))
        self.assertFalse(is_retired_title("Cystic Fibrosis"))

    def test_empty_title(self):
        self.assertFalse(is_retired_title(""))
        self.assertFalse(is_retired_title(None))


class TestGeneReviewsUrlGuard(unittest.TestCase):
    def test_numeric_id_builds_books_url(self):
        self.assertEqual(
            genereviews_url("NBK1211"),
            "https://www.ncbi.nlm.nih.gov/books/NBK1211/",
        )

    def test_shortname_id_falls_back_to_landing(self):
        # The NBKwagner case from the demo: a shortname-derived id is not a
        # real Bookshelf ID, so it must not produce a /books/NBKwagner/ link.
        landing = "https://www.ncbi.nlm.nih.gov/books/NBK1116/"
        self.assertEqual(genereviews_url("NBKwagner"), landing)
        self.assertEqual(genereviews_url("NBKaic"), landing)
        self.assertEqual(genereviews_url("NBKdel2q37_2"), landing)

    def test_empty_id_falls_back_to_landing(self):
        self.assertEqual(
            genereviews_url(""),
            "https://www.ncbi.nlm.nih.gov/books/NBK1116/",
        )


class TestRetrievalExcludesRetired(unittest.TestCase):
    """Query-shape guard: the default retrieval path filters retired rows."""

    def _run_query(self, **kwargs):
        captured = {}

        cur = MagicMock()
        cur.__enter__ = lambda s: cur
        cur.__exit__ = lambda *a: False
        cur.fetchall.return_value = []

        def _execute(sql, params):
            captured["sql"] = sql
            captured["params"] = params

        cur.execute.side_effect = _execute

        conn = MagicMock()
        conn.cursor.return_value = cur

        with patch("app.retrieval_genereviews.encode", return_value=[0.0] * 1024), \
             patch("app.retrieval_genereviews.get_connection", return_value=conn), \
             patch("app.retrieval_genereviews.release_connection"):
            from app.retrieval_genereviews import retrieve_genereviews
            retrieve_genereviews("vcam1 gene", **kwargs)

        return captured["sql"]

    def test_default_excludes_retired(self):
        sql = self._run_query()
        self.assertIn("retired = false", sql)

    def test_include_retired_omits_filter(self):
        sql = self._run_query(include_retired=True)
        self.assertNotIn("retired = false", sql)


if __name__ == "__main__":
    unittest.main()
