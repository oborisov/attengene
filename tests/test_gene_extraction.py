"""
Unit tests for _extract_gene_symbols (app.router).

Gene symbols are uppercase by HGNC convention, but clinicians type lowercase
("muc1 gene", "kmt2d gene"). The extractor takes bare uppercase tokens directly
and lowercase/mixed-case tokens only when a gene cue is present, so prose words
in phenotype queries don't get mistaken for genes (which would wrongly flip the
query to the variant route and fire ClinVar/PubMed on noise).

Run with: python -m unittest tests.test_gene_extraction
"""

import unittest

from app.router import _extract_gene_symbols as extract


class TestUppercaseTokens(unittest.TestCase):
    def test_bare_uppercase_gene(self):
        self.assertEqual(extract("BRCA1"), ["BRCA1"])

    def test_uppercase_in_sentence(self):
        self.assertEqual(extract("mutations of COL4A3"), ["COL4A3"])

    def test_stopwords_filtered(self):
        # GENE/DNA/VUS etc. are in _NOT_GENES.
        self.assertEqual(extract("what gene causes this"), [])


class TestCuedLowercaseTokens(unittest.TestCase):
    def test_gene_suffix_cue(self):
        self.assertEqual(extract("muc1 gene"), ["MUC1"])
        self.assertEqual(extract("kmt2d gene"), ["KMT2D"])

    def test_gene_prefix_cue(self):
        self.assertEqual(extract("gene kmt2d"), ["KMT2D"])

    def test_variant_in_cue(self):
        self.assertEqual(extract("variant in brca1"), ["BRCA1"])

    def test_mutations_of_cue(self):
        self.assertEqual(extract("mutations of col4a3"), ["COL4A3"])

    def test_normalized_to_uppercase(self):
        # Output is always canonical uppercase regardless of input case.
        self.assertEqual(extract("the Brca1 gene"), ["BRCA1"])


class TestProseDoesNotFalseMatch(unittest.TestCase):
    """Phenotype prose with no gene cue must yield no gene (stays phenotype route)."""

    def test_phenotype_question(self):
        self.assertEqual(extract("what causes Kabuki syndrome"), [])

    def test_renal_phenotype(self):
        self.assertEqual(extract("Alport syndrome kidney involvement"), [])

    def test_general_clinical_question(self):
        self.assertEqual(extract("how is cystic fibrosis diagnosed"), [])


class TestOrderAndDedup(unittest.TestCase):
    def test_dedup_uppercase_and_cued_same_gene(self):
        # "BRCA1 gene" matches both the bare-uppercase and the cued path;
        # the gene must appear once.
        self.assertEqual(extract("the BRCA1 gene"), ["BRCA1"])

    def test_order_preserved(self):
        out = extract("BRCA1 and the TP53 gene")
        self.assertEqual(out, ["BRCA1", "TP53"])


if __name__ == "__main__":
    unittest.main()
