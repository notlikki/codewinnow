import unittest
from plagiarism_detector.core import compare_sources, select_k

class AdaptiveKTests(unittest.TestCase):
    def test_length_bands_and_tiny_inputs(self):
        for n,k in [(0,1),(1,1),(2,2),(49,3),(50,4),(149,4),(150,5),(399,5),(400,7),(1199,7),(1200,9)]:
            with self.subTest(n=n): self.assertEqual(select_k(n,n),k)
    def test_shorter_file_controls_both_orders(self):
        self.assertEqual(select_k(40,5000),3)
        self.assertEqual(select_k(5000,40),3)
    def test_formatting_does_not_change_k_or_score(self):
        a='x=1\n'*20
        b='# a comment\n'+'x = 1  # another comment\n\n'*20
        result=compare_sources(a,b)
        self.assertEqual(result.k,4)
        self.assertEqual(result.token_similarity,1)
        self.assertEqual(result.ast_similarity,1)
    def test_explicit_override_still_supported(self):
        result=compare_sources('x=1','x=1',k=5)
        self.assertEqual(result.k,5)
        self.assertEqual(result.token_fingerprints,(0,0))
        with self.assertRaises(ValueError): compare_sources('x=1','x=1',k=0)
    def test_empty_input_has_no_false_evidence(self):
        result=compare_sources('','# nothing')
        self.assertEqual(result.token_similarity,0)
        self.assertEqual(result.ast_similarity,0)
    def test_mixed_lengths_choose_one_shared_k(self):
        a='x=1\n'*20; b=a+'y=2\n'*200
        self.assertEqual(compare_sources(a,b).k,compare_sources(b,a).k)
        self.assertEqual(compare_sources(a,b).token_similarity,compare_sources(b,a).token_similarity)
