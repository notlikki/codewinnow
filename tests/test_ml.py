import unittest
from plagiarism_detector.core import compare_sources
from plagiarism_detector.web import render
from plagiarism_detector.ml import detect, lexical

class MLTests(unittest.TestCase):
    def test_short_unrelated_is_not_perfect_match(self):
        self.assertEqual(compare_sources('x=1','pass').token_similarity,0)
    def test_template_keeps_source_placeholders(self):
        self.assertIn('print(&quot;{right}&quot;)',render('print("{right}")','pass').decode())
    def test_short_code_abstains(self):
        result=detect('x=1','pass')
        self.assertEqual(result['authorship_a']['verdict'],'Insufficient code')
        self.assertEqual(result['copy']['verdict'],'Insufficient code')
    def test_comments_and_docstrings_removed(self):
        self.assertEqual(lexical('"""license"""\nx=1 # comment'),lexical('x=99'))
    def test_invalid_code_rejected(self):
        with self.assertRaises(SyntaxError): detect('def :')

if __name__=='__main__': unittest.main()
