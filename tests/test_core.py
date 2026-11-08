import unittest

from plagiarism_detector.core import (
    FingerprintConfig,
    ast_stream,
    compare_sources,
    fingerprints,
    rolling_hashes,
    winnow,
)


class FingerprintTests(unittest.TestCase):
    def test_rolling_hash_count(self):
        self.assertEqual(len(rolling_hashes(["a", "b", "c", "d"], 3)), 2)

    def test_rightmost_minimum(self):
        self.assertEqual(winnow([5, 1, 1, 7], 3), [(1, 2)])

    def test_ast_normalisation_resists_renaming(self):
        left = "def f(x):\n return x + 1\n"
        right = "def g(value):\n return value + 99\n"
        result = compare_sources(left, right, k=2, window=2)
        self.assertGreater(result.ast_similarity, result.token_similarity)

    def test_short_program_has_no_fingerprints(self):
        self.assertEqual(fingerprints("x = 1", FingerprintConfig(k=50)), [])

    def test_ast_stream_is_nonempty(self):
        self.assertIn("FunctionDef", ast_stream("def f(): pass"))


if __name__ == "__main__":
    unittest.main()
