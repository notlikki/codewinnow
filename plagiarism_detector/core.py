"""Token and AST-normalised Rabin--Karp winnowing for Python source."""

from __future__ import annotations

import ast
import io
import keyword
import tokenize
from dataclasses import dataclass
from typing import Iterable, Literal

MODULUS = 2_305_843_009_213_693_951  # large prime below 2**61
FINGERPRINT_POLICY = "adaptive-k-v1"
BASE = 257


@dataclass(frozen=True)
class FingerprintConfig:
    """Fingerprint parameters. Window is measured in consecutive k-gram hashes."""

    k: int = 5
    window: int = 4
    mode: Literal["token", "ast"] = "token"

    def __post_init__(self) -> None:
        if self.k < 1 or self.window < 1:
            raise ValueError("k and window must be positive")


@dataclass(frozen=True)
class Comparison:
    token_similarity: float
    ast_similarity: float
    token_fingerprints: tuple[int, int]
    ast_fingerprints: tuple[int, int]
    k: int
    window: int
    token_counts: tuple[int, int]


def token_stream(source: str) -> list[str]:
    """Return lexical tokens while discarding comments, layout, and encoding."""
    ignored = {tokenize.ENCODING, tokenize.COMMENT, tokenize.NL,
               tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    result: list[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in ignored:
            continue
        # Keep keywords/operators distinct; names/literals remain meaningful here.
        result.append(f"{tokenize.tok_name[tok.type]}:{tok.string}")
    return result


class _NormalisingVisitor(ast.NodeVisitor):
    """Preorder structural stream with names/literals canonicalised by role."""

    def __init__(self) -> None:
        self.out: list[str] = []

    def visit(self, node: ast.AST) -> None:
        self.out.append(type(node).__name__)
        super().visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.out.append("NAME")
        self.out.append(type(node.ctx).__name__)

    def visit_arg(self, node: ast.arg) -> None:
        self.out.append("ARG")

    def visit_Constant(self, node: ast.Constant) -> None:
        # Literal class retains useful structure but suppresses superficial edits.
        self.out.append("CONST:" + type(node.value).__name__)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.out.append("ATTRIBUTE")
        self.visit(node.value)


def ast_stream(source: str) -> list[str]:
    """Return a syntax-aware, identifier- and literal-normalised token stream."""
    tree = ast.parse(source)
    visitor = _NormalisingVisitor()
    visitor.visit(tree)
    return visitor.out


def _symbol(token: str) -> int:
    """Deterministic byte hash; unlike hash(), it is stable between processes."""
    value = 0
    for byte in token.encode("utf-8"):
        value = (value * BASE + byte + 1) % MODULUS
    return value


def rolling_hashes(tokens: Iterable[str], k: int) -> list[int]:
    """Compute exact Rabin--Karp polynomial hashes for every token k-gram."""
    symbols = [_symbol(t) for t in tokens]
    if len(symbols) < k:
        return []
    high = pow(BASE, k - 1, MODULUS)
    current = 0
    for symbol in symbols[:k]:
        current = (current * BASE + symbol) % MODULUS
    hashes = [current]
    for old, new in zip(symbols, symbols[k:]):
        current = ((current - old * high) * BASE + new) % MODULUS
        hashes.append(current)
    return hashes


def winnow(hashes: list[int], window: int) -> list[tuple[int, int]]:
    """Select rightmost minima, returning unique `(hash, kgram_position)` values."""
    if not hashes:
        return []
    width = min(window, len(hashes))
    selected: list[tuple[int, int]] = []
    previous_position = -1
    for start in range(len(hashes) - width + 1):
        segment = hashes[start:start + width]
        minimum = min(segment)
        # Reverse search implements the rightmost-minimum convention.
        position = start + width - 1 - segment[::-1].index(minimum)
        if position != previous_position:
            selected.append((hashes[position], position))
            previous_position = position
    return selected


def fingerprints(source: str, config: FingerprintConfig = FingerprintConfig()) -> list[tuple[int, int]]:
    stream = token_stream(source) if config.mode == "token" else ast_stream(source)
    return winnow(rolling_hashes(stream, config.k), config.window)


def jaccard(left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]) -> float:
    """Jaccard similarity over selected hash values (positions are local evidence)."""
    a = {value for value, _ in left}
    b = {value for value, _ in right}
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def select_k(left_tokens: int, right_tokens: int) -> int:
    """Conservative length heuristic, shared by both files; not an optimized k.

    Use the shorter lexical stream to avoid suppressing matches in shorter files.
    Ignored comments/layout cannot affect this choice.
    """
    n = min(left_tokens, right_tokens)
    if n < 0:
        raise ValueError("token counts must be nonnegative")
    target = 3 if n < 50 else 4 if n < 150 else 5 if n < 400 else 7 if n < 1200 else 9
    return max(1, min(target, n))


def compare_sources(left: str, right: str, k: int | None = None, window: int = 4) -> Comparison:
    """Automatically choose a shared k, or honor an explicitly supplied k."""
    left_tokens, right_tokens = token_stream(left), token_stream(right)
    counts = (len(left_tokens), len(right_tokens))
    chosen_k = select_k(*counts) if k is None else k
    FingerprintConfig(chosen_k, window)  # Validate explicit overrides too.
    token_left = winnow(rolling_hashes(left_tokens, chosen_k), window)
    token_right = winnow(rolling_hashes(right_tokens, chosen_k), window)
    ast_left = fingerprints(left, FingerprintConfig(chosen_k, window, "ast")) if left_tokens else []
    ast_right = fingerprints(right, FingerprintConfig(chosen_k, window, "ast")) if right_tokens else []
    return Comparison(jaccard(token_left, token_right), jaccard(ast_left, ast_right),
                      (len(token_left), len(token_right)), (len(ast_left), len(ast_right)),
                      chosen_k, window, counts)
