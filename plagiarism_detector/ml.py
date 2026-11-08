"""Trainable classifiers. Code is parsed, never executed."""
from __future__ import annotations
import ast
import io
import json
import pickle
import tokenize
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from .core import FINGERPRINT_POLICY, compare_sources, select_k, token_stream

ROOT = Path(__file__).resolve().parent.parent

def lexical(source):
    """Remove comments, docstrings and literal contents to reduce provenance shortcuts."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if hasattr(node, 'body') and isinstance(node.body, list):
            node.body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))]
    source = ast.unparse(tree)
    out = []
    for t in tokenize.generate_tokens(io.StringIO(source).readline):
        if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        out.append('STRING' if t.type == tokenize.STRING else 'NUMBER' if t.type == tokenize.NUMBER else t.string)
    return ' '.join(out)

def pair_features(a, b):
    score = compare_sources(a, b)
    ta, tb = token_stream(a), token_stream(b)
    ca = Counter(type(n).__name__ for n in ast.walk(ast.parse(a)))
    cb = Counter(type(n).__name__ for n in ast.walk(ast.parse(b)))
    keys = ca.keys() | cb.keys()
    hist = sum(min(ca[k],cb[k]) for k in keys) / max(1,sum(max(ca[k],cb[k]) for k in keys))
    return [score.token_similarity, score.ast_similarity,
            min(len(ta),len(tb))/max(1,len(ta),len(tb)), hist,
            SequenceMatcher(None,ta,tb,autojunk=False).ratio()]

@lru_cache(maxsize=2)
def load_model(name):
    # Only load locally trained artifacts; never accept uploaded pickle files.
    path = ROOT / 'models' / (name + '.pkl')
    if not path.exists():
        raise ValueError('Models missing. Run scripts/train_models.py first.')
    with path.open('rb') as f:
        model = pickle.load(f)
    if name == 'copy' and getattr(model, 'fingerprint_policy_', None) != FINGERPRINT_POLICY:
        raise ValueError('Copy model uses outdated fingerprint settings. Retrain with scripts/train_models.py.')
    return model

def authorship(source):
    if len(source.encode()) > 100_000:
        raise ValueError('Each program is limited to 100 KB.')
    text = lexical(source)
    if len(token_stream(source)) < 30:
        return {'verdict':'Insufficient code', 'ai_score':None}
    p = float(load_model('authorship').predict_proba([text])[0,1])
    return {'verdict':'Likely AI-written' if p >= .7 else 'Likely human-written' if p <= .3 else 'Uncertain authorship', 'ai_score':round(p,4)}

def detect(left, right=None):
    result = {'authorship_a':authorship(left)}
    if right and right.strip():
        result['authorship_b'] = authorship(right)
        counts = (len(token_stream(left)), len(token_stream(right)))
        if min(counts) < 30:
            result['copy'] = {'verdict':'Insufficient code', 'copy_score':None}
        else:
            p = float(load_model('copy').predict_proba([pair_features(left,right)])[0,1])
            result['copy'] = {'verdict':'Likely copied/derived' if p >= .7 else 'No strong copy match' if p <= .3 else 'Uncertain copy relationship', 'copy_score':round(p,4), 'k':select_k(*counts)}
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Automatic ML code predictions')
    parser.add_argument('left',type=Path)
    parser.add_argument('right',type=Path,nargs='?')
    args=parser.parse_args()
    print(json.dumps(detect(args.left.read_text(),args.right.read_text() if args.right else None),indent=2))

if __name__ == '__main__':
    main()
