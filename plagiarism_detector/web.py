"""Local-only browser interface for pairwise source comparison."""

from __future__ import annotations

import argparse
import html
import json
import re
import tokenize
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from .core import compare_sources
from .ml import detect


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CodeWinnow — Code Similarity Detector</title>
<style>
body{max-width:1100px;margin:2rem auto;padding:0 1rem;font:16px system-ui;color:#172033;background:#fafafa}
h1{margin-bottom:.2rem}.note{color:#596579}form{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
textarea{width:100%;min-height:340px;box-sizing:border-box;padding:.8rem;font:13px ui-monospace,monospace;border:1px solid #b9c1cc;border-radius:6px}
button{margin-top:1rem;padding:.65rem 1rem;border:0;border-radius:6px;background:#155eef;color:white;font-weight:600;cursor:pointer}
.result{padding:1rem;margin:1.2rem 0;background:white;border-left:5px solid #155eef;box-shadow:0 1px 3px #0001}
.score{font-size:1.3rem;font-weight:700}.warning{font-size:.9rem;color:#596579}
@media(max-width:700px){form{grid-template-columns:1fr}}
</style></head><body><h1>CodeWinnow</h1>
<p class="note">Local-only Python plagiarism detection via token winnowing and AST normalisation. Paste a Python program for an automatic authorship prediction. Add a second program to check for copying. Experimental models; scores are not proof of authorship or copying.</p>
{result}
<form method="post"><label>Program A<textarea name="left" spellcheck="false" required>{left}</textarea></label>
<label>Program B (optional reference)<textarea name="right" spellcheck="false">{right}</textarea></label>
<div><button type="submit">Run ML detection</button></div></form></body></html>"""


def render(left: str = "", right: str = "", result: str = "") -> bytes:
    values = {'left': html.escape(left), 'right': html.escape(right), 'result': result}
    page = re.sub(r'\{(left|right|result)\}', lambda match: values[match.group(1)], PAGE)
    return page.encode()


class DetectorHandler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (HTTP method name)
        self._send(render())

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0: raise ValueError('negative length')
        except ValueError:
            self._send(render(result="<p class='result'>Invalid request length.</p>"), 400)
            return
        if length > 2_000_000:
            self._send(render(result="<p class='result'>Input is limited to 2 MB.</p>"), 413)
            return
        form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
        left, right = form.get("left", [""])[0], form.get("right", [""])[0]
        try:
            prediction = detect(left, right)
            cards = []
            for name, item in prediction.items():
                value = item.get('copy_score', item.get('ai_score'))
                detail = '' if value is None else f" — model score: {value:.1%}"
                cards.append("<div class='score'>" + html.escape(name.replace('_',' ').title() + ': ' + item['verdict'] + detail) + "</div>")
            if right.strip():
                score = compare_sources(left, right)
                cards.append(f"<p>Token similarity: {score.token_similarity:.1%}; AST similarity: {score.ast_similarity:.1%}</p><p>Automatic k: {score.k}; tokens A/B: {score.token_counts[0]}/{score.token_counts[1]}; window: {score.window}</p>")
            cards.append("<p class='warning'>Experimental baseline. Scores are uncalibrated model outputs. Copy training uses synthetic variants; authorship results may not generalize to new generators or classroom work.</p>")
            self._send(render(left, right, "<section class='result'>" + ''.join(cards) + "</section>"))
        except (SyntaxError, tokenize.TokenError, ValueError, RecursionError) as error:
            self._send(render(left, right, "<p class='result'>Input/model error: " + html.escape(str(error)) + "</p>"), 400)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.client_address[0]} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local code-similarity web interface")
    parser.add_argument("--host", default="127.0.0.1", help="default: 127.0.0.1 (local only)")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DetectorHandler)
    print(f"CodeWinnow running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
