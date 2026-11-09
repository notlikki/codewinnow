"""Local-only browser interface for pairwise source comparison."""

from __future__ import annotations

import argparse
import html
import re
import tokenize
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from .core import compare_sources
from .ml import detect


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CodeWinnow</title>
<style>
body{max-width:1000px;margin:2rem auto;padding:0 1rem;font:15px system-ui;color:#172033;background:#fafafa}
h1{font-size:1.3rem;margin:0 0 1.2rem}
form{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
textarea{width:100%;min-height:300px;box-sizing:border-box;padding:.8rem;font:13px ui-monospace,monospace;border:1px solid #b9c1cc;border-radius:6px;background:#fff}
button{margin-top:1rem;padding:.6rem 1.2rem;border:0;border-radius:6px;background:#155eef;color:#fff;font-weight:600;cursor:pointer}
.metrics{display:flex;gap:2rem;margin:1.5rem 0 1rem;padding:1rem 1.2rem;background:#fff;border-radius:8px;box-shadow:0 1px 3px #0001}
.metrics div{display:flex;flex-direction:column;gap:.2rem}
.metrics span:first-child{font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;color:#596579}
.metrics span:last-child{font-size:1.6rem;font-weight:700;font-variant-numeric:tabular-nums}
.verdict{padding:.9rem 1.2rem;border-radius:8px;font-size:1.1rem;font-weight:700}
.same{background:#dcfce7;color:#166534}.diff{background:#fee2e2;color:#991b1b}
.unsure{background:#fef9c3;color:#854d0e}.none{background:#e5e7eb;color:#374151}
.error{padding:1rem 1.2rem;background:#fee2e2;color:#991b1b;border-radius:8px;margin-top:1rem}
@media(max-width:700px){form{grid-template-columns:1fr}}
</style></head><body><h1>CodeWinnow</h1>
<form method="post"><label>Program A<textarea name="left" spellcheck="false" required>{left}</textarea></label>
<label>Program B<textarea name="right" spellcheck="false" required>{right}</textarea></label>
<div><button type="submit">Compare</button></div></form>
{result}</body></html>"""

VERDICTS = {
    "Likely copied/derived": ("same", "SAME CODE"),
    "No strong copy match": ("diff", "NOT THE SAME"),
    "Uncertain copy relationship": ("unsure", "UNCERTAIN"),
    "Insufficient code": ("none", "TOO SHORT TO JUDGE"),
}


def render(left: str = "", right: str = "", result: str = "") -> bytes:
    values = {'left': html.escape(left), 'right': html.escape(right), 'result': result}
    page = re.sub(r'\{(left|right|result)\}', lambda match: values[match.group(1)], PAGE)
    return page.encode()


def results(left: str, right: str) -> str:
    score = compare_sources(left, right)
    verdict = detect(left, right)["copy"]
    css, label = VERDICTS.get(verdict["verdict"], ("none", verdict["verdict"].upper()))
    confidence = verdict.get("copy_score")
    detail = "" if confidence is None else f" · {confidence:.0%} confidence"
    return (
        "<div class='metrics'>"
        f"<div><span>Token similarity</span><span>{score.token_similarity:.1%}</span></div>"
        f"<div><span>AST similarity</span><span>{score.ast_similarity:.1%}</span></div>"
        "</div>"
        f"<div class='verdict {css}'>{label}{detail}</div>"
    )


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
            self._send(render(result="<p class='error'>Invalid request length.</p>"), 400)
            return
        if length > 2_000_000:
            self._send(render(result="<p class='error'>Input is limited to 2 MB.</p>"), 413)
            return
        form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
        left, right = form.get("left", [""])[0], form.get("right", [""])[0]
        try:
            self._send(render(left, right, results(left, right)))
        except (SyntaxError, tokenize.TokenError, ValueError, RecursionError) as error:
            self._send(render(left, right, "<p class='error'>" + html.escape(str(error)) + "</p>"), 400)

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
