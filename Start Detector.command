#!/bin/zsh
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv || exit 1
  .venv/bin/python -m pip install -r requirements.txt || exit 1
fi
(sleep 1; open http://127.0.0.1:8000) &
.venv/bin/python -m plagiarism_detector.web
