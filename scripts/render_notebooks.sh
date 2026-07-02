#!/usr/bin/env bash
# Render lab notebooks to static HTML for the web (view without Jupyter).
# Output: labs/html/<name>.html  (committed; served by GitHub Pages).
#
# Usage (from repo root):
#   bash scripts/render_notebooks.sh
#
# Requires the lab venv (jupyter nbconvert). Run bash labs/setup-env.sh first.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python3"
fi

OUT="labs/html"
mkdir -p "$OUT"

shopt -s nullglob
notebooks=(labs/[0-9][0-9][0-9][0-9]-*.ipynb)
if [ ${#notebooks[@]} -eq 0 ]; then
  echo "No lab notebooks found."
  exit 0
fi

echo "Rendering ${#notebooks[@]} notebook(s) to $OUT/ ..."
"$PY" -m jupyter nbconvert \
  --to html \
  --template lab \
  --output-dir "$OUT" \
  "${notebooks[@]}"

echo "Done. Rendered HTML in $OUT/:"
ls -1 "$OUT"
