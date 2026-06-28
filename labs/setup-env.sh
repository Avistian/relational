#!/usr/bin/env bash
# Create and activate the lab virtual environment (run from repo root).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Creating .venv ..."
  python3 -m venv .venv
fi

echo "Installing lab dependencies ..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-labs.txt

echo "Registering Jupyter kernel 'relational-labs' ..."
.venv/bin/python -m ipykernel install --user --name relational-labs --display-name "Relational Labs (.venv)"

echo ""
echo "Done. Activate with:"
echo "  source .venv/bin/activate"
echo ""
echo "In Cursor/VS Code: select interpreter → $ROOT/.venv/bin/python"
echo "In Jupyter: pick kernel → Relational Labs (.venv)"
