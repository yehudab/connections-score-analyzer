#!/usr/bin/env bash
# Wrapper: activates the venv and runs connections_solver.py
# Usage:  ./solve.sh [--no-cdp] [--headed] [--debug] [--output win.png]
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
exec python connections_solver.py "$@"
