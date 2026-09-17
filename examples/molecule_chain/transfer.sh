#!/usr/bin/env bash
# transfer.sh — package the molecule_chain example together with the
# aimsChain library and tools it depends on, for transfer to a remote
# machine (e.g. the SLURM cluster that runs FHI-AIMS).
#
# The tarball preserves the aimsChain/examples/molecule_chain layout, so
# the driver's repo-root detection (run_chain.py -> ../../..) works
# unchanged on the remote.
#
# Usage:
#     ./transfer.sh               # -> molecule_chain_fhi_aims.tar.gz
#     ./transfer.sh --with-venv   # also ship the local .venv (see caveat)
#
# On the remote:
#     tar xzf molecule_chain_fhi_aims.tar.gz
#     cd aimsChain/examples/molecule_chain
#     ./setup_env.sh              # recreate the venv (reliable)
#     # or, if you shipped it:    mv .venv from the tarball root
#
# Caveat on --with-venv: a venv is bound to the base interpreter it was
# created with.  It works when the remote has the same python at the same
# path; otherwise use ./setup_env.sh instead.
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../.. && pwd)"                 # aimsChain repo root
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# --- example directory (this dir), minus run artifacts / venv / caches ---
EX="$STAGE/aimsChain/examples/molecule_chain"
mkdir -p "$EX"
tar -C . \
    --exclude='./.venv' \
    --exclude='./iterations' \
    --exclude='./paths' \
    --exclude='./optimized' \
    --exclude='./forces.log' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -cf - . | tar -C "$EX" -xf -

# --- aimsChain library + tools the driver imports ---
tar -C "$ROOT" \
    --exclude='./.git' \
    --exclude='./.venv' \
    --exclude='./examples' \
    --exclude='./samples' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -cf - src tools pyproject.toml README.md LICENSE \
    | tar -C "$STAGE/aimsChain" -xf -

# --- optional: local venv (fragile across machines, see caveat) ---
if [ "${1:-}" = "--with-venv" ]; then
    echo ">> including local .venv (portable only with a matching base python)"
    tar -C . -cf - .venv | tar -C "$EX" -xf -
fi

OUT="molecule_chain_fhi_aims.tar.gz"
tar -C "$STAGE" -czf "$OUT" .
echo ">> wrote $OUT"
echo "   remote:  tar xzf $OUT && cd aimsChain/examples/molecule_chain && ./setup_env.sh"
