#!/bin/bash
# Install pinned libxgboost.so for the xgboost janet module (modules/janet-xgboost.janet).
# GitHub ships no prebuilt lib; the official prebuilt artifact is the PyPI
# manylinux wheel (xgboost project's own release artifact - no python is
# installed or used). 3.x ships only the universal wheel (CUDA compiled in,
# CPU fallback at runtime - no GPU needed). Fetches the wheel, verifies
# sha256, extracts libxgboost.so + its bundled libgomp (auditwheel rpath
# requires the relative layout). Idempotent; re-run to reinstall.
set -e
VERSION="3.4.1"
WHEEL="xgboost-${VERSION}-py3-none-manylinux_2_28_x86_64.whl"
URL="https://files.pythonhosted.org/packages/99/c7/bd05c5c430feb347aa040fcc8870135d70b256718deee9bc7d2ca74a77ff/${WHEEL}"
SHA256="6adf2afa396da2ae8ed30295b50b99d4712eed9a6e0ce6cfe069290e4335e51f"
DEST="$HOME/.local/lib/xgboost"

command -v curl >/dev/null || { echo "error: curl not found" >&2; exit 1; }
command -v unzip >/dev/null || { echo "error: unzip not found" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "fetching xgboost v${VERSION} (official wheel) ..."
curl -fsSL -o "$TMP/$WHEEL" "$URL"
echo "$SHA256  $TMP/$WHEEL" | sha256sum -c -

mkdir -p "$DEST"
unzip -q -o "$TMP/$WHEEL" "xgboost/lib/libxgboost.so" "xgboost.libs/*" -d "$DEST"
echo "installed: $DEST/xgboost/lib/libxgboost.so"
echo "usage    : (import janet-xgboost) - resolution: \$XGBOOST_LIB, then the path above"
