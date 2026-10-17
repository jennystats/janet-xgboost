#!/bin/bash
# Install the pinned xgboost C library for the janet-xgboost module
# (jennystats/xgboost.janet). GitHub ships no prebuilt lib; the official
# prebuilt artifacts are the PyPI wheels (the xgboost project's own
# release artifacts - no python is installed or used). Detects the
# platform, fetches the matching official wheel, verifies sha256,
# extracts the library. Idempotent; re-run to reinstall.
#
# Per-platform layout: the linux manylinux wheel bundles libgomp
# (auditwheel rpath needs the relative xgboost.libs/ layout); the
# macOS wheels ship a single self-contained libxgboost.dylib.
# Windows: the xgboost project publishes no libxgboost.dll wheel
# artifact this binding can use - see the README's platform notes.
set -e
VERSION="3.4.1"
DEST="$HOME/.local/lib/xgboost"
# stock macOS has no sha256sum; its perl-based shasum reads the same
# "<hash>  <path>" checklist format
CHECKSUM="sha256sum"
[ "$(uname -s)" = Darwin ] && CHECKSUM="shasum -a 256"

OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS/$ARCH" in
  Linux/x86_64)  WHEEL="xgboost-${VERSION}-py3-none-manylinux_2_28_x86_64.whl" ;;
  Darwin/arm64)  WHEEL="xgboost-${VERSION}-py3-none-macosx_12_0_arm64.whl" ;;
  Darwin/x86_64) WHEEL="xgboost-${VERSION}-py3-none-macosx_10_15_x86_64.whl" ;;
  MINGW*/AMD64 | MSYS*/AMD64 | CYGWIN*/AMD64)
    echo "error: no official libxgboost.dll artifact - on Windows run this stack under WSL2 (see the README's platform notes)" >&2
    exit 1 ;;
  *)
    echo "error: unsupported platform $OS/$ARCH - check the xgboost PyPI page for an artifact matching your platform" >&2
    exit 1 ;;
esac

case "$WHEEL" in
  xgboost-${VERSION}-py3-none-manylinux_2_28_x86_64.whl)
    SHA256="6adf2afa396da2ae8ed30295b50b99d4712eed9a6e0ce6cfe069290e4335e51f"
    URL="https://files.pythonhosted.org/packages/99/c7/bd05c5c430feb347aa040fcc8870135d70b256718deee9bc7d2ca74a77ff/${WHEEL}" ;;
  xgboost-${VERSION}-py3-none-macosx_12_0_arm64.whl)
    SHA256="a7afd7dbace0951c93aa85ffe046e54bc40893f5b51cd3e7991eb157bf9c7c7c"
    URL="https://files.pythonhosted.org/packages/f7/94/e5c37a8972ad780edc1d8459d1931356344ca133f7f99ba9cfda516b5bba/${WHEEL}" ;;
  xgboost-${VERSION}-py3-none-macosx_10_15_x86_64.whl)
    SHA256="1ea15f15f661825b6a67d87674fb9604a1abb38dd0d4c5cf0486fc85f5203e83"
    URL="https://files.pythonhosted.org/packages/57/ea/0bdcd374241a86f1986e87e272516f0a70d841c3aa86aa9ca167fb651573/${WHEEL}" ;;
esac

command -v curl >/dev/null || { echo "error: curl not found" >&2; exit 1; }
command -v unzip >/dev/null || { echo "error: unzip not found" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "fetching xgboost v${VERSION} for $OS/$ARCH (official wheel) ..."
curl -fsSL -o "$TMP/$WHEEL" "$URL"
echo "$SHA256  $TMP/$WHEEL" | $CHECKSUM -c -

mkdir -p "$DEST"
case "$OS" in
  Darwin) unzip -q -o "$TMP/$WHEEL" "xgboost/lib/libxgboost.dylib" -d "$DEST"
          echo "installed: $DEST/xgboost/lib/libxgboost.dylib" ;;
  *)      unzip -q -o "$TMP/$WHEEL" "xgboost/lib/libxgboost.so" "xgboost.libs/*" -d "$DEST"
          echo "installed: $DEST/xgboost/lib/libxgboost.so" ;;
esac
echo "usage    : (import jennystats/xgboost) - resolution: \$XGBOOST_LIB, then the path above"
