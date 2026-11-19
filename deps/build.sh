#!/usr/bin/env bash
# Fetch bucket-graph-spprc at a pinned commit and compile the C shim
# (capi/bgspprc_c.cpp) into deps/build/libbgspprc_c.{dylib,so}.
# Needs git and a C++23 compiler (Clang >= 18 or GCC >= 14); set CXX to choose one.
set -euo pipefail
cd "$(dirname "$0")"

BGSPPRC_URL=https://github.com/spoorendonk/bucket-graph-spprc.git
BGSPPRC_COMMIT=afdc56c5d63632b27cc5df0514b3bd6073e253b3

if [ ! -d bucket-graph-spprc ]; then
    git clone --quiet "$BGSPPRC_URL" bucket-graph-spprc
fi
git -C bucket-graph-spprc fetch --quiet origin "$BGSPPRC_COMMIT" 2>/dev/null || true
git -C bucket-graph-spprc checkout --quiet "$BGSPPRC_COMMIT"

case "$(uname -s)" in
    Darwin) ext=dylib ;;
    *) ext=so ;;
esac
mkdir -p build
"${CXX:-c++}" -std=c++23 -O3 -DNDEBUG -fPIC -shared -pthread \
    -I bucket-graph-spprc/include ../capi/bgspprc_c.cpp -o "build/libbgspprc_c.$ext"
echo "built deps/build/libbgspprc_c.$ext"
