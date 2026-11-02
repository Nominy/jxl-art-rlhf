#!/usr/bin/env bash
# Build libjxl's jxl_from_tree + djxl (pinned to the version used for training) into tools/bin/.
#
#   macOS:  xcode-select --install && brew install cmake && scripts/build_libjxl.sh
#   Linux:  needs git, a C++17 compiler and cmake >= 3.16 (`pip install cmake` works too)
#
# Homebrew's jpeg-xl package ships cjxl/djxl but not jxl_from_tree (a libjxl dev tool), hence
# the source build. Takes ~5-10 minutes; reruns reuse the build directory.
set -euo pipefail

VERSION="${LIBJXL_VERSION:-v0.12.0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${LIBJXL_SRC:-$ROOT/cache/libjxl-$VERSION}"
BUILD="$SRC/build"
JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu)"

command -v cmake >/dev/null || { echo "cmake not found (macOS: brew install cmake; Linux: pip install cmake)"; exit 1; }
command -v git >/dev/null || { echo "git not found"; exit 1; }

if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$VERSION" --recurse-submodules --shallow-submodules \
    https://github.com/libjxl/libjxl.git "$SRC"
fi

cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=OFF \
  -DJPEGXL_ENABLE_DEVTOOLS=ON -DJPEGXL_ENABLE_TOOLS=ON \
  -DJPEGXL_ENABLE_BENCHMARK=OFF -DJPEGXL_ENABLE_EXAMPLES=OFF -DJPEGXL_ENABLE_MANPAGES=OFF \
  -DJPEGXL_ENABLE_JNI=OFF -DJPEGXL_ENABLE_SJPEG=OFF -DJPEGXL_ENABLE_OPENEXR=OFF \
  -DJPEGXL_ENABLE_PLUGINS=OFF -DJPEGXL_ENABLE_DOXYGEN=OFF -DJPEGXL_BUNDLE_LIBPNG=ON
cmake --build "$BUILD" --target jxl_from_tree djxl -j "$JOBS"

mkdir -p "$ROOT/tools/bin"
for t in jxl_from_tree djxl; do
  cp "$(find "$BUILD/tools" -type f -name "$t" -perm -u+x | head -n 1)" "$ROOT/tools/bin/$t"
done

# smoke test: a one-line tree must compile and decode to PNG
TMP="$(mktemp -d)"
printf 'Width 64\nHeight 64\n- Set 128\n' > "$TMP/t.txt"
"$ROOT/tools/bin/jxl_from_tree" "$TMP/t.txt" "$TMP/t.jxl"
"$ROOT/tools/bin/djxl" "$TMP/t.jxl" "$TMP/t.png" --quiet
test -s "$TMP/t.png"
rm -rf "$TMP"
echo "ok: $ROOT/tools/bin/{jxl_from_tree,djxl} (libjxl $VERSION)"
