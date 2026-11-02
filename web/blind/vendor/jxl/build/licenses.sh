#!/bin/bash
# Concatenate the licenses of everything linked into jxl.js/jxl.wasm: licenses.sh [out-file]
set -euo pipefail
out=$(realpath "${1:-/mnt/c/Users/User/Desktop/dev/jpxl/web/blind/vendor/jxl/LICENSE}")
cd /root/jxlwasm
: > "$out"
while IFS='|' read -r title path; do
  printf '================================================================================\n%s\n\n' "$title" >> "$out"
  tr -d '\r' < "$path" >> "$out"
  printf '\n' >> "$out"
done <<'EOF'
libjxl v0.12.0 (BSD-3-Clause)|libjxl/LICENSE
libjxl PATENTS (additional IP rights grant)|libjxl/PATENTS
Brotli (MIT)|libjxl/third_party/brotli/LICENSE
Highway (Apache-2.0 OR BSD-3-Clause; BSD-3-Clause text)|libjxl/third_party/highway/LICENSE-BSD3
skcms (BSD-3-Clause)|libjxl/third_party/skcms/LICENSE
libpng 1.6.59, emscripten port (libpng license)|emsdk/upstream/emscripten/cache/ports/libpng/libpng-1.6.59/LICENSE
zlib 1.3.2, emscripten port (zlib license)|emsdk/upstream/emscripten/cache/ports/zlib/zlib-1.3.2/LICENSE
Emscripten 6.0.13 runtime code in jxl.js (MIT OR UIUC/NCSA)|emsdk/upstream/emscripten/LICENSE
EOF
wc -l "$out"
