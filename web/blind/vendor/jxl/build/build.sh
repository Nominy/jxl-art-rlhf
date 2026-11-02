#!/bin/bash
# Build libjxl v0.12.0 (same release as tools/bin) to WebAssembly with jxl_wasm.cpp.
# Resumable: each stage leaves a marker in $WORK; make/link are incremental and rerun.
# Graceful stop: create $RUN/STOP (checked between stages); Ctrl+C mid-make is also safe
# (make resumes from finished objects on rerun).
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)   # holds jxl_wasm.cpp
RUN=${RUN:-/mnt/d/jxl-art/webab/wasm-build}
WORK=/root/jxlwasm
EMSDK_VERSION=6.0.13
LIBJXL_REF=v0.12.0
SIMD=${SIMD:-0}   # 1: build with -msimd128 (separate build/out dirs)
if [ "$SIMD" = 1 ]; then SUF=-simd FLAGS=-msimd128; else SUF= FLAGS=; fi
BUILD=$WORK/build$SUF
OUT=$RUN/out$SUF
mkdir -p "$WORK" "$OUT"
exec > >(tee -a "$RUN/run.log") 2>&1
T0=$(date +%s)

status() {  # state stage detail
  local now; now=$(date +%s)
  printf '{"state":"%s","stage":"%s","detail":"%s","elapsed_s":%d,"updated":"%s"}\n' \
    "$1" "$2" "$3" $((now - T0)) "$(date -Iseconds)" > "$RUN/status.json.tmp"
  mv "$RUN/status.json.tmp" "$RUN/status.json"
  printf '{"t":"%s","state":"%s","stage":"%s","detail":"%s"}\n' "$(date -Iseconds)" "$1" "$2" "$3" >> "$RUN/events.jsonl"
  echo "[$(date +%T)] $1 $2 $3"
}
check_stop() { if [ -f "$RUN/STOP" ]; then rm -f "$RUN/STOP"; status stopped "$1" "STOP file"; exit 0; fi; }
trap 'status failed "${STAGE:-?}" "exit $?"' ERR
stage() {  # name command...
  STAGE=$1; shift
  if [ -f "$WORK/.done-$STAGE" ]; then status skipped "$STAGE" "already done"; return; fi
  check_stop "$STAGE"
  status running "$STAGE" "start"
  "$@"
  touch "$WORK/.done-$STAGE"
  status done "$STAGE" ""
}

apt_deps() { command -v cmake >/dev/null || { apt-get update -q && apt-get install -y -q cmake; }; }
emsdk_install() {
  [ -d "$WORK/emsdk" ] || git clone -q https://github.com/emscripten-core/emsdk "$WORK/emsdk"
  cd "$WORK/emsdk" && ./emsdk install $EMSDK_VERSION && ./emsdk activate $EMSDK_VERSION
}
clone_libjxl() {
  rm -rf "$WORK/libjxl"
  git clone -q --depth 1 -b $LIBJXL_REF https://github.com/libjxl/libjxl "$WORK/libjxl"
  cd "$WORK/libjxl" && git submodule update --init --recursive --depth 1
}
configure() {
  mkdir -p "$BUILD" && cd "$BUILD"
  emcmake cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF \
    -DCMAKE_C_FLAGS="$FLAGS" -DCMAKE_CXX_FLAGS="$FLAGS" \
    -DJPEGXL_ENABLE_BENCHMARK=OFF -DJPEGXL_ENABLE_EXAMPLES=OFF -DJPEGXL_ENABLE_TOOLS=OFF \
    -DJPEGXL_ENABLE_MANPAGES=OFF -DJPEGXL_ENABLE_JNI=OFF -DJPEGXL_ENABLE_TCMALLOC=OFF \
    -DJPEGXL_ENABLE_SJPEG=OFF -DJPEGXL_ENABLE_OPENEXR=OFF -DJPEGXL_ENABLE_JPEGLI=OFF \
    -DJPEGXL_BUNDLE_LIBPNG=ON "$WORK/libjxl"
}
make_libs() {
  cd "$BUILD"
  local last=""
  emmake make -j"$(nproc)" jxl jxl_cms 2>&1 | while IFS= read -r line; do
    echo "$line"
    if [[ $line =~ ^\[\ *([0-9]+)%\] ]] && [ "${BASH_REMATCH[1]}" != "$last" ]; then
      last=${BASH_REMATCH[1]}; status running make_libs "${last}%"
    fi
  done
}
link() {
  local J=$WORK/libjxl B=$BUILD
  em++ -O3 $FLAGS -std=c++17 --bind --closure 1 -DJPEGXL_ENABLE_APNG=1 -DJXL_DEBUG_ON_ERROR=1 \
    -I"$J" -I"$J/lib/include" -I"$B/lib/include" -I"$J/third_party/highway" \
    -I"$J/third_party/skcms" -I"$J/third_party/googletest/googletest/include" \
    -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createJxlModule -sENVIRONMENT=web,worker \
    -sINVOKE_RUN=0 -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=1MB -sUSE_LIBPNG=1 \
    -o "$OUT/jxl.js" \
    "$SRC/jxl_wasm.cpp" "$J/tools/jxl_from_tree.cc" "$J/tools/no_memory_manager.cc" \
    "$J/lib/extras/common.cc" "$J/lib/extras/exif.cc" "$J/lib/extras/packed_image.cc" \
    "$J/lib/extras/dec/jxl.cc" "$J/lib/extras/dec/color_description.cc" \
    "$J/lib/extras/enc/encode.cc" "$J/lib/extras/enc/apng.cc" \
    "$B/lib/libjxl.a" "$B/lib/libjxl_cms.a" \
    "$B/third_party/brotli/libbrotlienc.a" "$B/third_party/brotli/libbrotlidec.a" \
    "$B/third_party/brotli/libbrotlicommon.a" "$B/third_party/highway/libhwy.a"
  cat > "$OUT/build-info.json" <<EOF
{"libjxl": "$LIBJXL_REF", "libjxl_commit": "$(git -C "$J" rev-parse HEAD)", "emscripten": "$(emcc --version | head -1)", "cflags": "$FLAGS", "built": "$(date -Iseconds)"}
EOF
}

status running start "emsdk $EMSDK_VERSION libjxl $LIBJXL_REF"
stage apt apt_deps
stage emsdk emsdk_install
source "$WORK/emsdk/emsdk_env.sh" >/dev/null
stage clone clone_libjxl
stage configure$SUF configure
check_stop make_libs
STAGE=make_libs; status running make_libs "start"; make_libs; status done make_libs ""
check_stop link
STAGE=link; status running link "start"; link; status done link ""
ls -l "$OUT"
status completed all "$OUT"
