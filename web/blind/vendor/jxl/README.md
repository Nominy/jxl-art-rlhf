# libjxl v0.12.0 → WebAssembly (jxl_from_tree + djxl-equivalent PNG decode)

`jxl.js` + `jxl.wasm` are an Emscripten build of [libjxl](https://github.com/libjxl/libjxl) **v0.12.0**
(tag commit `a7a9c787341cf703dede03c2009fa460cae5e5df`, the same release as `tools/bin/jxl_from_tree.exe`
and `djxl.exe`), Emscripten 6.0.13, `-O3 -msimd128` (WebAssembly SIMD: Chrome 91+, Firefox 89+,
Safari 16.4+). Exact versions are in `build/build-info.json`; the sources are not modified.

The module exports two Embind functions, wrapped by `../../lib/codec.js`:

- `jxl_from_tree(text)`: libjxl's `tools/jxl_from_tree.cc` `JxlFromTree` on an in-memory file;
  returns the JXL bytes or an error string (details on stderr).
- `decode(jxl)`: `DecodeImageJXL` + the APNG encoder with the parameters of
  `djxl in.jxl - --output_format png --bits_per_sample=8 --num_threads=1 --color_space=RGB_D65_SRG_Rel_SRG`
  (`jxlart/compiler.py decode`): an 8-bit sRGB PNG, RGBA when the image has alpha.

## Rebuild

`build/build.sh` (run as root in WSL Ubuntu 24.04 / any Linux; resumable, ~5 min on 12 cores) installs
emsdk, clones libjxl at the tag, builds `jxl`/`jxl_cms` with emcmake and links `build/jxl_wasm.cpp`.
`SIMD=1` gives this build (outputs in `$RUN/out-simd`); without it a scalar build. `build/licenses.sh`
assembles `LICENSE` from the build tree. The script is adapted from `scripts/build-libjxl.sh` of
[hjanuschka/jxl-art](https://github.com/hjanuschka/jxl-art) (Apache-2.0), whose prebuilt wasm
(branch `modernize-jxl-art`, commit `54807b4`, libjxl `a16d32d`) was evaluated first and not used: its
`decode` keeps the codestream bit depth (16-bit PNGs for `Bitdepth` > 8, 0..1-valued 8-bit PNGs for
`Bitdepth 1`) and does not convert to sRGB, and libjxl HEAD encodes different header bits than v0.12.0.

## Parity (tests/test_web_tree.py)

JXL bytes match the native tool except where the native tool itself is nondeterministic:
`jxl_from_tree` encodes an uninitialised `Image3F`, and for a few percent of programs the encoder's
choices depend on that heap garbage (output varies with the process environment; every wasm output
was reproduced by the native tool under some environment). The decoder differs from native djxl by at
most 1 (float rounding, AVX2 vs WASM) in a small fraction of pixels.

## Licenses

`LICENSE` contains libjxl (BSD-3-Clause, plus its PATENTS grant), Brotli (MIT), Highway (BSD-3-Clause
option), skcms (BSD-3-Clause), libpng 1.6.59 and zlib 1.3.2 (Emscripten ports), and the Emscripten
runtime (MIT).
