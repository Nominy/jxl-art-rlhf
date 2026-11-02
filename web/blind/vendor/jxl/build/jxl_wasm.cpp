// Embind wrapper exposing libjxl's jxl_from_tree and a djxl-equivalent PNG decode.
// decode() mirrors `djxl in.jxl - --output_format png --bits_per_sample=8
//   --num_threads=1 --color_space=RGB_D65_SRG_Rel_SRG` (jxlart/compiler.py decode).
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "lib/extras/dec/jxl.h"
#include "lib/extras/enc/apng.h"
#include "lib/extras/enc/encode.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/status.h"

using emscripten::val;

namespace jpegxl {
namespace tools {
::jxl::Status JxlFromTree(const char* in, const char* out, const char* tree_out);
}  // namespace tools
}  // namespace jpegxl

// Result buffer; the returned typed_memory_view stays valid until the next call (JS copies it).
static std::vector<uint8_t> g_out;

static val view() {
  return val(emscripten::typed_memory_view(g_out.size(), g_out.data()));
}

val jxl_from_tree(std::string code) {
  {
    std::ofstream in("/in.txt", std::ios::binary);
    in << code;
    if (!in) return val("cannot write /in.txt");
  }
  std::remove("/out.jxl");
  if (!jpegxl::tools::JxlFromTree("/in.txt", "/out.jxl", nullptr)) {
    return val("jxl_from_tree failed");
  }
  std::ifstream out("/out.jxl", std::ios::binary);
  if (!out) return val("jxl_from_tree wrote no output");
  g_out.assign(std::istreambuf_iterator<char>(out), std::istreambuf_iterator<char>());
  return view();
}

val decode(val input) {
  std::vector<uint8_t> jxl = emscripten::convertJSArrayToNumberVector<uint8_t>(input);
  std::unique_ptr<jxl::extras::Encoder> png = jxl::extras::GetAPNGEncoder();
  if (!png) return val("PNG encoder unavailable");
  jxl::extras::JXLDecompressParams dparams;
  dparams.max_downsampling = 0;
  dparams.accepted_formats = png->AcceptedFormats();
  dparams.display_nits = 0.0;
  dparams.color_space = "RGB_D65_SRG_Rel_SRG";
  dparams.color_space_for_cmyk = "sRGB";
  dparams.render_spotcolors = true;
  dparams.coalescing = true;
  dparams.allow_partial_input = false;
  dparams.runner = nullptr;  // single-threaded (djxl --num_threads=1 gives identical output)
  dparams.runner_opaque = nullptr;
  dparams.output_bitdepth.type = JXL_BIT_DEPTH_CUSTOM;
  dparams.output_bitdepth.bits_per_sample = 8;
  jxl::extras::PackedPixelFile ppf;
  size_t decoded_bytes = 0;
  if (!jxl::extras::DecodeImageJXL(jxl.data(), jxl.size(), dparams, &decoded_bytes, &ppf)) {
    return val("DecompressJxlToPackedPixelFile failed");
  }
  jxl::extras::EncodedImage encoded;
  if (!png->Encode(ppf, &encoded, nullptr) || encoded.bitstreams.empty()) {
    return val("PNG encode failed");
  }
  g_out = std::move(encoded.bitstreams.front());
  return view();
}

EMSCRIPTEN_BINDINGS(jxl_art) {
  emscripten::function("jxl_from_tree", &jxl_from_tree);
  emscripten::function("decode", &decode);
}
