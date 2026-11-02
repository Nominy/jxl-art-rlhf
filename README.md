# jxl-art-rlhf

A 3M-parameter language model that writes [JPEG XL art](https://jpegxl.info/art/): tiny programs
(MA-tree "recipes") that libjxl's `jxl_from_tree` compiles into images of at most 256 bytes. It
is trained to write *valid random* programs, then tuned toward *beautiful* ones with RLHF on your
own A/B picks.

- Write-up: [nafty.is-a.dev/posts/jxl-art-rlhf](https://nafty.is-a.dev/posts/jxl-art-rlhf/)
- Blind test in your browser (the model runs on your device): [nafty.is-a.dev/jxl-blind-test](https://nafty.is-a.dev/jxl-blind-test/)
- Weights and the full RLHF run (every pick, image and snapshot): [huggingface.co/Nafty/jxl-art-rlhf](https://huggingface.co/Nafty/jxl-art-rlhf)

## How it works

1. **Data** (`jxlart.build_data`): random programs from the tree grammar plus mutations of
   accepted ones, compiled with libjxl and kept only if valid, ≤256 bytes, non-blank and not
   duplicates. No gallery program is copied or mutated; anything ≥0.8 similar to a known piece
   is rejected.
2. **SFT** (`jxlart.sft`): the [Supra2.5-Pico](https://huggingface.co/SupraLabs/Supra2.5-Pico)
   architecture (and tokenizer) trained on that data.
3. **RLHF** (`jxlart.rlhf.loop`): a local web page shows pairs from the current model; you pick
   the better one. A Bradley–Terry "taste model" on CLIP features refits after every pick; every
   25 picks the policy takes 20 KL-regularised policy-gradient steps, then a fresh batch is
   sampled. Rewards include a variety bonus (penalising look-alikes within a batch) and a
   penalty for reproducing known gallery art (`jxlart.novelty`).

## Setup

```bash
git clone https://github.com/Nominy/jxl-art-rlhf && cd jxl-art-rlhf
python -m pip install -r requirements.txt
hf auth login                       # only needed to push; the run is public at huggingface.co/Nafty/jxl-art-rlhf
python -m jxlart.hub setup          # base model files, CLIP, gallery art (art/), checks libjxl tools
```

libjxl tools (`tools/bin/jxl_from_tree`, `tools/bin/djxl`, libjxl v0.12.0):

- **macOS**: `xcode-select --install && brew install cmake && scripts/build_libjxl.sh`
  (Homebrew's `jpeg-xl` lacks `jxl_from_tree`, so it is built from source).
- **Windows**: from the [v0.12.0 release](https://github.com/libjxl/libjxl/releases/tag/v0.12.0)
  `jxl-x64-windows-static.zip`, extract `jxl_from_tree.exe` and `djxl.exe` into `tools/bin/`
  (the zip uses Deflate64: use 7-Zip).
- **Linux**: `scripts/build_libjxl.sh`.

## Continue on another machine

```bash
# machine A: stop the loop (create runs/rlhf-v2/STOP), then
python -m jxlart.hub push --run runs/rlhf-v2
# machine B
python -m jxlart.hub pull --run runs/rlhf-v2
python -m jxlart.rlhf.loop --run runs/rlhf-v2        # open http://127.0.0.1:8765
```

A run folder is self-contained: `policy/step_00000.safetensors` is the starting model (and KL
reference), `policy/online.safetensors` the current one, plus snapshots every 5 steps, all picks,
images and logs. Weights are plain safetensors shared by both backends, so a run moves between
PyTorch/CUDA and MLX freely; only the optimizer's momentum restarts when the backend changes.
`push` refuses to overwrite a run that was pushed from another machine since your last sync.

## Backends

| | policy (sampling + RL) | CLIP features, taste model |
|---|---|---|
| Windows/Linux + NVIDIA | PyTorch, CUDA graphs (`--backend torch`, default) | PyTorch CUDA |
| macOS, Apple silicon | MLX (`--backend mlx`, default on macOS) | PyTorch MPS |

## Commands

```bash
python -m jxlart.rlhf.loop --run runs/rlhf-v2                 # label + train (resume)
python -m jxlart.rlhf.loop --run runs/new --sft path/to/sft.safetensors   # start a new run
python -m jxlart.generate --policy runs/rlhf-v2/policy/online.safetensors --n 128 --out out/latest
python -m jxlart.novelty --run runs/rlhf-v2 --out reports/novelty        # copying vs the gallery
python -m jxlart.evaluate --run runs/rlhf-v2 --train data/synth --out reports/eval   # step-by-step metrics
python -m jxlart.blind_ab --run runs/rlhf-v2 --a step_00000 --b step_00820 --out reports/blind-ab   # blind A/B
python -m jxlart.build_data --run runs/data --out data/synth --target 24000
python -m jxlart.sft --run runs/sft --data data/synth --init scratch --gallery-repeat 0
python -m jxlart.launch <session> <run-dir> -- <command>      # detached psmux/tmux session
```

Long jobs write `status.json`, `events.jsonl` and `run.log` to their run folder; create a `STOP`
file there (or press Ctrl+C once) for a graceful stop that saves a checkpoint.

Labeling keys: ← / → pick, ↓ equal, `x` both bad, `u` undo, `s` skip, `c` show the recipe.

## In the browser

`web/blind/` is a static page that runs the policy in plain JavaScript (`lib/slm.js`) and libjxl as
WebAssembly (`vendor/jxl/`, from [hjanuschka/jxl-art](https://github.com/hjanuschka/jxl-art)) to
play the blind A/B test without a server. `python -m jxlart.export_web --run runs/rlhf-v2 --out
web/blind/assets` writes the float16 weights and tokenizer table; serve the folder with any static
server (`python -m http.server -d web/blind`). `tests/test_web_parity.py` and `tests/test_web_tree.py`
check the JavaScript model, tokenizer, tree parser and codec against the Python pipeline.

## Credits

- Model architecture and tokenizer: [SupraLabs/Supra2.5-Pico](https://huggingface.co/SupraLabs/Supra2.5-Pico) (Apache-2.0).
- Image features: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32).
- Compiler/decoder: [libjxl](https://github.com/libjxl/libjxl) (BSD-3-Clause).
- Gallery art belongs to its authors (credited in `art/manifest.json`); it is downloaded locally
  by `python -m jxlart.fetch_gallery` and not redistributed here.
- In-browser libjxl: our Emscripten build of libjxl v0.12.0 (`web/blind/vendor/jxl/`), build script adapted from [hjanuschka/jxl-art](https://github.com/hjanuschka/jxl-art) (Apache-2.0).
