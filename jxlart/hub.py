"""Move models and RLHF runs between machines through a private Hugging Face repo.

  python -m jxlart.hub setup                      # base model files, CLIP, gallery art (once per machine)
  python -m jxlart.hub push --run runs/rlhf-v2    # upload the run (weights, snapshots, picks, images, logs)
  python -m jxlart.hub pull --run runs/rlhf-v2    # download it (stop the loop first on both sides)

A run dir is self-contained: policy/step_00000.safetensors is the SFT policy and KL reference,
policy/online.safetensors the current weights. Weights are framework-neutral safetensors, so a run
pushed from PyTorch/CUDA continues under MLX and back; only the Adam state restarts on a backend
switch. Each push writes hub.json (host, time, step, picks); `push` refuses when the remote was
pushed from elsewhere since this machine's last sync, so two devices never silently overwrite labels.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import socket
import sys
import time
import uuid
from pathlib import Path

from .compiler import ROOT

DEFAULT_REPO = "Nafty/jxl-art-rlhf"
LOCAL_ONLY = ["STOP", "status.json", "exit_code.txt", "launch.json", "*.tmp", ".hub_state.json"]
BASE_FILES = {
    "SupraLabs/Supra2.5-Pico": ("models/Supra2.5-Pico", ["config.json", "modeling_slm.py", "tokenizer.json",
                                                         "tokenizer_config.json", "tokenization_pascaline.py",
                                                         "special_tokens_map.json", "model.safetensors", "LICENSE"]),
    "openai/clip-vit-base-patch32": ("models/clip-vit-base-patch32", ["config.json", "preprocessor_config.json",
                                                                       "tokenizer.json", "tokenizer_config.json",
                                                                       "vocab.json", "merges.txt",
                                                                       "special_tokens_map.json", "pytorch_model.bin"]),
}


def _active(run: Path) -> bool:
    """True if the loop is running on this machine (fresh heartbeat in status.json)."""
    s = run / "status.json"
    if not s.exists():
        return False
    st = json.loads(s.read_text(encoding="utf-8"))
    return st.get("state") in ("running", "waiting", "initializing") and time.time() - st.get("heartbeat", 0) < 30


def _summary(run: Path) -> dict:
    step, picks = 0, 0
    try:
        from safetensors import safe_open
        with safe_open(str(run / "policy" / "online.safetensors"), "np") as f:
            step = json.loads(f.metadata()["meta"]).get("step", 0)
    except Exception:
        pass
    p = run / "prefs.jsonl"
    if p.exists():
        for line in p.open(encoding="utf-8"):
            picks += -1 if '"undo"' in line else 1
    return {"step": step, "picks": picks}


def _state(run: Path) -> dict:
    p = run / ".hub_state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _remote_hub_json(api, repo: str, name: str) -> dict | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    try:
        path = hf_hub_download(repo, f"runs/{name}/hub.json", cache_dir=ROOT / "cache" / "hub-meta",
                               force_download=True)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def setup() -> None:
    from huggingface_hub import hf_hub_download
    for repo, (dest, files) in BASE_FILES.items():
        for f in files:
            hf_hub_download(repo, f, local_dir=ROOT / dest)
        print(f"ok  {repo} -> {dest}")
    if not (ROOT / "art" / "manifest.json").exists():
        from . import fetch_gallery
        sys.argv = [sys.argv[0]]
        fetch_gallery.main()
    from . import compiler
    missing = [p.name for p in (compiler.JXL_FROM_TREE, compiler.DJXL) if not p.exists()]
    if missing:
        hint = "scripts/build_libjxl.sh" if sys.platform != "win32" else "README: Windows tools"
        print(f"missing libjxl tools {missing}: run {hint}")
    else:
        print("ok  libjxl tools")


def push(run: Path, repo: str, force: bool) -> None:
    from huggingface_hub import HfApi
    if _active(run):
        sys.exit(f"{run} is running here; stop it first (create {run / 'STOP'})")
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    name = run.name
    remote = _remote_hub_json(api, repo, name)
    known = _state(run).get("push_id")
    if remote and remote["push_id"] != known and not force:
        sys.exit(f"remote run was pushed from {remote['host']} at {remote['time']} (step {remote['step']}, "
                 f"{remote['picks']} picks) after this machine last synced; pull first or use --force")
    info = {"push_id": uuid.uuid4().hex, "host": socket.gethostname(), "platform": platform.platform(),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **_summary(run)}
    (run / "hub.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    api.upload_folder(folder_path=str(run), path_in_repo=f"runs/{name}", repo_id=repo,
                      ignore_patterns=LOCAL_ONLY,
                      commit_message=f"{name}: step {info['step']}, {info['picks']} picks from {info['host']}")
    (run / ".hub_state.json").write_text(json.dumps({"push_id": info["push_id"]}), encoding="utf-8")
    print(f"pushed {run} -> https://huggingface.co/{repo}/tree/main/runs/{name}  ({info})")


def pull(run: Path, repo: str) -> None:
    from huggingface_hub import snapshot_download
    if _active(run):
        sys.exit(f"{run} is running here; stop it first (create {run / 'STOP'})")
    name = run.name
    staging = ROOT / "cache" / "hub" / repo.replace("/", "__")
    snapshot_download(repo, allow_patterns=[f"runs/{name}/**"], local_dir=staging)
    src = staging / "runs" / name
    if not src.exists():
        sys.exit(f"no runs/{name} in {repo}")
    run.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.rglob("*"):
        if f.is_file():
            dst = run / f.relative_to(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            n += 1
    info = json.loads((run / "hub.json").read_text(encoding="utf-8"))
    (run / ".hub_state.json").write_text(json.dumps({"push_id": info["push_id"]}), encoding="utf-8")
    print(f"pulled {n} files into {run}: pushed from {info['host']} at {info['time']} "
          f"(step {info['step']}, {info['picks']} picks)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "push", "pull"])
    ap.add_argument("--run", type=Path)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--force", action="store_true", help="push even if the remote changed since the last sync")
    args = ap.parse_args()
    if args.cmd == "setup":
        return setup()
    if not args.run:
        ap.error("--run is required")
    run = args.run if args.run.is_absolute() else ROOT / args.run
    if args.cmd == "push":
        push(run, args.repo, args.force)
    else:
        pull(run, args.repo)


if __name__ == "__main__":
    main()
