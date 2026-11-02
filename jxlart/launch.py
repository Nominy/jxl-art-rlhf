"""Launch a long-running job in a detached terminal-multiplexer session with a durable run dir.

  python -m jxlart.launch <session-name> <run-dir> -- python -m jxlart.rlhf.loop --run <run-dir> ...

Windows uses psmux (PowerShell), macOS/Linux use tmux. Writes <run-dir>/launch.json, records the
exit code in <run-dir>/exit_code.txt, and the session closes when the job ends (logs, status and
events stay in the run dir). Prints attach / status / graceful-stop / resume instructions.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .compiler import ROOT

MUX = "psmux" if os.name == "nt" else "tmux"


def ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv or argv.index("--") != 2:
        sys.exit(__doc__)
    name, run_dir, cmd = argv[0], Path(argv[1]), argv[3:]
    run_dir = (ROOT / run_dir).resolve() if not run_dir.is_absolute() else run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if subprocess.run([MUX, "has-session", "-t", name], capture_output=True).returncode == 0:
        sys.exit(f"{MUX} session {name!r} already exists; attach with: {MUX} attach -t {name}")
    exit_file = run_dir / "exit_code.txt"
    exit_file.unlink(missing_ok=True)
    (run_dir / "launch.json").write_text(json.dumps(
        {"session": name, "cwd": str(ROOT), "argv": cmd, "launched": time.strftime("%Y-%m-%dT%H:%M:%S")},
        indent=1), encoding="utf-8")
    if MUX == "psmux":
        cmdline = " ".join(ps_quote(c) if i else c for i, c in enumerate(cmd))
        ps = (f"$env:PYTHONUNBUFFERED='1'; $env:PYTHONIOENCODING='utf-8'; & {cmdline}; "
              f"$c=$LASTEXITCODE; Set-Content -Path {ps_quote(str(exit_file))} -Value $c; exit")
        subprocess.run([MUX, "new-session", "-d", "-s", name, "-c", str(ROOT)], check=True)
        time.sleep(1.5)
        subprocess.run([MUX, "send-keys", "-t", name, ps, "Enter"], check=True)
        stop = f"New-Item {run_dir / 'STOP'}"
    else:
        sh = (f"export PYTHONUNBUFFERED=1; {shlex.join(cmd)}; "
              f"echo $? > {shlex.quote(str(exit_file))}")
        subprocess.run([MUX, "new-session", "-d", "-s", name, "-c", str(ROOT), "bash", "-lc", sh], check=True)
        stop = f"touch {shlex.quote(str(run_dir / 'STOP'))}"
    print(f"""session : {name}
attach  : {MUX} attach -t {name}      (detach: Ctrl+B then d)
run dir : {run_dir}
status  : {run_dir / 'status.json'}
events  : {run_dir / 'events.jsonl'}
log     : {run_dir / 'run.log'}
stop    : {stop}   (graceful; or Ctrl+C once in the pane)
resume  : rerun the same command (see launch.json); it continues from the last checkpoint
exit    : {exit_file} (written when the process ends; the session then closes)""")


if __name__ == "__main__":
    main()
