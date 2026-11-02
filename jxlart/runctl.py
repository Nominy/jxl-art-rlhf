"""Durable run control: status.json, events.jsonl, log, graceful stop, live terminal view.

Every long job owns a run directory:
  status.json   atomically rewritten snapshot (state, phase, nested progress, ETA, heartbeat)
  events.jsonl  append-only machine-readable events (start/checkpoint/error/stop/complete)
  run.log       timestamped human log
  STOP          create this file (or press Ctrl+C once) to request a graceful stop
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table

STATES = ("initializing", "running", "waiting", "stopping", "stopped", "failed", "completed")


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _fmt_dur(s: float | None) -> str:
    if s is None or s != s or s == float("inf"):
        return "?"
    s = int(s)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


@dataclass
class Level:
    name: str
    total: int | None
    done: int = 0
    unit: str = "items"
    started: float = field(default_factory=time.time)
    base_done: int = 0  # units already complete when this process started (resume)

    def snapshot(self) -> dict:
        el = time.time() - self.started
        new = self.done - self.base_done
        rate = new / el if el > 0 and new > 0 else None
        eta = (self.total - self.done) / rate if rate and self.total is not None else None
        return {"name": self.name, "done": self.done, "total": self.total, "unit": self.unit,
                "rate_per_s": rate, "eta_s": eta, "elapsed_s": el}


class Run:
    def __init__(self, run_dir: str | Path, kind: str, config: dict, live: bool = True):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.kind = kind
        self.config = config
        self.started = time.time()
        self.state = "initializing"
        self.phase = ""
        self.levels: dict[str, Level] = {}
        self.metrics: dict = {}
        self.counters: dict[str, int] = {}
        self.last_event = ""
        self.last_activity = time.time()
        self.stop_requested = False
        self._lock = threading.RLock()
        self._last_status = 0.0
        self._log = open(self.dir / "run.log", "a", encoding="utf-8")
        self._events = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self.console = Console()
        self._live = Live(self._render(), console=self.console, refresh_per_second=1, transient=False) if live else None
        stop_file = self.dir / "STOP"
        if stop_file.exists():
            stop_file.unlink()
        signal.signal(signal.SIGINT, self._on_signal)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, self._on_signal)
        self.event("start", kind=kind, pid=os.getpid(), argv=sys.argv, config=config)

    # ---------------------------------------------------------------- control
    def _on_signal(self, signum, frame):
        if self.stop_requested:
            raise KeyboardInterrupt
        self.request_stop(f"signal {signum}")

    def request_stop(self, reason: str) -> None:
        if not self.stop_requested:
            self.stop_requested = True
            self.set_state("stopping")
            self.event("stop_requested", reason=reason)

    def should_stop(self) -> bool:
        if not self.stop_requested and (self.dir / "STOP").exists():
            self.request_stop("STOP file")
        return self.stop_requested

    # ---------------------------------------------------------------- reporting
    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        with self._lock:
            self._log.write(line + "\n")
            self._log.flush()
        self.last_event = msg
        self.last_activity = time.time()
        if self._live:
            self._live.console.print(line, markup=False, highlight=False)
        else:
            print(line, flush=True)
        self.status()

    def event(self, name: str, /, **data) -> None:
        rec = {"ts": time.time(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": name, **data}
        with self._lock:
            self._events.write(json.dumps(rec, default=str) + "\n")
            self._events.flush()
        self.last_activity = time.time()
        if name != "start":
            short = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
            self.log(f"[{name}] " + " ".join(f"{k}={v}" for k, v in short.items()))
        else:
            self.status(force=True)

    def set_state(self, state: str, phase: str | None = None) -> None:
        assert state in STATES, state
        self.state = state
        if phase is not None:
            self.phase = phase
        self.status(force=True)

    def level(self, name: str, total: int | None, unit: str = "items", done: int = 0) -> Level:
        lv = Level(name, total, done, unit, base_done=done)
        self.levels[name] = lv
        self.status(force=True)
        return lv

    def drop_level(self, name: str) -> None:
        self.levels.pop(name, None)

    def advance(self, name: str, n: int = 1) -> None:
        self.levels[name].done += n
        self.last_activity = time.time()
        self.status()

    def count(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def metric(self, **kw) -> None:
        self.metrics.update(kw)
        self.last_activity = time.time()
        self.status()

    def snapshot(self) -> dict:
        return {
            "kind": self.kind, "state": self.state, "phase": self.phase, "pid": os.getpid(),
            "run_dir": str(self.dir.resolve()), "started": self.started,
            "elapsed_s": time.time() - self.started, "heartbeat": time.time(),
            "heartbeat_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds_since_activity": time.time() - self.last_activity,
            "stop_requested": self.stop_requested,
            "levels": [lv.snapshot() for lv in self.levels.values()],
            "counters": self.counters, "metrics": self.metrics, "last_event": self.last_event,
        }

    def status(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_status < 1.0:
            return
        self._last_status = now
        with self._lock:
            atomic_write_json(self.dir / "status.json", self.snapshot())
        if self._live:
            self._live.update(self._render())

    def _render(self) -> Table:
        t = Table(title=f"{self.kind} — {self.dir}", expand=False)
        t.add_column("level")
        t.add_column("progress", justify="right")
        t.add_column("rate", justify="right")
        t.add_column("ETA", justify="right")
        el = time.time() - self.started
        t.add_row("state", f"{self.state} {self.phase}", f"elapsed {_fmt_dur(el)}",
                  f"idle {_fmt_dur(time.time() - self.last_activity)}")
        for lv in getattr(self, "levels", {}).values():
            s = lv.snapshot()
            tot = "?" if s["total"] is None else s["total"]
            pct = f" {100 * s['done'] / s['total']:.1f}%" if s["total"] else ""
            rate = f"{s['rate_per_s']:.2f} {lv.unit}/s" if s["rate_per_s"] else "-"
            t.add_row(lv.name, f"{s['done']}/{tot}{pct}", rate, _fmt_dur(s["eta_s"]))
        if getattr(self, "counters", None):
            t.add_row("counters", " ".join(f"{k}={v}" for k, v in self.counters.items()), "", "")
        for k, v in list(getattr(self, "metrics", {}).items())[:24]:
            t.add_row(k, f"{v:.5g}" if isinstance(v, float) else str(v), "", "")
        t.add_row("stop", f"create {self.dir / 'STOP'} or Ctrl+C once", "", "")
        return t

    # ---------------------------------------------------------------- lifecycle
    def _heartbeat(self) -> None:
        """Refresh status.json every 5 s even while the main thread is busy (stalls stay visible)."""
        while not self._hb_stop.wait(5.0):
            self.status(force=True)

    def __enter__(self):
        if self._live:
            self._live.start()
        self.set_state("running")
        self._hb_stop = threading.Event()
        threading.Thread(target=self._heartbeat, daemon=True).start()
        return self

    def __exit__(self, et, ev, tb):
        self._hb_stop.set()
        if et is None:
            final = "stopped" if self.stop_requested else "completed"
            self.set_state(final)
            self.event(final, elapsed_s=round(time.time() - self.started, 1))
        else:
            self.set_state("failed")
            self.event("failed", error=f"{et.__name__}: {ev}")
        if self._live:
            self._live.stop()
        self._log.close()
        self._events.close()
        return False
