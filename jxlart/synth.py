"""Synthetic tree programs: random grammar sampling and mutation of seed programs.

Distributions are estimated from the seed corpus (gallery trees) so synthetic
data stays near the style of human JXL art while widening coverage.
"""
from __future__ import annotations

import copy
import random
from collections import Counter
from dataclasses import dataclass

from . import dsl
from .dsl import Leaf, Program, Split, Frame

CANVAS_SIDES = [128, 192, 256, 320, 384, 512, 640, 768, 1024]
COMMON_PROPS = ["c", "x", "y", "N", "W", "W-WW-NW+NWW", "W+N-NW", "W-NW", "NW-N", "N-NE",
                "N-NN", "W-WW", "WGH", "|N|", "|W|", "Prev", "PPrev", "PrevErr", "PrevAbs", "g"]


@dataclass
class CorpusStats:
    props: Counter
    preds: Counter
    headers: Counter  # programs containing header
    header_values: dict[str, Counter]
    n_programs: int

    @classmethod
    def from_programs(cls, progs: list[Program]) -> "CorpusStats":
        props, preds, headers = Counter(), Counter(), Counter()
        values: dict[str, Counter] = {}
        for p in progs:
            seen = set()
            for f in p.frames:
                for name, args in f.headers:
                    seen.add(name)
                    values.setdefault(name, Counter())[args] += 1
                for n in dsl.tree_nodes(f.tree):
                    if isinstance(n, Split):
                        props[n.prop] += 1
                    else:
                        preds[n.predictor] += 1
            headers.update(seen)
        for p in COMMON_PROPS:  # keep every common property reachable
            props[p] += 1
        for p in dsl.PREDICTORS:
            preds[p] += 1
        return cls(props, preds, headers, values, len(progs))

    def p_header(self, name: str) -> float:
        return self.headers.get(name, 0) / max(1, self.n_programs)


def _choice(rng: random.Random, counter: Counter):
    keys = list(counter)
    return rng.choices(keys, weights=[counter[k] for k in keys])[0]


class Generator:
    def __init__(self, stats_from: list[Program], rng: random.Random, donors):
        """stats_from: programs whose header/property/predictor frequencies set the sampling priors
        (no program structure is copied). donors: callable returning a Program for crossover."""
        self.stats = CorpusStats.from_programs(stats_from)
        self.rng = rng
        self.donors = donors

    # ------------------------------------------------------------ random programs
    def random_headers(self) -> tuple[list[tuple[str, tuple]], int, int, int]:
        r, st = self.rng, self.stats
        h: list[tuple[str, tuple]] = []
        bitdepth = 8
        if r.random() < max(0.6, st.p_header("Bitdepth")):
            bitdepth = r.choice([1, 2, 3, 4, 5, 6, 8, 8, 8, 10, 10, 12, 16])
            h.append(("Bitdepth", (bitdepth,)))
        up = 1
        if r.random() < st.p_header("Upsample"):
            up = r.choice([2, 4, 8])
        w = h_ = 1024
        if up > 1 or r.random() < st.p_header("Width"):
            w = r.choice([s for s in CANVAS_SIDES if s * up <= 2048])
            h_ = w if r.random() < 0.6 else r.choice([s for s in CANVAS_SIDES if s * up <= 2048])
            h += [("Width", (w,)), ("Height", (h_,))]
        if up > 1:
            h.append(("Upsample", (up,)))
        if r.random() < st.p_header("RCT"):
            vals = st.header_values.get("RCT")
            h.append(("RCT", _choice(r, vals) if vals and r.random() < 0.7 else (r.randrange(0, 42),)))
        for name in ("Squeeze", "DeltaPalette", "Gaborish", "XYB"):
            if r.random() < st.p_header(name):
                h.append((name, ()))
        if r.random() < st.p_header("Orientation"):
            h.append(("Orientation", (r.randint(1, 8),)))
        if r.random() < st.p_header("EPF"):
            h.append(("EPF", (r.randint(1, 3),)))
        return h, bitdepth, w, h_

    def split_value(self, prop: str, bitdepth: int, w: int, h: int) -> int:
        r = self.rng
        if prop == "c":
            return r.choice([0, 0, 1, 1, 2])
        if prop == "g":
            return r.choice([0, 1, 2, 3])
        if prop == "x":
            return r.randrange(0, w)
        if prop == "y":
            return r.randrange(0, h)
        scale = max(2, (1 << min(bitdepth, 16)) // r.choice([2, 4, 8, 16, 64]))
        v = int(round(r.gauss(0, scale / 3)))
        if prop.startswith("|") or prop.endswith("Abs") or prop.endswith("AbsErr") or prop == "WGH":
            v = abs(v)
        return v

    def random_leaf(self, bitdepth: int) -> Leaf:
        r = self.rng
        pred = _choice(r, self.stats.preds)
        hi = 1 << min(bitdepth, 16)
        if pred == "Set":
            off = r.randrange(-hi // 4, hi) if r.random() < 0.9 else r.randrange(-hi, hi)
        else:
            mag = r.choice([1, 1, 2, 3, 5, 8, 16, 32, max(1, hi // 16)])
            off = r.choice([0, 0, 1, -1]) if r.random() < 0.3 else r.randint(-mag, mag)
        return Leaf(pred, off)

    def random_tree(self, bitdepth: int, w: int, h: int, depth: int = 0, max_depth: int = 6) -> dsl.Node:
        r = self.rng
        if depth < max_depth and r.random() < 0.9 * (0.72 ** depth):
            prop = _choice(r, self.stats.props)
            if prop == "c" and depth > 2:
                prop = r.choice(["x", "y", "N", "W", "W-NW", "N-NE"])
            return Split(prop, self.split_value(prop, bitdepth, w, h),
                         self.random_tree(bitdepth, w, h, depth + 1, max_depth),
                         self.random_tree(bitdepth, w, h, depth + 1, max_depth))
        return self.random_leaf(bitdepth)

    def random_program(self) -> Program:
        headers, bd, w, h = self.random_headers()
        return Program([Frame(headers, [], self.random_tree(bd, w, h, max_depth=self.rng.randint(2, 7)))])

    # ------------------------------------------------------------ mutations
    @staticmethod
    def _frame_dims(f: Frame) -> tuple[int, int, int]:
        bd = dsl.header_value(f, "Bitdepth", (8,))[0]
        w = dsl.header_value(f, "Width", (1024,))[0]
        h = dsl.header_value(f, "Height", (1024,))[0]
        return bd, w, h

    def _nodes(self, f: Frame) -> list[dsl.Node]:
        return list(dsl.tree_nodes(f.tree))

    def _replace_subtree(self, f: Frame, target: dsl.Node, new: dsl.Node) -> None:
        if f.tree is target:
            f.tree = new
            return
        for n in dsl.tree_nodes(f.tree):
            if isinstance(n, Split):
                if n.left is target:
                    n.left = new
                    return
                if n.right is target:
                    n.right = new
                    return

    def mutate(self, prog: Program, n_ops: int | None = None) -> Program:
        r = self.rng
        p = copy.deepcopy(prog)
        ops = [self.m_jitter_value, self.m_jitter_offset, self.m_swap_pred, self.m_swap_prop,
               self.m_new_subtree, self.m_crossover, self.m_header, self.m_prune, self.m_spline]
        weights = [3, 3, 2, 2, 1.5, 1.5, 2, 1, 0.5]
        for _ in range(n_ops or r.choice([1, 1, 2, 2, 3, 4])):
            r.choices(ops, weights)[0](p)
        return p

    def _frame(self, p: Program) -> Frame:
        return self.rng.choice(p.frames)

    def m_jitter_value(self, p: Program) -> None:
        splits = [n for n in self._nodes(self._frame(p)) if isinstance(n, Split)]
        if splits:
            n = self.rng.choice(splits)
            d = max(1, abs(n.value) // 4)
            n.value += self.rng.randint(-d, d) or self.rng.choice([-1, 1])

    def m_jitter_offset(self, p: Program) -> None:
        leaves = [n for n in self._nodes(self._frame(p)) if isinstance(n, Leaf)]
        if leaves:
            n = self.rng.choice(leaves)
            d = max(1, abs(n.offset) // 3)
            n.offset += self.rng.randint(-d, d) or self.rng.choice([-1, 1])

    def m_swap_pred(self, p: Program) -> None:
        leaves = [n for n in self._nodes(self._frame(p)) if isinstance(n, Leaf)]
        if leaves:
            self.rng.choice(leaves).predictor = _choice(self.rng, self.stats.preds)

    def m_swap_prop(self, p: Program) -> None:
        splits = [n for n in self._nodes(self._frame(p)) if isinstance(n, Split) and n.prop not in ("c", "x", "y")]
        if splits:
            self.rng.choice(splits).prop = self.rng.choice(
                [k for k in self.stats.props if k not in ("c", "x", "y", "g")])

    def m_new_subtree(self, p: Program) -> None:
        f = self._frame(p)
        bd, w, h = self._frame_dims(f)
        self._replace_subtree(f, self.rng.choice(self._nodes(f)),
                              self.random_tree(bd, w, h, depth=2, max_depth=self.rng.randint(3, 5)))

    def m_crossover(self, p: Program) -> None:
        donor = self.donors()
        f = self._frame(p)
        sub = copy.deepcopy(self.rng.choice(self._nodes(self.rng.choice(donor.frames))))
        self._replace_subtree(f, self.rng.choice(self._nodes(f)), sub)

    def m_prune(self, p: Program) -> None:
        f = self._frame(p)
        splits = [n for n in self._nodes(f) if isinstance(n, Split)]
        if splits:
            n = self.rng.choice(splits)
            self._replace_subtree(f, n, self.rng.choice([n.left, n.right]))

    def m_header(self, p: Program) -> None:
        r = self.rng
        f = p.frames[0]
        names = [n for n, _ in f.headers]
        choice = r.choice(["Bitdepth", "RCT", "Upsample", "Squeeze", "Orientation", "DeltaPalette",
                           "Gaborish", "Width", "drop"])
        if choice == "drop":
            removable = [i for i, (n, _) in enumerate(f.headers) if n not in ("NotLast", "Width", "Height")]
            if removable:
                del f.headers[r.choice(removable)]
            return
        args = {
            "Bitdepth": lambda: (r.choice([1, 2, 4, 6, 8, 10, 12, 16]),),
            "RCT": lambda: (r.randrange(0, 42),),
            "Upsample": lambda: (r.choice([1, 2, 4, 8]),),
            "Squeeze": lambda: (), "DeltaPalette": lambda: (), "Gaborish": lambda: (),
            "Orientation": lambda: (r.randint(1, 8),),
            "Width": lambda: (r.choice(CANVAS_SIDES),),
        }[choice]()
        if choice in names:
            idx = names.index(choice)
            if args:
                f.headers[idx] = (choice, args)
            else:
                del f.headers[idx]
        else:
            f.headers.insert(0, (choice, args))
            if choice == "Width" and "Height" not in names:
                f.headers.insert(1, ("Height", (r.choice(CANVAS_SIDES),)))

    def m_spline(self, p: Program) -> None:
        f = self._frame(p)
        if not f.splines:
            return self.m_jitter_offset(p)
        sp = self.rng.choice(f.splines)
        if self.rng.random() < 0.5:
            i = self.rng.randrange(len(sp.points))
            x, y = sp.points[i]
            sp.points[i] = (round(x + self.rng.gauss(0, 20)), round(y + self.rng.gauss(0, 20)))
        else:
            i = self.rng.choice([0, 32, 64, 96])
            sp.coeffs[i] = round(sp.coeffs[i] * self.rng.uniform(0.5, 1.5) + self.rng.gauss(0, 0.5), 3)
