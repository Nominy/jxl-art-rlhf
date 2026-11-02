"""Parser and canonical printer for the `jxl_from_tree` DSL (libjxl v0.12.0).

The grammar mirrors `tools/jxl_from_tree.cc::ParseNode`: header tokens and
splines may appear anywhere before a tree node finishes; a frame ends when its
MA tree is complete; `NotLast` requests another frame. Tokens after the last
frame are ignored by the reference parser and dropped here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PROPERTIES = [
    "c", "g", "y", "x", "|N|", "|W|", "N", "W", "W-WW-NW+NWW", "W+N-NW", "W-NW",
    "NW-N", "N-NE", "N-NN", "W-WW", "WGH", "PrevAbs", "Prev", "PrevAbsErr", "PrevErr",
    "PPrevAbs", "PPrev", "PPrevAbsErr", "PPrevErr",
]
PROPERTY_SET = set(PROPERTIES) | {
    f"Prev{i}{s}" for i in range(1, 20) for s in ("Abs", "", "AbsErr", "Err")
}
PREDICTORS = [
    "Set", "W", "N", "AvgW+N", "Select", "Gradient", "Weighted", "NE", "NW", "WW",
    "AvgW+NW", "AvgN+NW", "AvgN+NE", "AvgAll",
]
PREDICTOR_SET = set(PREDICTORS)

# Header name -> argument kinds ("u" unsigned int, "i" signed int, "f" float, "s" symbol).
HEADER_ARGS: dict[str, tuple[str, ...]] = {
    "Width": ("u",), "Height": ("u",), "Squeeze": (), "GroupShift": ("u",), "XYB": (),
    "CbYCr": (), "HiddenChannel": ("u",), "RCT": ("u",), "Orientation": ("u",),
    "Alpha": (), "Bitdepth": ("u",), "FloatExpBits": ("u",), "FramePos": ("i", "i"),
    "NotLast": (), "Upsample": ("u",), "Upsample_EC": ("u",), "Animation": (),
    "AnimationFPS": ("u", "u"), "Duration": ("u",), "BlendMode": ("s",),
    "SplineQuantizationAdjustment": ("u",), "Gaborish": (), "DeltaPalette": (),
    "EPF": ("u",), "Noise": ("f",) * 8, "XYBFactors": ("f",) * 3, "PQ": (), "HLG": (),
    "Rec2100": (), "P3": (), "16BitBuffers": (),
}
BLEND_MODES = {"kAdd", "kReplace", "kBlend", "kAlphaWeightedAdd", "kMul"}
SPLINE_COEFFS = 32 * 4  # 3 color DCTs + 1 sigma DCT, 32 coefficients each

_INT_RE = re.compile(r"[+-]?\d+")
_UINT_RE = re.compile(r"\+?\d+")
_FLOAT_RE = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")


class ParseError(ValueError):
    pass


@dataclass
class Leaf:
    predictor: str
    offset: int


@dataclass
class Split:
    prop: str
    value: int
    left: "Node"  # taken when property > value
    right: "Node"


Node = Leaf | Split


@dataclass
class Spline:
    coeffs: list[float]
    points: list[tuple[float, float]]


@dataclass
class Frame:
    headers: list[tuple[str, tuple]] = field(default_factory=list)
    splines: list[Spline] = field(default_factory=list)
    tree: Node | None = None


@dataclass
class Program:
    frames: list[Frame]


def _parse_uint(t: str) -> int:
    if not _UINT_RE.fullmatch(t):
        raise ParseError(f"invalid unsigned int {t!r}")
    return int(t)


def _parse_int(t: str) -> int:
    if not _INT_RE.fullmatch(t):
        raise ParseError(f"invalid int {t!r}")
    return int(t)


def _parse_float(t: str) -> float:
    if not _FLOAT_RE.fullmatch(t):
        raise ParseError(f"invalid float {t!r}")
    return float(t)


class _Tokens:
    def __init__(self, text: str):
        self.toks = text.split()
        self.i = 0

    def next(self) -> str:
        if self.i >= len(self.toks):
            raise ParseError("unexpected end of input")
        t = self.toks[self.i]
        self.i += 1
        return t


def _parse_header(name: str, tk: _Tokens) -> tuple:
    args = []
    for kind in HEADER_ARGS[name]:
        t = tk.next()
        if kind == "u":
            args.append(_parse_uint(t))
        elif kind == "i":
            args.append(_parse_int(t))
        elif kind == "f":
            args.append(_parse_float(t))
        else:
            if t not in BLEND_MODES:
                raise ParseError(f"invalid BlendMode {t!r}")
            args.append(t)
    _check_header(name, args)
    return tuple(args)


def _check_header(name: str, args: list) -> None:
    if name == "Bitdepth" and not 1 <= args[0] <= 32:
        raise ParseError("Bitdepth out of range")
    if name in ("Upsample", "Upsample_EC") and args[0] not in (1, 2, 4, 8):
        raise ParseError(f"invalid {name}")
    if name == "EPF" and args[0] > 3:
        raise ParseError("invalid EPF")
    if name == "Noise" and any(not 0.0 <= v <= 1.0 for v in args):
        raise ParseError("invalid Noise")
    if name == "HiddenChannel" and args[0] > 16:
        raise ParseError("invalid HiddenChannel")


def _parse_node(tk: _Tokens, frame: Frame, depth: int = 0) -> Node:
    while True:
        t = tk.next()
        if t == "if":
            prop = tk.next()
            if prop not in PROPERTY_SET:
                raise ParseError(f"unexpected property {prop!r}")
            if tk.next() != ">":
                raise ParseError("expected '>'")
            value = _parse_int(tk.next())
            if depth > 64:
                raise ParseError("tree too deep")
            left = _parse_node(tk, frame, depth + 1)
            right = _parse_node(tk, frame, depth + 1)
            return Split(prop, value, left, right)
        if t == "-":
            pred = tk.next()
            if pred not in PREDICTOR_SET:
                raise ParseError(f"unexpected predictor {pred!r}")
            o = tk.next()
            sign = 1
            if o in ("-", "+"):
                sign = -1 if o == "-" else 1
                o = tk.next()
            return Leaf(pred, sign * _parse_int(o))
        if t == "/*":
            while tk.i < len(tk.toks) and tk.next() != "*/":
                pass
            continue
        if t == "Spline":
            coeffs = [_parse_float(tk.next()) for _ in range(SPLINE_COEFFS)]
            points = []
            while True:
                x = tk.next()
                if x == "EndSpline":
                    break
                points.append((_parse_float(x), _parse_float(tk.next())))
            if not points:
                raise ParseError("spline without control points")
            frame.splines.append(Spline(coeffs, points))
            continue
        if t in HEADER_ARGS:
            frame.headers.append((t, _parse_header(t, tk)))
            continue
        raise ParseError(f"unexpected node type {t!r}")


def parse(text: str) -> Program:
    tk = _Tokens(text)
    frames = []
    while True:
        frame = Frame()
        frame.tree = _parse_node(tk, frame)
        frames.append(frame)
        if not any(name == "NotLast" for name, _ in frame.headers):
            return Program(frames)


def _fmt_float(v: float) -> str:
    s = format(v, ".7g")
    return "0" if s in ("-0", "0") else s


def _fmt_args(args: tuple) -> str:
    out = []
    for a in args:
        out.append(_fmt_float(a) if isinstance(a, float) else str(a))
    return " ".join(out)


def _fmt_tree(node: Node, depth: int, out: list[str]) -> None:
    pad = "  " * depth
    if isinstance(node, Leaf):
        out.append(f"{pad}- {node.predictor} {node.offset}")
        return
    out.append(f"{pad}if {node.prop} > {node.value}")
    _fmt_tree(node.left, depth + 1, out)
    _fmt_tree(node.right, depth + 1, out)


def to_text(prog: Program) -> str:
    out: list[str] = []
    for fi, frame in enumerate(prog.frames):
        if fi:
            out.append("")
        for name, args in frame.headers:
            out.append(f"{name} {_fmt_args(args)}".rstrip())
        for sp in frame.splines:
            out.append("Spline")
            for row in range(4):
                out.append(" ".join(_fmt_float(c) for c in sp.coeffs[row * 32:(row + 1) * 32]))
            out.extend(f"{_fmt_float(x)} {_fmt_float(y)}" for x, y in sp.points)
            out.append("EndSpline")
        assert frame.tree is not None
        _fmt_tree(frame.tree, 0, out)
    return "\n".join(out) + "\n"


def canonicalize(text: str) -> str:
    return to_text(parse(text))


def header_value(frame: Frame, name: str, default=None):
    """Last occurrence wins, matching the reference parser's assignment semantics."""
    val = default
    for n, args in frame.headers:
        if n == name:
            val = args
    return val


def output_size(prog: Program) -> tuple[int, int]:
    """Decoded image size, as computed by JxlFromTree from the first frame."""
    f0 = prog.frames[0]
    w = header_value(f0, "Width", (1024,))[0]
    h = header_value(f0, "Height", (1024,))[0]
    x0, y0 = header_value(f0, "FramePos", (0, 0))
    up = header_value(f0, "Upsample", (1,))[0]
    return (w + x0) * up, (h + y0) * up


def tree_nodes(node: Node):
    yield node
    if isinstance(node, Split):
        yield from tree_nodes(node.left)
        yield from tree_nodes(node.right)
