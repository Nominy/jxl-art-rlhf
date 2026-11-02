"""Download the jpegxl.info JXL art gallery: every .jxl plus its tree "source code" when published.

  python -m jxlart.fetch_gallery            # -> art/manifest.json, art/<section>/NNN_title.jxl|.tree.txt

The art belongs to its authors (credited in each manifest entry); it is fetched locally, never
committed. It is used only as a reference for the copy penalty and for sampling priors.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html as H
import json
import re
import urllib.request
import zlib
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

from .compiler import ROOT

ART_DIR = ROOT / "art"
SITE = "https://jpegxl.info"
PAGES = ["/art/", "/art/big_gallery.html", "/art/hdr.html", "/art/2021-04_jon.html"]
TOKEN = re.compile(r'<article class="art-feature">.*?</article>|<h2[^>]*>(.*?)</h2>|<h3[^>]*>(.*?)</h3>'
                   r'|<p[^>]*>(.*?)</p>|<img([^>]*)src="(data:image/jxl;base64,[^"]+|/images/art/[^"]+\.jxl)"([^>]*)>',
                   re.S)


def _get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()


def _strip(x: str) -> str:
    return H.unescape(re.sub(r"<[^>]+>", "", x)).strip()


def decode_zcode(z: str) -> str | None:
    """Tree text from a jxl-art editor `zcode` parameter (raw deflate or zlib, url-safe base64)."""
    z = unquote(H.unescape(z))
    raw = base64.b64decode(z.replace("-", "+").replace("_", "/") + "=" * (-len(z) % 4))
    for wbits in (-15, 15, 31):
        try:
            return zlib.decompress(raw, wbits).decode()
        except zlib.error:
            pass
    return None


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:60] or "untitled"


def _folder(page: str, section: str | None) -> str:
    if page == "index":
        return "compos" if section == "JXL Art Compos" else "featured"
    return {"hdr.html": "hdr", "2021-04_jon.html": "early_jon_2021"}.get(page, f"gallery/{section}")


def scrape() -> list[dict]:
    pages = {p: _get(SITE + p).decode() for p in PAGES}
    index_html = pages["/art/"]
    # the index page links the "Tropical Island Sunset with JXL logo overlay" source in prose, not its caption
    m = re.search(r'zcode=([^"&\s]+)">\(&ldquo;source code', index_html)
    tropical = m.group(1) if m else None
    items = []
    for p, pg in pages.items():
        page = p.rsplit("/", 1)[-1] or "index"
        sec = title = meta = src = place = None
        for t in TOKEN.finditer(pg):
            h2, h3, pa, a1, img, a2 = t.groups()
            if t.group(0).startswith("<article"):
                continue
            if h2:
                sec, title = _strip(h2), None
            elif h3:
                title, meta, src = _strip(h3), None, None
            elif pa is not None:
                pm = re.search(r"(First|Second|Third) place", pa)
                if pm:
                    place = pm.group(1)
                if re.search(r"\d+ bytes", pa):
                    meta = _strip(pa)
                    zz = re.search(r'zcode=([^"&\s]+)', pa)
                    src = zz.group(1) if zz else None
            elif img:
                alt = re.search(r'alt="([^"]*)"', (a1 or "") + (a2 or ""))
                alt = H.unescape(alt.group(1)) if alt else ""
                same = not alt or alt == title
                name = title if same else alt
                z = src if same else None
                if name == "Tropical Island Sunset with JXL logo overlay" and not z:
                    z = tropical
                items.append({"page": page, "section": sec, "title": name or alt, "caption": meta if same else None,
                              "place": place, "img": img, "zcode": z})
                place = None
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    items = scrape()
    cache: dict[str, bytes] = {}
    counters: Counter = Counter()
    manifest = []
    for it in items:
        if it["img"].startswith("data:"):
            data = base64.b64decode(it["img"].split(",", 1)[1])
        else:
            url = SITE + it["img"]
            data = cache.setdefault(url, _get(url))
        folder = _folder(it["page"], it["section"])
        counters[folder] += 1
        base = f"{folder}/{counters[folder]:03d}_" + (f"{it['place']}_" if it["place"] else "") + _slug(it["title"])
        (out / folder).mkdir(parents=True, exist_ok=True)
        (out / f"{base}.jxl").write_bytes(data)
        tree = decode_zcode(it["zcode"]) if it["zcode"] else None
        if tree:
            (out / f"{base}.tree.txt").write_text(tree, encoding="utf-8")
        manifest.append({
            "file": f"{base}.jxl", "tree": f"{base}.tree.txt" if tree else None, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "title": it["title"], "caption": it["caption"],
            "place": it["place"], "section": it["section"],
            "page": f"{SITE}/art/" + ("" if it["page"] == "index" else it["page"]),
            "source_url": f"https://jxl-art.lucaversari.it/?zcode={it['zcode']}" if it["zcode"] else None})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(manifest)} pieces, {sum(1 for m in manifest if m['tree'])} with source, "
          f"{sum(m['bytes'] for m in manifest)} bytes -> {out}")


if __name__ == "__main__":
    main()
