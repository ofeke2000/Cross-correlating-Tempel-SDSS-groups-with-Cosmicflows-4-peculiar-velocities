#!/usr/bin/env python3
"""Build the roadmap page: ``docs/real_data_roadmap.md`` -> ``roadmap_site/roadmap.html``.

The Markdown file is the only source of truth. This script never edits it, and never
writes inside ``roadmap_assets/`` except ``roadmap_assets/vendor/`` (and only with
``--fetch-vendor``). Rerunning it is safe and deterministic: the same Markdown and the
same assets give a byte-identical page (no timestamps, no randomness).

Usage, inside the project venv::

    python roadmap_site/build_roadmap.py                 # build the page
    python roadmap_site/build_roadmap.py --check         # validate only, write nothing
    python roadmap_site/build_roadmap.py --fetch-vendor  # once per machine: KaTeX + fonts

The page is self-contained apart from ``roadmap_assets/``: the plots it shows, and
``vendor/`` (KaTeX, Literata, IBM Plex Mono). It opens straight from disk, no server,
and works offline once the vendor files are fetched. Without them it still opens, with
system fonts and raw ``$...$`` in place of rendered math.

Stage metadata
--------------
A heading becomes a timeline stage when the first non-blank line after it opens a
``<!-- roadmap`` comment (invisible in any Markdown preview). Inside it is YAML::

    ## Stage 2: Build the two analysis samples

    <!-- roadmap
    current: true                 # optional; exactly one stage carries it
    missions: |
      [x] Tempel collapsed to one row per group
      [ ] Phase B per-source velocity samples
    plots:                        # optional; files live in roadmap_site/roadmap_assets/
      - file: stage02_sky_samples.png
        caption: "CF4 centers inside the Tempel footprint"
        notes: >
          Two or three sentences. Math such as $\\zeta_1 > 0$ is fine.
    -->

- ``missions`` is a ``|`` block, one ``[x]`` (done) or ``[ ]`` (to do) line per mission;
  an indented line continues the mission above it. At least one mission is required.
- A stage is *done* when every mission is ``[x]``, *in progress* when some are, and
  *planned* when none are. ``current`` is independent of that: it marks where work is now.
- ``plots`` entries need ``file`` (a path relative to ``roadmap_assets/``) and
  ``caption``; ``notes`` is optional. Quote a caption that contains ``": "``.

Everything else comes from the stage's own text, so nothing is written twice:

- the short explanation is the paragraph opening with ``**What this ...**``;
- the exit test is the paragraph opening with ``**Done when:**``;
- the prompt is the fenced block after the ``**Prompt to start it:**`` line;
- the rest is the long explanation (subsections such as a ``###`` inside a stage
  stay part of it).

A stage ends at the next stage heading or the next ``##`` heading. ``##`` sections
before the first stage form the page introduction; those after the last stage form
the appendix. Math is ``$...$`` inline and ``$$...$$`` displayed, anywhere outside code.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import os
import re
import sys
import tarfile
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

import markdown
import yaml

SITE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SITE_DIR.parent
SOURCE = REPO_ROOT / "docs" / "real_data_roadmap.md"
OUTPUT = SITE_DIR / "roadmap.html"
ASSETS_DIRNAME = "roadmap_assets"
ASSETS = SITE_DIR / ASSETS_DIRNAME
VENDOR = ASSETS / "vendor"

META_OPEN = "<!-- roadmap"
META_CLOSE = "-->"
META_KEYS = frozenset({"current", "missions", "plots"})
PLOT_KEYS = frozenset({"file", "caption", "notes"})

SHORT_LEAD = re.compile(r"^\*\*What this[^*]*\*\*\s*")
DONE_LEAD = re.compile(r"^\*\*Done when:?\*\*:?\s*")
PROMPT_LEAD = re.compile(r"^\*\*Prompt to start it:?\*\*:?\s*$")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
MISSION = re.compile(r"^\[([ xX])\]\s+(.*)$")
STAGE_HEADING = re.compile(r"^(?P<label>.+?)(?:\s*\[(?P<added>\d{4}-\d{2}-\d{2})\])?\s*:\s+(?P<title>.+)$")
STAGE_CODE = re.compile(r"^Stage\s+(\w+)")
PROMPT_PREVIEW_LINES = 4  # a prompt with more lines than this, or more characters than
PROMPT_PREVIEW_CHARS = 320  # this (after reflowing), is folded until expanded
PROMPT_KEEP_BREAK = re.compile(r"^\s*([-*+]\s|\d+[.)]\s|$)")


# ---------------------------------------------------------------------------------------
# Vendored front-end files (fetched once with --fetch-vendor, pinned by npm's sha512)
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class VendorPackage:
    """One npm tarball and the files kept from it.

    ``files`` maps a glob inside the tarball to a directory under ``vendor/``.
    ``font_css`` names fontsource CSS files whose ``@font-face`` blocks are kept for
    the font files that were copied; they are merged into ``vendor/fonts/fonts.css``.
    """

    url: str
    integrity: str
    files: tuple[tuple[str, str], ...]
    font_css: tuple[str, ...] = ()


VENDOR_PACKAGES = (
    VendorPackage(
        url="https://registry.npmjs.org/katex/-/katex-0.19.0.tgz",
        integrity="sha512-v6Tznz3zJ7u3niRCoDTsumM2+HA2XXcCu+WAacCeHD2z3p9A9Ks987o5FzfTGBN0e8A0vjEgIDvLbICrXpdw/Q==",
        files=(
            ("package/dist/katex.min.css", "katex"),
            ("package/dist/katex.min.js", "katex"),
            ("package/dist/contrib/auto-render.min.js", "katex"),
            ("package/dist/fonts/*.woff2", "katex/fonts"),
        ),
    ),
    VendorPackage(
        url="https://registry.npmjs.org/@fontsource-variable/literata/-/literata-5.3.0.tgz",
        integrity="sha512-1Jw7xvOcoGyB83lxB2nG4bfNbqtX2/TTbdh9sHAOmU064SvmSdulDtA5RBu5JYWf64fdaKX7xU+z3GO1T2bCHA==",
        files=tuple(
            (f"package/files/literata-{subset}-opsz-*.woff2", "fonts")
            for subset in ("latin", "latin-ext", "greek")
        ),
        font_css=("package/opsz.css", "package/opsz-italic.css"),
    ),
    VendorPackage(
        url="https://registry.npmjs.org/@fontsource/ibm-plex-mono/-/ibm-plex-mono-5.3.0.tgz",
        integrity="sha512-eTgnZjZEGk1QtD3ZstF+Vclo2HLAni8YMy34/DxllwZvyz1lR/1RF/xTiAquOBO7MvqBx8D2Ig2WCPMVfdZu7Q==",
        files=tuple(
            (f"package/files/ibm-plex-mono-{subset}-{weight}-normal.woff2", "fonts")
            for subset in ("latin", "latin-ext")
            for weight in (400, 500)
        ),
        font_css=("package/400.css", "package/500.css"),
    ),
)
VENDOR_REQUIRED = (
    "katex/katex.min.css",
    "katex/katex.min.js",
    "katex/auto-render.min.js",
    "fonts/fonts.css",
)
FONT_FACE_BLOCK = re.compile(r"/\*\s*([\w-]+)\s*\*/\s*(@font-face\s*\{[^}]*\})")
FONT_SRC = re.compile(r"src:\s*url\(\./files/([^)]+\.woff2)\)\s*format\('([^']+)'\)[^;]*;")


def fetch_vendor() -> None:
    """Download the pinned packages, check their sha512, and unpack the kept files."""
    font_faces: list[str] = []
    for package in VENDOR_PACKAGES:
        print(f"fetching {package.url}")
        with urllib.request.urlopen(package.url, timeout=120) as response:
            data = response.read()
        digest = "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()
        if digest != package.integrity:
            raise SystemExit(f"integrity check failed for {package.url}; nothing unpacked from it")
        copied: set[str] = set()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            for pattern, destination in package.files:
                matches = sorted(name for name in members if fnmatch(name, pattern))
                if not matches:
                    raise SystemExit(f"{package.url}: no file matches {pattern}")
                for name in matches:
                    target = VENDOR / destination / PurePosixPath(name).name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(tar.extractfile(members[name]).read())
                    copied.add(target.name)
            for css_name in package.font_css:
                css = tar.extractfile(members[css_name]).read().decode()
                for name, block in FONT_FACE_BLOCK.findall(css):
                    if f"{name}.woff2" in copied:
                        font_faces.append(FONT_SRC.sub(r"src: url(\1) format('\2');", block))
    fonts_css = VENDOR / "fonts" / "fonts.css"
    fonts_css.write_text("\n\n".join(font_faces) + "\n", encoding="utf-8")
    print(f"vendor files are in {VENDOR.relative_to(REPO_ROOT)}")


def missing_vendor_files() -> list[str]:
    return [name for name in VENDOR_REQUIRED if not (VENDOR / name).is_file()]


# ---------------------------------------------------------------------------------------
# Parsing the Markdown
# ---------------------------------------------------------------------------------------

@dataclass
class Mission:
    done: bool
    text: str


@dataclass
class Plot:
    file: str
    caption: str
    notes: str


@dataclass
class Section:
    heading: str
    body: str


@dataclass
class Stage:
    heading: str
    line: int
    label: str
    code: str
    title: str
    target: str
    added: str
    current: bool
    missions: list[Mission]
    plots: list[Plot]
    short: str
    done_when: str
    prompt: str
    long: str

    @property
    def n_done(self) -> int:
        return sum(m.done for m in self.missions)

    @property
    def state(self) -> str:
        if self.n_done == len(self.missions):
            return "done"
        return "started" if self.n_done else "planned"

    @property
    def anchor(self) -> str:
        return slugify(self.label or self.title)


@dataclass
class Roadmap:
    title: str
    intro: list[Section] = field(default_factory=list)
    items: list[Stage | Section] = field(default_factory=list)
    appendix: list[Section] = field(default_factory=list)

    @property
    def stages(self) -> list[Stage]:
        return [item for item in self.items if isinstance(item, Stage)]


class BuildErrors(Exception):
    def __init__(self, messages: list[str]):
        super().__init__("\n".join(messages))
        self.messages = messages


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def find_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """(line index, level, text) of every ATX heading outside fenced code."""
    headings, in_fence = [], False
    for i, line in enumerate(lines):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        match = None if in_fence else HEADING.match(line)
        if match:
            headings.append((i, len(match.group(1)), match.group(2)))
    return headings


def blocks_of(lines: list[str]) -> list[tuple[int, int]]:
    """Half-open line ranges of the paragraph-level blocks; a fenced block is one block."""
    blocks, i, n = [], 0, len(lines)
    while i < n:
        if not lines[i].strip():
            i += 1
            continue
        start = i
        if FENCE.match(lines[i]):
            i += 1
            while i < n and not FENCE.match(lines[i]):
                i += 1
            i = min(i + 1, n)
        else:
            while i < n and lines[i].strip() and not FENCE.match(lines[i]):
                i += 1
        blocks.append((start, i))
    return blocks


def trim(lines: list[str]) -> list[str]:
    """Drop leading/trailing blank lines and horizontal rules."""
    def filler(line: str) -> bool:
        return not line.strip() or re.fullmatch(r"\s*(-{3,}|\*{3,}|_{3,})\s*", line) is not None

    start, end = 0, len(lines)
    while start < end and filler(lines[start]):
        start += 1
    while end > start and filler(lines[end - 1]):
        end -= 1
    return lines[start:end]


def parse_missions(raw: object, where: str, errors: list[str]) -> list[Mission]:
    if not isinstance(raw, str):
        errors.append(f"{where}: 'missions' must be a '|' block of '[x] ...' / '[ ] ...' lines")
        return []
    missions: list[Mission] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        match = MISSION.match(line.strip())
        if match and not line[:1].isspace():
            missions.append(Mission(done=match.group(1) != " ", text=match.group(2).strip()))
        elif missions and line[:1].isspace():
            missions[-1].text += " " + line.strip()
        else:
            errors.append(f"{where}: mission line must start with '[x] ' or '[ ] ': {line.strip()!r}")
    if not missions:
        errors.append(f"{where}: a stage needs at least one mission")
    return missions


def parse_plots(raw: object, where: str, errors: list[str]) -> list[Plot]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        errors.append(f"{where}: 'plots' must be a list")
        return []
    plots = []
    for k, entry in enumerate(raw, 1):
        if not isinstance(entry, dict):
            errors.append(f"{where}: plot {k} must be a mapping with file/caption/notes")
            continue
        unknown = sorted(set(entry) - PLOT_KEYS)
        if unknown:
            errors.append(f"{where}: plot {k} has unknown keys {unknown} (allowed: {sorted(PLOT_KEYS)})")
        file, caption = entry.get("file"), entry.get("caption")
        if not file or not caption:
            errors.append(f"{where}: plot {k} needs both 'file' and 'caption'")
            continue
        path = PurePosixPath(str(file))
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"{where}: plot file {file!r} must be a relative path inside {ASSETS_DIRNAME}/")
        elif not (ASSETS / path).is_file():
            errors.append(f"{where}: plot file {ASSETS_DIRNAME}/{file} does not exist")
        plots.append(Plot(file=str(path), caption=str(caption).strip(), notes=str(entry.get("notes") or "").strip()))
    return plots


def split_heading(text: str) -> tuple[str, str, str, str]:
    """'Stage 4: Selection function  -> `src/...`' -> (label, added, title, target)."""
    main, _, target = text.partition("→")
    match = STAGE_HEADING.match(main.strip())
    if not match:
        return "", "", main.strip(), target.strip()
    title = match.group("title").strip()
    return match.group("label").strip(), match.group("added") or "", title[:1].upper() + title[1:], target.strip()


def stage_code(label: str, fallback: str) -> str:
    match = STAGE_CODE.match(label)
    if not match:
        return fallback
    return match.group(1).upper() + ("A" if "addendum" in label.lower() else "")


def parse_stage(heading: str, line_no: int, meta_text: str, body: list[str], errors: list[str]) -> Stage:
    where = f"line {line_no} ({heading[:48]})"
    try:
        meta = yaml.safe_load(meta_text) or {}
    except yaml.YAMLError as exc:
        errors.append(f"{where}: metadata is not valid YAML: {exc}")
        meta = {}
    if not isinstance(meta, dict):
        errors.append(f"{where}: metadata must be a YAML mapping")
        meta = {}
    unknown = sorted(set(meta) - META_KEYS)
    if unknown:
        errors.append(f"{where}: unknown metadata keys {unknown} (allowed: {sorted(META_KEYS)})")
    current = meta.get("current", False)
    if not isinstance(current, bool):
        errors.append(f"{where}: 'current' must be true or false")
        current = False

    taken: set[int] = set()
    short = done_when = prompt = ""
    blocks = blocks_of(body)
    for k, (start, end) in enumerate(blocks):
        text = "\n".join(body[start:end])
        if not short and SHORT_LEAD.match(text):
            short, taken = SHORT_LEAD.sub("", text), taken | set(range(start, end))
        elif not done_when and DONE_LEAD.match(text):
            done_when, taken = DONE_LEAD.sub("", text), taken | set(range(start, end))
        elif not prompt and PROMPT_LEAD.match(text):
            nxt = blocks[k + 1] if k + 1 < len(blocks) else None
            if nxt is None or not FENCE.match(body[nxt[0]]):
                errors.append(f"{where}: 'Prompt to start it:' must be followed by a fenced code block")
                continue
            prompt = "\n".join(body[nxt[0] + 1 : nxt[1] - 1])
            taken |= set(range(start, nxt[1]))
    if not short:
        errors.append(f"{where}: no paragraph opening with **What this ...** (the short explanation)")
    long_md = "\n".join(trim([line for i, line in enumerate(body) if i not in taken]))
    long_md = re.sub(r"\n{3,}", "\n\n", long_md)

    label, added, title, target = split_heading(heading)
    return Stage(
        heading=heading,
        line=line_no,
        label=label,
        code=stage_code(label, fallback=str(line_no)),
        title=title,
        target=target,
        added=added,
        current=current,
        missions=parse_missions(meta.get("missions"), where, errors),
        plots=parse_plots(meta.get("plots"), where, errors),
        short=short,
        done_when=done_when,
        prompt=prompt,
        long=long_md,
    )


def parse(text: str) -> Roadmap:
    lines = text.splitlines()
    errors: list[str] = []
    headings = find_headings(lines)
    title = next((t for _, level, t in headings if level == 1), None)
    if title is None:
        raise BuildErrors(["the roadmap needs a '# ' title heading"])

    meta_of: dict[int, tuple[int, int]] = {}  # heading line -> (meta open line, meta close line)
    for i, _, heading in headings:
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines) and lines[j].strip() == META_OPEN:
            k = next((m for m in range(j + 1, len(lines)) if lines[m].strip() == META_CLOSE), None)
            if k is None:
                errors.append(f"line {i + 1} ({heading[:48]}): '{META_OPEN}' is never closed with '{META_CLOSE}'")
                continue
            meta_of[i] = (j, k)

    # A boundary heading starts a new page item: every stage heading, and every '#'/'##'.
    boundaries = [(i, level, t) for i, level, t in headings if i in meta_of or level <= 2]
    roadmap = Roadmap(title=title)
    seen_stage = False
    trailing: list[Section] = []
    for n, (i, level, heading) in enumerate(boundaries):
        end = boundaries[n + 1][0] if n + 1 < len(boundaries) else len(lines)
        if i in meta_of:
            j, k = meta_of[i]
            meta_text = "\n".join(lines[j + 1 : k])
            roadmap.items.extend(trailing)  # '##' sections between two stages stay in place
            trailing = []
            roadmap.items.append(parse_stage(heading, i + 1, meta_text, lines[k + 1 : end], errors))
            seen_stage = True
            continue
        body = "\n".join(trim(lines[i + 1 : end]))
        if level == 1:
            if body:
                roadmap.intro.append(Section(heading="", body=body))
            continue
        section = Section(heading=heading, body=body)
        (trailing if seen_stage else roadmap.intro).append(section)
    roadmap.appendix = trailing

    stages = roadmap.stages
    if not stages:
        errors.append(f"no stages found: put a '{META_OPEN}' block under each stage heading")
    current = [s for s in stages if s.current]
    if len(current) > 1:
        errors.append("more than one stage has 'current: true': " + ", ".join(s.label or s.title for s in current))
    codes = [s.anchor for s in stages]
    duplicates = sorted({c for c in codes if codes.count(c) > 1})
    if duplicates:
        errors.append(f"two stages share a heading label: {duplicates}")
    if errors:
        raise BuildErrors(errors)
    return roadmap


# ---------------------------------------------------------------------------------------
# Rendering Markdown fragments
# ---------------------------------------------------------------------------------------

_MD = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"], output_format="html")
_CODE_OR_FENCE = re.compile(r"(^(```|~~~).*?^\2[^\n]*$|(`+).+?\3)", re.S | re.M)
_DISPLAY_MATH = re.compile(r"\$\$.+?\$\$", re.S)
_INLINE_MATH = re.compile(r"(?<![\\$])\$(?=\S)[^$\n]+?(?<=\S)\$(?!\d)")
_TOKEN = "KTXMATH{:04d}XTK"
_HREF = re.compile(r'href="([^"]*)"')
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _protect_math(text: str) -> tuple[str, list[str]]:
    """Swap $..$ / $$..$$ outside code for opaque tokens so Markdown cannot touch them."""
    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return _TOKEN.format(len(spans) - 1)

    pieces, last = [], 0
    for code in _CODE_OR_FENCE.finditer(text):
        prose = text[last : code.start()]
        pieces.append(_INLINE_MATH.sub(stash, _DISPLAY_MATH.sub(stash, prose)))
        pieces.append(code.group(0))
        last = code.end()
    pieces.append(_INLINE_MATH.sub(stash, _DISPLAY_MATH.sub(stash, text[last:])))
    return "".join(pieces), spans


def _rewrite_href(match: re.Match[str]) -> str:
    """Links in the roadmap are relative to docs/; the page lives in roadmap_site/."""
    href = match.group(1)
    if not href or href.startswith(("#", "/")) or _SCHEME.match(href):
        return match.group(0)
    target = os.path.relpath(SOURCE.parent / urllib.parse.unquote(href), SITE_DIR)
    return f'href="{urllib.parse.quote(Path(target).as_posix())}"'


def render(text: str) -> str:
    protected, spans = _protect_math(text)
    _MD.reset()
    out = _MD.convert(protected)
    for k, span in enumerate(spans):
        out = out.replace(_TOKEN.format(k), html.escape(span, quote=False))
    out = _HREF.sub(_rewrite_href, out)
    return out.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")


def render_inline(text: str) -> str:
    out = render(text).strip()
    if out.startswith("<p>") and out.endswith("</p>") and out.count("<p>") == 1:
        out = out[3:-4]
    return out


def plain(text: str) -> str:
    """Rendered inline text with tags stripped, for title/alt attributes."""
    return html.escape(html.unescape(re.sub(r"<[^>]+>", "", render_inline(text))))


# ---------------------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------------------

def ring_svg(stage: Stage) -> str:
    total = len(stage.missions)
    parts = []
    if stage.state == "done":
        parts.append('<circle class="ring-fill" cx="12" cy="12" r="8.5"/>')
    else:
        parts.append('<circle class="ring-track" cx="12" cy="12" r="8.5"/>')
        if stage.n_done:
            share = 100 * stage.n_done / total
            parts.append(
                '<circle class="ring-arc" cx="12" cy="12" r="8.5" pathLength="100" '
                f'stroke-dasharray="{share:.2f} 100" transform="rotate(-90 12 12)"/>'
            )
    if stage.current:
        parts.insert(0, '<circle class="ring-halo" cx="12" cy="12" r="8.5"/>')
    return f'<svg class="ring" viewBox="0 0 24 24" aria-hidden="true">{"".join(parts)}</svg>'


def count_text(stage: Stage) -> str:
    noun = "mission" if len(stage.missions) == 1 else "missions"
    return f"{stage.n_done} of {len(stage.missions)} {noun}"


STATE_WORDS = {"done": "done", "started": "in progress", "planned": "planned"}

CHECK_DONE = (
    '<svg class="check" viewBox="0 0 16 16" aria-hidden="true"><rect class="check-box is-done" x="1" y="1" '
    'width="14" height="14" rx="4"/><path class="check-mark" d="M4.6 8.3l2.2 2.2 4.6-4.9"/></svg>'
)
CHECK_TODO = (
    '<svg class="check" viewBox="0 0 16 16" aria-hidden="true"><rect class="check-box" x="1.5" y="1.5" '
    'width="13" height="13" rx="3.5"/></svg>'
)


def reflow(prompt: str) -> str:
    """Join the hard-wrapped lines of a prompt into paragraphs.

    The Markdown wraps prompts at ~95 columns; shown and pasted as-is they break twice.
    Blank lines, list items and indented lines keep their own line.
    """
    out: list[str] = []
    for line in prompt.splitlines():
        joinable = out and out[-1].strip() and not PROMPT_KEEP_BREAK.match(line) and not line[:1].isspace()
        if joinable:
            out[-1] = out[-1].rstrip() + " " + line.strip()
        else:
            out.append(line.rstrip())
    return "\n".join(out)


def asset_url(file: str) -> str:
    """URL of a plot, with a content hash so a replaced file is never served stale."""
    digest = hashlib.sha1((ASSETS / file).read_bytes()).hexdigest()[:10]
    return f"{ASSETS_DIRNAME}/{urllib.parse.quote(file)}?v={digest}"


def render_stage(stage: Stage, index: int, travelled: bool) -> str:
    more_id = f"{stage.anchor}-more"
    classes = ["stage", f"is-{stage.state}"]
    if stage.current:
        classes.append("is-current")
    if travelled:
        classes.append("is-travelled")
    meta = [f'<span class="stage-num">{html.escape(stage.label or f"Stage {index}")}</span>']
    meta.append(f'<span class="stage-count">{count_text(stage)}</span>')
    if stage.current:
        meta.append('<span class="stage-flag">Current stage</span>')
    if stage.added:
        meta.append(f'<span class="stage-added">Added {html.escape(stage.added)}</span>')

    out = [f'<li class="{" ".join(classes)}" id="{stage.anchor}">']
    out.append(f'<div class="stage-marker">{ring_svg(stage)}</div>')
    out.append('<div class="stage-body">')
    out.append(f'<div class="stage-meta">{"".join(meta)}</div>')
    out.append(
        f'<h2 class="stage-title"><button type="button" class="stage-toggle" aria-controls="{more_id}" '
        f'aria-expanded="false">{render_inline(stage.title)}</button></h2>'
    )
    if stage.target:
        out.append(f'<p class="stage-target">{render_inline(stage.target)}</p>')
    out.append(f'<div class="stage-short">{render(stage.short)}</div>')
    if stage.long:
        out.append(
            f'<details class="more" id="{more_id}"><summary>'
            '<svg class="chev" viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3.5L10.5 8 6 12.5"/></svg>'
            '<span class="more-closed">Read the full stage</span><span class="more-open">Hide the full stage</span>'
            f'</summary><div class="more-body prose">{render(stage.long)}</div></details>'
        )

    out.append('<ul class="missions">')
    for mission in stage.missions:
        state = "is-done" if mission.done else "is-todo"
        sr = "Done: " if mission.done else "To do: "
        out.append(
            f'<li class="{state}">{CHECK_DONE if mission.done else CHECK_TODO}'
            f'<span><span class="sr">{sr}</span>{render_inline(mission.text)}</span></li>'
        )
    out.append("</ul>")

    if stage.done_when:
        done_html = render(stage.done_when)
        label = '<span class="done-label">Done when</span> '
        done_html = done_html.replace("<p>", "<p>" + label, 1) if done_html.startswith("<p>") else label + done_html
        out.append(f'<div class="done-when">{done_html}</div>')

    if stage.prompt:
        prompt = reflow(stage.prompt)
        folds = prompt.count("\n") + 1 > PROMPT_PREVIEW_LINES or len(prompt) > PROMPT_PREVIEW_CHARS
        out.append(f'<div class="prompt{" is-folded" if folds else ""}">')
        out.append(
            '<div class="prompt-bar"><span>Prompt to start this stage</span>'
            '<button type="button" class="copy" aria-live="polite">Copy</button></div>'
        )
        out.append(f"<pre><code>{html.escape(prompt, quote=False)}</code></pre>")
        if folds:
            out.append('<button type="button" class="prompt-expand" aria-expanded="false">Show the whole prompt</button>')
        out.append("</div>")

    if stage.plots:
        out.append('<div class="figures">')
        for plot in stage.plots:
            notes = f'<div class="fig-notes">{render(plot.notes)}</div>' if plot.notes else ""
            out.append(
                '<figure><button type="button" class="fig-open" aria-label="Open the plot larger">'
                f'<img src="{asset_url(plot.file)}" alt="{plain(plot.caption)}" loading="lazy" decoding="async">'
                f'</button><figcaption><div class="fig-caption">{render_inline(plot.caption)}</div>{notes}'
                "</figcaption></figure>"
            )
        out.append("</div>")
    out.append("</div></li>")
    return "\n".join(out)


def render_section_list(sections: list[Section], css_class: str) -> str:
    if not sections:
        return ""
    out = [f'<div class="{css_class}">']
    for section in sections:
        if not section.heading:
            out.append(f'<div class="prose">{render(section.body)}</div>')
            continue
        out.append(
            f'<details class="aside" id="{slugify(section.heading)}"><summary>'
            '<svg class="chev" viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3.5L10.5 8 6 12.5"/></svg>'
            f'<span>{render_inline(section.heading)}</span></summary>'
            f'<div class="aside-body prose">{render(section.body)}</div></details>'
        )
    out.append("</div>")
    return "\n".join(out)


def split_lead(intro: list[Section]) -> tuple[str, list[Section]]:
    """The first paragraph of the introduction becomes the lead under the title."""
    for n, section in enumerate(intro):
        lines = section.body.splitlines()
        blocks = blocks_of(lines)
        if not blocks:
            continue
        start, end = blocks[0]
        lead = "\n".join(lines[start:end])
        rest = "\n".join(trim(lines[end:]))
        remaining = intro[:n] + ([Section(section.heading, rest)] if rest else []) + intro[n + 1 :]
        return lead, remaining
    return "", intro


def render_page(roadmap: Roadmap, vendor_ok: bool) -> str:
    stages = roadmap.stages
    n_done = sum(s.state == "done" for s in stages)
    missions_done = sum(s.n_done for s in stages)
    missions_total = sum(len(s.missions) for s in stages)
    current_index = next((k for k, s in enumerate(stages) if s.current), None)
    current = stages[current_index] if current_index is not None else None

    strip = []
    for stage in stages:
        classes = ["strip-item", f"is-{stage.state}"] + (["is-current"] if stage.current else [])
        tip = f"{plain(stage.label)}: {plain(stage.title)} ({count_text(stage)})"
        strip.append(
            f'<a class="{" ".join(classes)}" href="#{stage.anchor}" title="{tip}" aria-label="{tip}">'
            f'{ring_svg(stage)}<span>{html.escape(stage.code)}</span></a>'
        )
    stage_word = "stage" if len(stages) == 1 else "stages"
    status = f"{n_done} of {len(stages)} {stage_word} done, {missions_done} of {missions_total} missions."
    if current is not None:
        status += (
            f' Now on <a href="#{current.anchor}">{html.escape(current.label)}</a>: '
            f"{render_inline(current.title)}."
        )

    lead, intro = split_lead(roadmap.intro)
    items, stage_index = [], 0
    for item in roadmap.items:
        if isinstance(item, Stage):
            travelled = current_index is not None and stage_index < current_index
            items.append(render_stage(item, stage_index + 1, travelled))
            stage_index += 1
        else:
            items.append(
                f'<li class="interlude"><h2 class="interlude-title">{render_inline(item.heading)}</h2>'
                f'<div class="prose">{render(item.body)}</div></li>'
            )

    head_assets = ""
    tail_assets = ""
    if vendor_ok:
        head_assets = (
            f'<link rel="stylesheet" href="{ASSETS_DIRNAME}/vendor/fonts/fonts.css">\n'
            f'<link rel="stylesheet" href="{ASSETS_DIRNAME}/vendor/katex/katex.min.css">'
        )
        tail_assets = (
            f'<script defer src="{ASSETS_DIRNAME}/vendor/katex/katex.min.js"></script>\n'
            f'<script defer src="{ASSETS_DIRNAME}/vendor/katex/auto-render.min.js"></script>'
        )

    source_rel = SOURCE.relative_to(REPO_ROOT).as_posix()
    script_rel = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    page = TEMPLATE
    replacements = {
        "{{title_text}}": plain(roadmap.title),
        "{{title}}": render_inline(roadmap.title),
        "{{lead}}": render(lead) if lead else "",
        "{{strip}}": "\n".join(strip),
        "{{status}}": status,
        "{{intro}}": render_section_list(intro, "asides"),
        "{{timeline}}": "\n".join(items),
        "{{appendix}}": render_section_list(roadmap.appendix, "asides asides-end"),
        "{{head_assets}}": head_assets,
        "{{tail_assets}}": tail_assets,
        "{{css}}": CSS,
        "{{js}}": JS,
        "{{footer}}": (
            f"Generated from <code>{source_rel}</code> by <code>{script_rel}</code>. "
            "Edit the Markdown and rerun the script; never edit this page."
        ),
    }
    for key, value in replacements.items():
        page = page.replace(key, value)
    return page


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{{title_text}}</title>
{{head_assets}}
<style>
{{css}}
</style>
</head>
<body>
<main class="page">
<header class="masthead">
<h1>{{title}}</h1>
<div class="lead">{{lead}}</div>
<nav class="strip" aria-label="Stages">
{{strip}}
</nav>
<p class="status">{{status}}</p>
</header>
{{intro}}
<ol class="timeline">
{{timeline}}
</ol>
{{appendix}}
<footer class="foot">{{footer}}</footer>
</main>
<dialog class="lightbox" aria-label="Plot">
<button type="button" class="lightbox-close" aria-label="Close">
<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8"/></svg>
</button>
<img alt="">
<div class="lightbox-caption"></div>
</dialog>
{{tail_assets}}
<script>
{{js}}
</script>
</body>
</html>
"""

CSS = r"""
:root {
  --paper: #eef1f5;
  --surface: #ffffff;
  --sunk: #e3e7ed;
  --ink: #162030;
  --ink-2: #4a5568;
  --rule: #c5ccd7;
  --accent: #0d6872;
  --accent-wash: #d3e9ec;
  --plot-bg: #ffffff;
  --serif: "Literata Variable", Literata, KaTeX_Main, KaTeX_AMS, "STIX Two Text", "DejaVu Serif", Georgia, serif;
  --mono: "IBM Plex Mono", ui-monospace, "DejaVu Sans Mono", Menlo, monospace;
  --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
  --rail: 2.25rem;
  --gap: 1.1rem;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #0c1017;
    --surface: #141a24;
    --sunk: #1a212d;
    --ink: #e1e6ed;
    --ink-2: #9ba5b5;
    --rule: #2a3341;
    --accent: #5cc2cb;
    --accent-wash: #12333a;
  }
}
*, *::before, *::after { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font: 400 1.0625rem/1.6 var(--serif);
  font-optical-sizing: auto;
  text-rendering: optimizeLegibility;
}
a { color: var(--accent); text-underline-offset: 0.18em; text-decoration-thickness: 1px; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 4px; }
code {
  font-family: var(--mono);
  font-size: 0.84em;
  background: var(--sunk);
  padding: 0.08em 0.34em;
  border-radius: 4px;
  overflow-wrap: anywhere;
}
.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }

.page { max-width: 48rem; margin: 0 auto; padding: 4.5rem 1.25rem 5rem; }

/* masthead */
.masthead h1 {
  font-weight: 520;
  font-size: clamp(2.05rem, 5vw, 3.2rem);
  line-height: 1.06;
  letter-spacing: -0.022em;
  margin: 0 0 1.25rem;
  text-wrap: balance;
}
.lead { color: var(--ink-2); font-size: 1.125rem; max-width: 40rem; }
.lead p { margin: 0; }
.strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem 0.1rem;
  margin: 2.5rem 0 0.9rem;
}
.strip-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.3rem;
  width: 2.75rem;
  padding: 0.35rem 0 0.3rem;
  border-radius: 8px;
  color: var(--ink-2);
  text-decoration: none;
  font-size: 0.8125rem;
  font-variant-numeric: tabular-nums;
  transition: background-color 160ms ease;
}
.strip-item .ring { width: 1.6rem; height: 1.6rem; }
.strip-item.is-current { color: var(--accent); font-weight: 600; }
@media (hover: hover) and (pointer: fine) { .strip-item:hover { background: var(--sunk); } }
.status { margin: 0; font-size: 1.0625rem; max-width: 40rem; }

/* rings: filled = done, arc = in progress, hollow = planned, halo = current */
.ring { overflow: visible; display: block; }
.ring circle { fill: none; stroke-width: 2; }
.ring .ring-track { fill: var(--paper); stroke: var(--rule); }
.ring .ring-arc { stroke: var(--ink); stroke-width: 3; stroke-linecap: round; }
.ring .ring-fill { fill: var(--ink); stroke: var(--ink); }
.is-current .ring .ring-track { stroke: var(--accent); stroke-opacity: 0.35; }
.is-current .ring .ring-arc { stroke: var(--accent); }
.is-current .ring .ring-fill { fill: var(--accent); stroke: var(--accent); }
.ring .ring-halo { stroke: var(--accent); transform-box: fill-box; transform-origin: center; }
@media (prefers-reduced-motion: no-preference) {
  .ring-halo { animation: halo 2.6s var(--ease-out) infinite; }
}
@media (prefers-reduced-motion: reduce) {
  .ring-halo { opacity: 0.3; transform: scale(1.45); }
}
@keyframes halo {
  0% { transform: scale(1); opacity: 0.6; }
  70%, 100% { transform: scale(1.85); opacity: 0; }
}

/* collapsible reference sections (introduction, appendix) */
.asides { margin-top: 2.25rem; display: grid; gap: 0.15rem; }
.asides-end { margin-top: 1rem; }
.aside > summary, .more > summary {
  list-style: none;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.3rem 0;
  border-radius: 6px;
}
.aside > summary::-webkit-details-marker, .more > summary::-webkit-details-marker { display: none; }
.aside > summary { font-weight: 560; }
.chev { width: 0.95rem; height: 0.95rem; flex: none; fill: none; stroke: currentColor; stroke-width: 1.8;
  stroke-linecap: round; stroke-linejoin: round; transition: transform 200ms var(--ease-out); }
details[open] > summary .chev { transform: rotate(90deg); }
.aside-body { padding: 0.25rem 0 1.25rem 1.4rem; }
@media (prefers-reduced-motion: no-preference) {
  .aside[open] > .aside-body, .more[open] > .more-body { animation: reveal 240ms var(--ease-out); }
}
@keyframes reveal { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: none; } }

/* timeline */
.timeline { list-style: none; margin: 3.75rem 0 0; padding: 0; }
.stage, .interlude { position: relative; padding: 0 0 3.25rem calc(var(--rail) + var(--gap)); }
.stage::before, .interlude::before {
  content: "";
  position: absolute;
  left: calc(var(--rail) / 2 - 1px);
  top: 0;
  bottom: 0;
  width: 2px;
  background: var(--rule);
}
.timeline > :first-child::before { top: 0.9rem; }
.timeline > :last-child::before { bottom: auto; height: 0.9rem; }
.stage.is-travelled::before { background: var(--ink-2); }
.stage-marker {
  position: absolute;
  left: 0;
  top: 0.05rem;
  width: var(--rail);
  display: flex;
  justify-content: center;
  padding: 0.25rem 0;
  background: var(--paper);
}
.stage-marker .ring { width: 1.6rem; height: 1.6rem; }
.stage.is-current .stage-marker { top: 1.15rem; }
.stage.is-current .stage-body {
  background: var(--surface);
  border-radius: 16px;
  padding: 1.2rem 1.4rem 1.4rem;
  margin-left: -0.4rem;
}
.stage-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.2rem 0.9rem;
  align-items: baseline;
  font-size: 0.875rem;
  color: var(--ink-2);
  font-variant-numeric: tabular-nums;
}
.stage-num { font-weight: 620; color: var(--ink); }
.stage-flag { color: var(--accent); font-weight: 600; }
.stage-title {
  font-size: 1.4rem;
  line-height: 1.25;
  font-weight: 600;
  letter-spacing: -0.012em;
  margin: 0.25rem 0 0.4rem;
  text-wrap: balance;
}
.stage.is-planned .stage-title { font-weight: 540; }
.stage-toggle {
  all: unset;
  cursor: pointer;
  border-radius: 4px;
}
.stage-toggle:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
@media (hover: hover) and (pointer: fine) {
  .stage-toggle:hover { text-decoration: underline; text-decoration-thickness: 1px; text-underline-offset: 0.18em; }
}
.stage-target { margin: 0 0 0.5rem; font-size: 0.9375rem; color: var(--ink-2); }
.stage-short p { margin: 0.4rem 0 0; }
.more { margin-top: 0.45rem; }
.more > summary { color: var(--accent); font-size: 0.9375rem; font-weight: 560; }
.more[open] .more-closed, .more:not([open]) .more-open { display: none; }
.more-body { padding: 0.4rem 0 0.4rem; }

/* prose from the Markdown */
.prose > :first-child { margin-top: 0; }
.prose p, .prose ul, .prose ol { margin: 0.75rem 0; }
.prose ul, .prose ol { padding-left: 1.3rem; }
.prose li { margin: 0.3rem 0; }
.prose li::marker { color: var(--ink-2); }
.prose h3, .prose h4 { font-size: 1.0625rem; font-weight: 640; margin: 1.6rem 0 0.4rem; }
.prose pre { background: var(--sunk); border-radius: 10px; padding: 0.8rem 1rem; overflow-x: auto; }
.prose pre code { background: none; padding: 0; }
.prose hr { border: 0; height: 1px; background: var(--rule); margin: 1.5rem 0; }
.table-wrap { overflow-x: auto; margin: 1rem 0; }
.prose table { border-collapse: collapse; font-size: 0.875rem; line-height: 1.45; min-width: 100%; }
.prose th, .prose td { text-align: left; vertical-align: top; padding: 0.45rem 0.7rem 0.45rem 0; border-bottom: 1px solid var(--rule); }
.prose th { font-weight: 620; }

/* missions */
.missions { list-style: none; margin: 1.05rem 0 0; padding: 0; display: grid; gap: 0.4rem; }
.missions > li { display: grid; grid-template-columns: 1rem 1fr; gap: 0.65rem; align-items: start; }
.missions > li.is-done { color: var(--ink-2); }
.check { width: 1rem; height: 1rem; margin-top: 0.33rem; }
.check-box { fill: none; stroke: var(--ink-2); stroke-width: 1.4; }
.check-box.is-done { fill: var(--ink-2); stroke: none; }
.is-current .check-box.is-done { fill: var(--accent); }
.check-mark { fill: none; stroke: var(--paper); stroke-width: 1.9; stroke-linecap: round; stroke-linejoin: round; }
.is-current .check-mark { stroke: var(--surface); }

/* done when */
.done-when { margin-top: 1.05rem; font-size: 0.96875rem; color: var(--ink-2); }
.done-label { font-weight: 620; color: var(--ink); }
.done-when p { margin: 0; }

/* prompt box */
.prompt { margin-top: 1.15rem; background: var(--sunk); border-radius: 12px; }
.is-current .prompt { background: var(--paper); }
.prompt-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  padding: 0.5rem 0.5rem 0.15rem 1rem;
  font-size: 0.875rem;
  color: var(--ink-2);
}
.copy {
  font: 560 0.875rem/1 var(--serif);
  color: var(--paper);
  background: var(--ink);
  border: 0;
  border-radius: 8px;
  padding: 0.5rem 0.85rem;
  min-width: 4.6rem;
  cursor: pointer;
  transition: transform 160ms var(--ease-out), background-color 160ms ease;
}
.copy:active { transform: scale(0.97); }
.copy[data-state="copied"] { background: var(--accent); }
@media (hover: hover) and (pointer: fine) { .copy:hover { background: var(--ink-2); } }
.prompt pre {
  margin: 0;
  padding: 0.45rem 1rem 0.95rem;
  font: 400 0.8125rem/1.6 var(--mono);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.prompt pre code { background: none; padding: 0; font-size: inherit; }
.prompt.is-folded:not(.is-expanded) pre {
  max-height: calc(1.6em * 4 + 0.45rem);
  overflow: hidden;
  -webkit-mask-image: linear-gradient(to bottom, #000 45%, transparent);
  mask-image: linear-gradient(to bottom, #000 45%, transparent);
}
.prompt-expand {
  all: unset;
  cursor: pointer;
  display: block;
  margin: -0.3rem 1rem 0;
  padding: 0 0 0.75rem;
  font-size: 0.875rem;
  font-weight: 560;
  color: var(--accent);
  border-radius: 4px;
}
.prompt-expand:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* figures */
.figures { display: grid; grid-template-columns: repeat(auto-fill, minmax(15.5rem, 1fr)); gap: 1.4rem 1.25rem; margin-top: 1.5rem; }
.figures figure { margin: 0; }
.fig-open {
  all: unset;
  display: block;
  cursor: zoom-in;
  border-radius: 10px;
  overflow: hidden;
  background: var(--plot-bg);
  transition: transform 160ms var(--ease-out);
}
.fig-open:active { transform: scale(0.985); }
.fig-open:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
.fig-open img { display: block; width: 100%; height: auto; }
.figures figcaption { margin-top: 0.6rem; font-size: 0.9375rem; line-height: 1.5; }
.fig-caption { font-weight: 600; }
.fig-notes { color: var(--ink-2); }
.fig-notes p { margin: 0.25rem 0 0; }

/* lightbox */
.lightbox {
  border: 0;
  padding: 0;
  border-radius: 16px;
  background: var(--surface);
  color: var(--ink);
  width: min(94vw, 84rem);
  max-height: 94vh;
  overflow: auto;
}
.lightbox::backdrop { background: rgb(6 10 16 / 0.74); }
@media (prefers-reduced-motion: no-preference) {
  .lightbox[open] { animation: pop 220ms var(--ease-out); }
  .lightbox[open]::backdrop { animation: fade 220ms ease; }
}
@keyframes pop { from { opacity: 0; transform: scale(0.96); } to { opacity: 1; transform: none; } }
@keyframes fade { from { opacity: 0; } to { opacity: 1; } }
.lightbox img { display: block; width: 100%; max-height: calc(94vh - 9rem); object-fit: contain; background: var(--plot-bg); }
.lightbox-caption { padding: 1rem 1.4rem 1.3rem; max-width: 46rem; font-size: 0.96875rem; }
.lightbox-caption .fig-notes p { margin-top: 0.3rem; }
.lightbox-close {
  position: absolute;
  top: 0.7rem;
  right: 0.7rem;
  width: 2.25rem;
  height: 2.25rem;
  display: grid;
  place-items: center;
  border: 0;
  border-radius: 999px;
  background: rgb(22 32 48 / 0.72);
  color: #fff;
  cursor: pointer;
  transition: transform 160ms var(--ease-out);
}
.lightbox-close:active { transform: scale(0.94); }
.lightbox-close svg { width: 0.9rem; height: 0.9rem; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; }

.interlude-title { font-size: 1.15rem; margin: 0 0 0.4rem; }
.foot { margin-top: 3.5rem; font-size: 0.875rem; color: var(--ink-2); }

@media (max-width: 40rem) {
  :root { --rail: 1.75rem; --gap: 0.75rem; }
  .page { padding: 2.75rem 1rem 3.5rem; }
  .stage-marker .ring { width: 1.35rem; height: 1.35rem; }
  .stage.is-current .stage-body { padding: 1rem 1rem 1.15rem; margin-left: -0.25rem; }
  .stage-title { font-size: 1.25rem; }
  .strip-item { width: 2.45rem; }
}
"""

JS = r"""
(() => {
  const lightbox = document.querySelector(".lightbox");

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.append(area);
      area.select();
      const ok = document.execCommand("copy");
      area.remove();
      return ok;
    }
  }

  document.addEventListener("click", async (event) => {
    const copy = event.target.closest(".copy");
    if (copy) {
      const ok = await copyText(copy.closest(".prompt").querySelector("code").textContent);
      copy.dataset.state = ok ? "copied" : "failed";
      copy.textContent = ok ? "Copied" : "Copy failed";
      clearTimeout(copy.resetTimer);
      copy.resetTimer = setTimeout(() => {
        copy.textContent = "Copy";
        delete copy.dataset.state;
      }, 1600);
      return;
    }
    const expand = event.target.closest(".prompt-expand");
    if (expand) {
      const open = expand.closest(".prompt").classList.toggle("is-expanded");
      expand.textContent = open ? "Show less" : "Show the whole prompt";
      expand.setAttribute("aria-expanded", String(open));
      return;
    }
    const toggle = event.target.closest(".stage-toggle");
    if (toggle) {
      const details = document.getElementById(toggle.getAttribute("aria-controls"));
      if (details) details.open = !details.open;
      return;
    }
    const figure = event.target.closest(".fig-open");
    if (figure) {
      const img = figure.querySelector("img");
      lightbox.querySelector("img").src = img.src;
      lightbox.querySelector("img").alt = img.alt;
      lightbox.querySelector(".lightbox-caption").innerHTML =
        figure.closest("figure").querySelector("figcaption").innerHTML;
      lightbox.showModal();
      return;
    }
    if (event.target === lightbox || event.target.closest(".lightbox-close")) lightbox.close();
  });

  document.querySelectorAll("details.more").forEach((details) => {
    details.addEventListener("toggle", () => {
      const button = document.querySelector(`[aria-controls="${details.id}"]`);
      if (button) button.setAttribute("aria-expanded", String(details.open));
    });
  });

  document.addEventListener("DOMContentLoaded", () => {
    if (!window.renderMathInElement) return;
    window.renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  });
})();
"""


# ---------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="validate the roadmap; write nothing")
    parser.add_argument("--fetch-vendor", action="store_true", help="download KaTeX and the fonts into roadmap_assets/vendor/")
    args = parser.parse_args(argv)

    if args.fetch_vendor:
        fetch_vendor()

    try:
        roadmap = parse(SOURCE.read_text(encoding="utf-8"))
    except BuildErrors as exc:
        print(f"{SOURCE.relative_to(REPO_ROOT)} has problems; the page was not written:", file=sys.stderr)
        for message in exc.messages:
            print(f"  - {message}", file=sys.stderr)
        return 1

    stages = roadmap.stages
    missing = missing_vendor_files()
    if missing:
        print(
            "warning: vendor files missing (" + ", ".join(missing) + "); the page will use system fonts and "
            "show raw $...$. Run once with --fetch-vendor.",
            file=sys.stderr,
        )
    if not any(s.current for s in stages):
        print("warning: no stage has 'current: true'", file=sys.stderr)

    by_state = {state: sum(s.state == state for s in stages) for state in STATE_WORDS}
    current = next((s for s in stages if s.current), None)
    summary = (
        f"{len(stages)} stages: " + ", ".join(f"{by_state[k]} {STATE_WORDS[k]}" for k in STATE_WORDS)
        + f"; {sum(s.n_done for s in stages)} of {sum(len(s.missions) for s in stages)} missions done"
        + f"; {sum(len(s.plots) for s in stages)} plots"
        + (f"; current: {current.label}" if current else "")
    )
    if args.check:
        print(f"ok: {summary}")
        return 0

    page = render_page(roadmap, vendor_ok=not missing)
    previous = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else None
    if previous == page:
        print(f"unchanged: {OUTPUT.relative_to(REPO_ROOT)} ({summary})")
    else:
        OUTPUT.write_text(page, encoding="utf-8")
        print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
