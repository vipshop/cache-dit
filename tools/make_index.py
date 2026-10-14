#!/usr/bin/env python3
"""Emit the static PEP 503 index the CUDA wheels of cache-dit are installed through.

The wheels live as GitHub Release assets of this repository rather than on PyPI:
they carry a local version (``1.5.3+cu13``), which PyPI refuses, and the CUDA line
is meant to move on its own -- one index per toolkit, ``whl/cu13/`` today.

Two builds of the same distribution exist, and the index is what tells them apart:
PyPI serves the pure-Python one, this tree serves the one with the SVDQuant
extension compiled in.  The local version is what makes that work: on the same
version it sorts above the plain release, so pip prefers this build, and a
pure-Python wheel of that version which is already installed no longer satisfies the
requirement unchanged.  An unpinned install takes the highest version across the
indexes, so a newer plain release on PyPI would win over an older build here.
``docs/user_guide/INSTALL.md`` is that note in the form a user needs.

What the tree holds::

    <out>/<prefix>/whl/cu13/index.html               # PEP 503 root: one anchor per project
    <out>/<prefix>/whl/cu13/cache-dit/index.html     # PEP 503 project: one anchor per file
    <out>/<prefix>/index.html                        # human landing page
    <out>/.nojekyll                                  # GitHub Pages runs Jekyll otherwise

Anchors point at the release assets (PEP 503 puts no constraint on where the files
live) and carry ``#sha256=``, which pip enforces.  The output is MERGED into
whatever is already published: a new version must leave the previous version's
links in place, or every ``pip install cache-dit==<old>`` written down against
them stops resolving.  A file rebuilt under the same name replaces its own anchor,
so a re-release cannot leave two hashes for one file.

Stdlib only, so it runs wherever the release script runs.

Usage::

    # the project Pages branch: an empty prefix, because the repository name
    # already supplies the cache-dit/ path.  The glob has to select the channel's
    # wheels -- anything of the distribution without the channel's local version is
    # refused, the pure-Python release included.
    tools/make_index.py --out <gh-pages worktree> --prefix "" \\
        --release-url https://github.com/vipshop/cache-dit/releases/download/v1.5.3+cu13 \\
        --site-url https://vipshop.github.io/cache-dit \\
        "dist/cache_dit-1.5.3+cu13-"*.whl
    # a mirror on another site, under a directory of its own
    tools/make_index.py --out <other site> --prefix cache-dit \\
        --site-url https://deftruth.github.io "dist/cache_dit-1.5.3+cu13-"*.whl
"""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import sys
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

# The project name reaches the renderers through these, which is why they are
# module level: this is a one-shot CLI, and threading a parameter through every
# page template would buy nothing.  main() sets them from --project.
DEFAULT_PROJECT = "cache-dit"
PROJECT = DEFAULT_PROJECT
DEFAULT_CHANNEL = "cu13"
CHANNEL = DEFAULT_CHANNEL
# The directory PEP 503 requires for the project, and the prefix every wheel
# filename of this distribution carries.
NORMALIZED = re.sub(r"[-_.]+", "-", PROJECT).lower()
WHEEL_PREFIX = PROJECT.replace("-", "_") + "-"
WHEEL_TAG = re.compile(r"^(\d+(?:\.\d+)*(?:[abc]|rc)?\d*(?:\.post\d+)?(?:\.dev\d+)?)(?:\+(.+))?$")

_PAGE = """<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="pypi:repository-version" content="1.0">
    <title>{title}</title>
  </head>
  <body>
    <h1>{title}</h1>
{body}
  </body>
</html>
"""


@dataclass(frozen=True)
class Wheel:
  """One published wheel, as far as the index needs to know about it."""

  filename: str
  version: str
  local: str
  platform_tag: str
  sha256: str
  requires_python: str | None


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _requires_python(path: Path) -> str | None:
  """The wheel's own Requires-Python, for the page's data-requires-python."""
  with zipfile.ZipFile(path) as archive:
    metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    if len(metadata) != 1:
      raise SystemExit(f"make_index: {path.name} has {len(metadata)} .dist-info/METADATA files")
    text = archive.read(metadata[0]).decode()
  match = re.search(r"^Requires-Python:\s*(.+)$", text, re.M)
  return match.group(1).strip() if match else None


def read_wheel(path: Path) -> Wheel | None:
  """Parse one wheel; None for a wheel that has no place in this index.

  Two things disqualify a wheel: it is not this distribution, or it does not carry
  this channel's local version.  The second one is what keeps the pure-Python
  release out -- it is the same distribution, the same version number and the same
  package, and an index that means "with the extension compiled in" must not serve
  it, however it was passed in.
  """
  name = path.name
  if not name.endswith(".whl"):
    raise SystemExit(f"make_index: {name} is not a wheel")
  if not name.startswith(WHEEL_PREFIX):
    return None
  parts = name[:-len(".whl")].split("-")
  if len(parts) < 5:
    raise SystemExit(f"make_index: {name} is not a well-formed wheel filename")
  match = WHEEL_TAG.match(parts[1])
  if match is None or (match.group(2) or "") != CHANNEL:
    return None
  return Wheel(
    filename=name,
    version=match.group(1),
    local=match.group(2) or "",
    platform_tag=parts[-1],
    sha256=_sha256(path),
    requires_python=_requires_python(path),
  )


class _LinkParser(HTMLParser):
  """Anchors of an already-published page, as {filename: attributes}.

  Merging happens at this level rather than through a manifest: the published page
  IS the state, so there is nothing that can drift out of sync with it.
  """

  def __init__(self) -> None:
    super().__init__()
    self.links: dict[str, dict[str, str]] = {}
    self._attrs: dict[str, str] | None = None
    self._text = ""

  def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
    if tag == "a":
      self._attrs = {key: value for key, value in attrs if value is not None}
      self._text = ""

  def handle_data(self, data: str) -> None:
    if self._attrs is not None:
      self._text += data

  def handle_endtag(self, tag: str) -> None:
    if tag == "a" and self._attrs is not None:
      name = self._text.strip()
      if name:
        self.links[name] = self._attrs
      self._attrs = None


def read_existing(page: Path) -> dict[str, dict[str, str]]:
  if not page.is_file():
    return {}
  parser = _LinkParser()
  parser.feed(page.read_text(encoding="utf-8"))
  parser.close()
  return parser.links


def _anchor(text: str, attrs: dict[str, str]) -> str:
  order = ["href"] + sorted(key for key in attrs if key != "href")
  rendered = " ".join(f'{key}="{html.escape(attrs[key], quote=True)}"' for key in order)
  return f"    <a {rendered}>{html.escape(text)}</a><br />"


def _attrs(wheel: Wheel, release_url: str) -> dict[str, str]:
  # quote() percent-encodes the "+" of a local version, which is what GitHub's
  # own browser_download_url emits as well.
  href = f"{release_url.rstrip('/')}/{quote(wheel.filename)}#sha256={wheel.sha256}"
  attrs = {"href": href}
  if wheel.requires_python:
    attrs["data-requires-python"] = wheel.requires_python
  return attrs


def _write(path: Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def _version_key(filename: str) -> tuple:
  """Sort key for a wheel filename: release first, then local, then ABI.

  A name the regex cannot parse sorts first rather than raising: the page is
  regenerated from whatever it already holds, so an anchor added by hand must not
  break the regeneration.
  """
  parts = filename[:-len(".whl")].split("-")
  match = WHEEL_TAG.match(parts[1]) if len(parts) > 1 else None
  if not match:
    return ((), "", filename)
  release = tuple(int(token) for token in re.findall(r"\d+", match.group(1)))
  return (release, match.group(2) or "", filename)


def _render_project(links: dict[str, dict[str, str]]) -> str:
  body = "\n".join(_anchor(name, links[name]) for name in sorted(links, key=_version_key))
  return _PAGE.format(title=f"Links for {PROJECT}", body=body)


def _render_root(project_href: str) -> str:
  return _PAGE.format(
    title=f"Simple index for {PROJECT}",
    body=f'    <a href="{html.escape(project_href, quote=True)}">{PROJECT}</a><br />',
  )


def _channels(prefix_dir: Path) -> list[str]:
  """Every channel already published under the prefix, newest release last."""
  whl_dir = prefix_dir / "whl"
  if not whl_dir.is_dir():
    return []
  return sorted(item.name for item in whl_dir.iterdir() if item.is_dir())


def _channel_files(prefix_dir: Path, channel: str) -> list[str]:
  project_dir = prefix_dir / "whl" / channel / NORMALIZED
  if not project_dir.is_dir():
    return []
  return sorted(read_existing(project_dir / "index.html"), key=_version_key)


def _render_landing(prefix_dir: Path, index_base: str, channels: list[str]) -> str:
  """The human landing page; index_base is the URL of the channel directory."""
  base = index_base.rstrip("/")
  commands = "\n".join(f"pip install {PROJECT} --extra-index-url {base}/{channel}/"
                       for channel in channels)
  local_versions = ", ".join(f"<code>+{channel}</code>" for channel in channels)
  platforms: set[str] = set()
  rows: list[str] = []
  for channel in channels:
    files = _channel_files(prefix_dir, channel)
    platforms.update(file[:-len(".whl")].rsplit("-", 1)[-1] for file in files)
    listing = "\n".join(f"      <li><code>{html.escape(file)}</code></li>" for file in files)
    rows.append(f'    <h2><a href="{base}/{channel}/">{base}/{channel}/</a></h2>\n'
                f"    <ul>\n{listing}\n    </ul>")
  platform_note = ", ".join(sorted(platforms)) or "see the file names"
  body = f"""    <p>
      PEP 503 index of the Cache-DiT wheels with the SVDQuant CUDA extension
      compiled in.  The extension is built for the CUDA 13 toolkit, so pick the
      CUDA your torch was built against
      (<code>python -c "import torch; print(torch.version.cuda)"</code>):
    </p>
    <pre>{commands}</pre>
    <p>
      <strong>Install torch first.</strong> These wheels link against a CUDA 13
      torch; nothing here carries torch of its own, so pip only resolves it when it
      is missing -- from PyPI, where it is not the cu13 build.  Install it from the
      PyTorch index you normally use, then install this package.
    </p>
    <p>
      The command adds this index as <code>--extra-index-url</code> rather than
      <code>--index-url</code>: it holds <code>{PROJECT}</code> alone, with no
      dependencies, so PyPI has to stay the index those come from.  That is also the
      form <code>uv</code> resolves: it searches the extra indexes before the default
      one, so naming PyPI as the extra index instead would leave it considering the
      PyPI release only.
    </p>
    <p>
      <code>{PROJECT}</code> is the same distribution PyPI lists, built here with
      the SVDQuant extension compiled in and tagged with the local version
      {local_versions}.  That suffix is what keeps the two apart: on the same version
      it sorts above the plain release, so pip takes this build, and a pure-Python
      wheel of that version which is already installed no longer satisfies the
      requirement unchanged.  Installing this replaces it -- one distribution, one
      <code>cache_dit/</code> package.  It also supersedes the retired
      <code>cache-dit-cu13</code> distribution.
    </p>
    <p>
      Pin the version when the channel is what you want:
      <code>{PROJECT}==&lt;version&gt;+{CHANNEL}</code>.  An unpinned
      <code>pip install -U {PROJECT}</code> takes the highest version across the
      indexes, so a newer pure-Python release on PyPI would win over an older build
      here, and the install would quietly fall back to a wheel without the extension.
    </p>
    <p>
      Platform tags: <code>{html.escape(platform_note)}</code> -- the tag reflects the
      GLIBC symbols the extension actually needs, so pip refuses a wheel this
      machine cannot run.
    </p>
{chr(10).join(rows)}"""
  return _PAGE.format(title=f"{PROJECT} wheel index", body=body)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("wheels", nargs="+", type=Path, help="wheel files, or directories to scan")
  parser.add_argument("--out", required=True, type=Path, help="site repository root, merged into")
  parser.add_argument("--prefix",
                      default="",
                      help="path under the site root that holds the index; empty for a project "
                      "Pages branch, whose repository name already supplies the project path")
  parser.add_argument("--project",
                      default=DEFAULT_PROJECT,
                      help="distribution name to index; only its own wheels are listed")
  parser.add_argument("--channel",
                      default=DEFAULT_CHANNEL,
                      help="CUDA channel: the sub-index name, and the local version every wheel "
                      "indexed here has to carry")
  parser.add_argument("--release-url",
                      required=True,
                      help="directory URL the wheels are served from, e.g. "
                      "https://github.com/vipshop/cache-dit/releases/download/v1.5.3+cu13")
  parser.add_argument("--site-url",
                      required=True,
                      help="where this tree is served, for the landing page")
  args = parser.parse_args(argv)

  # The project name reaches the renderers through these, which is why they are
  # module level: this is a one-shot CLI, and threading a parameter through every
  # page template would buy nothing.
  global PROJECT, NORMALIZED, WHEEL_PREFIX, CHANNEL
  PROJECT = args.project
  NORMALIZED = re.sub(r"[-_.]+", "-", PROJECT).lower()
  WHEEL_PREFIX = PROJECT.replace("-", "_") + "-"
  CHANNEL = args.channel

  paths: list[Path] = []
  for given in args.wheels:
    if given.is_dir():
      paths += sorted(given.rglob("*.whl"))
    elif given.is_file():
      paths.append(given)
    else:
      raise SystemExit(f"make_index: no such file or directory: {given}")
  if not paths:
    raise SystemExit(f"make_index: no wheels found in {[str(path) for path in args.wheels]}")

  wheels, skipped = [], []
  for path in paths:
    wheel = read_wheel(path)
    (wheels if wheel else skipped).append(wheel or path.name)
  if not wheels:
    raise SystemExit(f"make_index: none of the {len(paths)} wheels is {PROJECT} with the "
                     f"+{CHANNEL} local version")

  prefix_dir = args.out / args.prefix
  # An empty prefix is a site whose root already carries the project name, which
  # is the case for a project Pages branch (vipshop.github.io/cache-dit/).
  index_base = "/".join(part for part in (args.site_url.rstrip("/"), args.prefix.strip("/"), "whl")
                        if part)
  project_page = prefix_dir / "whl" / args.channel / NORMALIZED / "index.html"
  links = read_existing(project_page)
  before = len(links)
  for wheel in wheels:
    links[wheel.filename] = _attrs(wheel, args.release_url)
  _write(project_page, _render_project(links))
  _write(prefix_dir / "whl" / args.channel / "index.html", _render_root(f"{NORMALIZED}/"))
  channels = _channels(prefix_dir)
  _write(prefix_dir / "index.html", _render_landing(prefix_dir, index_base, channels))
  # GitHub Pages runs Jekyll otherwise, which adds a build step this tree has no
  # use for.  It has to sit at the branch root to take effect.
  (args.out / ".nojekyll").touch()

  print(f"{args.prefix}/whl/{args.channel}/{NORMALIZED}/: {len(links)} links, "
        f"{len(links) - before} new")
  print(f"{args.prefix}/: index for channels: {', '.join(channels)}")
  print(f"{args.out}/.nojekyll")
  if skipped:
    print(f"skipped {len(skipped)} wheel(s) that are not {PROJECT} with a "
          f"+{CHANNEL} local version: {', '.join(skipped)}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
