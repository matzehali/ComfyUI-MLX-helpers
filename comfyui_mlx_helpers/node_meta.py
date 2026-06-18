"""Node display version (from the git tag) + Apple-logo naming + logging.

Shared infrastructure for the ComfyUI MLX port projects. The version shown in a
node title is read from a repo's git tag *without spawning a subprocess* (so it
works even when ComfyUI is launched from a GUI app with a minimal PATH). Both
lightweight and annotated tags are supported; if HEAD is past the newest tag the
version gets a ``-dirty`` suffix.

Each consumer project resolves *its own* version against *its own* checkout::

    from comfyui_mlx_helpers import node_meta
    meta = node_meta.for_repo(__file__, fallback="v0.4", log_prefix="DA2-MLX")

    NODE_DISPLAY_NAME_MAPPINGS = {
        "LoadFooMLX": meta.versioned(f"{meta.LOGO} MLX Model Loader Foo"),
    }
    meta.log("weights loaded")

``for_repo`` walks up from the given file to find the ``.git`` directory, so the
node title reflects the *consumer's* tag, not the helpers package version.
"""

from __future__ import annotations

import re
import zlib
from copy import deepcopy
from pathlib import Path

LOGO = ""  # Apple logo (U+F8FF)
_BAR = "═" * 60


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _repo_git_dir(repo_root: Path) -> Path | None:
    git_path = repo_root / ".git"
    if git_path.is_dir():
        return git_path
    if git_path.is_file():  # worktree/submodule: ".git" is a file pointing elsewhere
        content = _read_text(git_path) or ""
        if content.lower().startswith("gitdir:"):
            target = Path(content.split(":", 1)[1].strip())
            return target if target.is_absolute() else (repo_root / target).resolve()
    return None


def _find_git_dir(start: Path) -> Path | None:
    """Walk up from *start* looking for a usable .git directory."""
    start = start.resolve()
    candidates = [start] if start.is_dir() else []
    candidates += list(start.parents)
    for root in candidates:
        git_dir = _repo_git_dir(root)
        if git_dir is not None:
            return git_dir
    return None


def _head_commit(git_dir: Path) -> str | None:
    head = _read_text(git_dir / "HEAD")
    if not head:
        return None
    if head.startswith("ref:"):
        return _read_text(git_dir / head.split(":", 1)[1].strip())
    return head


def _deref_tag_object(git_dir: Path, sha: str) -> str:
    """Resolve an annotated-tag object SHA to the commit SHA it points at.

    Lightweight tags already point at a commit, so *sha* is returned unchanged.
    Missing/packed objects fall back to *sha* so the caller degrades to a
    ``-dirty`` suffix rather than crashing.
    """
    obj_path = git_dir / "objects" / sha[:2] / sha[2:]
    try:
        raw = zlib.decompress(obj_path.read_bytes())
    except (OSError, zlib.error):
        return sha
    null_pos = raw.find(b"\0")
    if null_pos < 0 or not raw[:null_pos].startswith(b"tag "):
        return sha  # already a commit object
    for line in raw[null_pos + 1:].split(b"\n"):
        if line.startswith(b"object "):
            return line[7:].decode("ascii").strip()
    return sha


def _tag_commits(git_dir: Path) -> dict[str, str]:
    tags: dict[str, str] = {}
    refs_dir = git_dir / "refs" / "tags"
    if refs_dir.is_dir():
        for path in refs_dir.rglob("*"):
            if path.is_file():
                sha = _read_text(path)
                if sha:
                    tags[path.relative_to(refs_dir).as_posix()] = _deref_tag_object(git_dir, sha)

    packed = _read_text(git_dir / "packed-refs")
    if packed:
        last_tag: str | None = None
        for line in packed.splitlines():
            if not line or line.startswith("#"):
                continue
            if line.startswith("^"):  # peeled ref: commit SHA for the preceding annotated tag
                if last_tag is not None:
                    tags[last_tag] = line[1:].strip()
                last_tag = None
                continue
            last_tag = None
            try:
                sha, ref = line.split(" ", 1)
            except ValueError:
                continue
            prefix = "refs/tags/"
            if ref.startswith(prefix):
                tag_name = ref[len(prefix):]
                tags.setdefault(tag_name, _deref_tag_object(git_dir, sha))
                last_tag = tag_name
    return tags


def _semver_key(tag: str) -> tuple[int, tuple[int, ...], str]:
    match = re.fullmatch(r"v?(\d+(?:\.\d+)*)", tag)
    if match:
        return (1, tuple(int(part) for part in match.group(1).split(".")), tag)
    return (0, (), tag)


def resolve_version(anchor, fallback: str = "v0.1") -> str:
    """Return the newest semver-ish git tag for the repo containing *anchor*.

    *anchor* may be a file (e.g. ``__file__``) or a directory; the search walks
    up to the enclosing checkout. Returns ``fallback`` when there is no git
    checkout (e.g. a packaged install) or no version tags.
    """
    git_dir = _find_git_dir(Path(anchor))
    if git_dir is None:
        return fallback
    tags = {name: sha for name, sha in _tag_commits(git_dir).items() if re.match(r"v?\d", name)}
    if not tags:
        return fallback
    tag = max(tags, key=_semver_key)
    return tag if _head_commit(git_dir) == tags[tag] else f"{tag}-dirty"


def with_mlx_metadata(input_types: dict, help_text: str) -> dict:
    """Add a shared read-only help widget to a ComfyUI INPUT_TYPES dict."""
    result = deepcopy(input_types)
    optional = result.setdefault("optional", {})
    optional["mlx_node_help"] = (
        "STRING",
        {
            "default": help_text.strip(),
            "multiline": True,
            # Help is node documentation, not user state; visible but not editable.
            "readonly": True,
            "disabled": True,
            "serialize": False,
            "tooltip": "Reference help for this node's exposed values.",
        },
    )
    return result


class RepoMeta:
    """Per-repo naming/logging bound to a single checkout's version + log prefix."""

    LOGO = LOGO

    def __init__(self, version: str, log_prefix: str = "MLX"):
        self.VERSION = version
        self.log_prefix = log_prefix

    def versioned(self, name: str) -> str:
        """``name`` -> ``name vX.Y`` (the shared node-title convention)."""
        return f"{name} {self.VERSION}"

    # LTX-compatible alias.
    versioned_display_name = versioned

    def with_metadata(self, input_types: dict, help_text: str) -> dict:
        return with_mlx_metadata(input_types, help_text)

    def banner(self, title: str, *lines: str) -> None:
        print(_BAR)
        print(f"  {title}")
        for line in lines:
            if line:
                print(f"  {line}")
        print(_BAR)

    # LTX-compatible alias.
    status_banner = banner

    def done(self, title: str, *lines: str) -> None:
        self.banner(f"{title} — Complete", *lines)

    status_done = done

    def log(self, message: str) -> None:
        print(f"[{self.log_prefix} {self.VERSION}] {message}")

    status_line = log


def for_repo(anchor, fallback: str = "v0.1", log_prefix: str = "MLX") -> RepoMeta:
    """Build a :class:`RepoMeta` whose version is the git tag of *anchor*'s repo."""
    return RepoMeta(resolve_version(anchor, fallback), log_prefix)


# --- the helpers package's own version (for the nodes it registers) ----------
VERSION = resolve_version(__file__, "v0.1")
_self = RepoMeta(VERSION, "MLX-helpers")


def versioned(name: str) -> str:
    """``name`` -> ``name vX.Y`` using the helpers package's own version."""
    return _self.versioned(name)


versioned_display_name = versioned


def banner(title: str, *lines: str) -> None:
    _self.banner(title, *lines)


status_banner = banner


def status_done(title: str, *lines: str) -> None:
    _self.done(title, *lines)


def log(message: str) -> None:
    _self.log(message)


status_line = log
