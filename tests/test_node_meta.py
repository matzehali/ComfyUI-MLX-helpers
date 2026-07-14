from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from comfyui_mlx_helpers.node_meta import resolve_version


class NodeMetaWorktreeTests(unittest.TestCase):
    def _fake_worktree(self, root: Path, *, head: str, tag: str) -> Path:
        common = root / "main" / ".git"
        metadata = common / "worktrees" / "v3"
        checkout = root / "consumer-v3"
        source = checkout / "nodes.py"
        metadata.mkdir(parents=True)
        (common / "refs" / "heads" / "experimental").mkdir(parents=True)
        (common / "refs" / "tags").mkdir(parents=True)
        checkout.mkdir()
        source.write_text("# node", encoding="utf-8")
        (checkout / ".git").write_text(
            f"gitdir: {metadata}\n",
            encoding="utf-8",
        )
        (metadata / "commondir").write_text("../..\n", encoding="utf-8")
        (metadata / "HEAD").write_text(
            "ref: refs/heads/experimental/v3-nodes\n",
            encoding="utf-8",
        )
        (common / "refs" / "heads" / "experimental" / "v3-nodes").write_text(
            f"{head}\n",
            encoding="utf-8",
        )
        (common / "refs" / "tags" / "v0.25").write_text(
            f"{tag}\n",
            encoding="utf-8",
        )
        return source

    def test_linked_worktree_reads_shared_tag_and_branch_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self._fake_worktree(Path(tmp), head="a" * 40, tag="a" * 40)
            self.assertEqual(resolve_version(source), "v0.25")

    def test_linked_worktree_marks_commit_after_tag_dirty(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self._fake_worktree(Path(tmp), head="b" * 40, tag="a" * 40)
            self.assertEqual(resolve_version(source), "v0.25-dirty")


if __name__ == "__main__":
    unittest.main()
