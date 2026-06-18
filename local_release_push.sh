#!/usr/bin/env bash
set -euo pipefail

project_name="ComfyUI-MLX-helpers"
default_tag="v0.1"
remote_name="${REMOTE_NAME:-origin}"
python_bin="${PYTHON_BIN:-/Applications/ComfyUI/venv/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
  python_bin="$(command -v python3)"
fi

current_branch="$(git branch --show-current)"
if [[ -z "${current_branch}" ]]; then
  echo "Not on a branch; refusing to push from detached HEAD." >&2
  exit 1
fi

if git rev-parse --verify HEAD >/dev/null 2>&1; then
  head_tags="$(git tag --points-at HEAD | tr '\n' ' ')"
  latest_tag="$(git describe --tags --abbrev=0 2>/dev/null || true)"
  head_sha="$(git rev-parse --short HEAD)"
else
  head_tags=""
  latest_tag=""
  head_sha="unborn"
fi

echo "Current branch: ${current_branch}"
echo "Current HEAD:   ${head_sha}"
echo "Tag(s) at HEAD: ${head_tags:-none}"
echo "Latest tag:     ${latest_tag:-none}"
echo "Project:        ${project_name}"
echo
git status --short
echo

new_tag="${1:-${TAG:-}}"
if [[ -z "${new_tag}" ]]; then
  read -r -p "New tag to create and push [${default_tag}]: " new_tag
  new_tag="${new_tag:-${default_tag}}"
fi

if git rev-parse -q --verify "refs/tags/${new_tag}" >/dev/null; then
  echo "Tag already exists locally: ${new_tag}" >&2
  exit 1
fi

if git ls-remote --exit-code --tags "${remote_name}" "refs/tags/${new_tag}" >/dev/null 2>&1; then
  echo "Tag already exists on ${remote_name}: ${new_tag}" >&2
  exit 1
fi

echo "Running project checks..."
"${python_bin}" -m py_compile comfyui_mlx_helpers/*.py

if [[ -n "$(git status --porcelain)" ]]; then
  echo
  commit_message="${COMMIT_MESSAGE:-}"
  if [[ -z "${commit_message}" ]]; then
    read -r -p "Working tree is dirty. Commit all current changes with message [Release ${new_tag}]: " commit_message
    commit_message="${commit_message:-Release ${new_tag}}"
  fi
  git add -A
  git commit -m "${commit_message}"
fi

echo
echo "Creating tag ${new_tag} at $(git rev-parse --short HEAD)..."
git tag "${new_tag}"

echo "Pushing branch ${current_branch}..."
git push "${remote_name}" "${current_branch}"

echo "Pushing tag ${new_tag}..."
git push "${remote_name}" "${new_tag}"

echo
echo "Done. Pushed ${current_branch} and ${new_tag}."
