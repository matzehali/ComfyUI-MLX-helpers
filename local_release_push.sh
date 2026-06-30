#!/usr/bin/env bash
set -euo pipefail

current_branch="$(git branch --show-current)"
if [[ -z "${current_branch}" ]]; then
  echo "Not on a branch; refusing to push from detached HEAD." >&2
  exit 1
fi

head_tags="$(git tag --points-at HEAD | tr '\n' ' ')"
latest_tag="$(git describe --tags --abbrev=0 2>/dev/null || true)"
head_sha="$(git rev-parse --short HEAD)"

suggested_tag="v0.1"
if [[ "${latest_tag}" =~ ^(v?)([0-9]+([.][0-9]+)*)$ ]]; then
  tag_prefix="${BASH_REMATCH[1]}"
  tag_version="${BASH_REMATCH[2]}"
  IFS='.' read -r -a tag_parts <<< "${tag_version}"
  last_part_index="$((${#tag_parts[@]} - 1))"
  tag_parts[${last_part_index}]="$((${tag_parts[${last_part_index}]} + 1))"
  suggested_tag="${tag_prefix}$(IFS='.'; echo "${tag_parts[*]}")"
fi

echo "Current branch: ${current_branch}"
echo "Current HEAD:   ${head_sha}"
echo "Tag(s) at HEAD: ${head_tags:-none}"
echo "Latest tag:     ${latest_tag:-none}"
echo
git status --short
echo

read -r -p "New tag to create and push (current: ${latest_tag:-none}, e.g. ${suggested_tag}): " new_tag
if [[ -z "${new_tag}" ]]; then
  echo "No tag entered; aborting." >&2
  exit 1
fi

if git rev-parse -q --verify "refs/tags/${new_tag}" >/dev/null; then
  echo "Tag already exists locally: ${new_tag}" >&2
  exit 1
fi

if git ls-remote --exit-code --tags origin "refs/tags/${new_tag}" >/dev/null 2>&1; then
  echo "Tag already exists on origin: ${new_tag}" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo
  read -r -p "Working tree is dirty. Commit all current changes first? [y/N] " commit_answer
  case "${commit_answer}" in
    [yY]|[yY][eE][sS])
      read -r -p "Commit message: " commit_message
      if [[ -z "${commit_message}" ]]; then
        echo "No commit message entered; aborting." >&2
        exit 1
      fi
      git add -A
      git commit -m "${commit_message}"
      ;;
    *)
      echo "Dirty working tree left uncommitted; aborting tag/push." >&2
      exit 1
      ;;
  esac
fi

echo
echo "Creating tag ${new_tag} at $(git rev-parse --short HEAD)..."
git tag "${new_tag}"

echo "Pushing branch ${current_branch}..."
git push origin "${current_branch}"

echo "Pushing tag ${new_tag}..."
git push origin "${new_tag}"

echo
echo "Done. Pushed ${current_branch} and ${new_tag}."
