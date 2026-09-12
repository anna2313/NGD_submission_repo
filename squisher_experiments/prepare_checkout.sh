#!/usr/bin/env bash

set -euo pipefail

UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/GMvandeVen/continual-learning.git}"
UPSTREAM_COMMIT="e6d795a"
TARGET_DIR="${1:-../continual-learning-squisher}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH="${REPO_DIR}/squisher_experiments/source/squisher_from_upstream_e6d795a.patch.gz"
EXPECTED_PATCH_SHA256="4d9cc8461bd8c7dc170cabd0d1412f60df4160b75f41091070dca458d1e3dd7f"

if [[ -e "${TARGET_DIR}" ]]; then
  printf 'Refusing to overwrite existing path: %s\n' "${TARGET_DIR}" >&2
  exit 2
fi

if command -v sha256sum >/dev/null 2>&1; then
  observed_patch_sha256="$(sha256sum "${PATCH}" | awk '{print $1}')"
else
  observed_patch_sha256="$(shasum -a 256 "${PATCH}" | awk '{print $1}')"
fi
if [[ "${observed_patch_sha256}" != "${EXPECTED_PATCH_SHA256}" ]]; then
  printf 'Patch checksum mismatch: expected %s, observed %s\n' \
    "${EXPECTED_PATCH_SHA256}" "${observed_patch_sha256}" >&2
  exit 2
fi

git clone "${UPSTREAM_URL}" "${TARGET_DIR}"
git -C "${TARGET_DIR}" checkout --detach "${UPSTREAM_COMMIT}"
gzip -dc "${PATCH}" | git -C "${TARGET_DIR}" apply --check -
gzip -dc "${PATCH}" | git -C "${TARGET_DIR}" apply -
git -C "${TARGET_DIR}" diff --check

printf 'Prepared patched checkout at %s\n' "${TARGET_DIR}"
printf 'Create an isolated uv environment as documented in squisher_experiments/README.md.\n'
