#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source src/commit_utils/set_env_vars.sh

container="${1:?Usage: $0 <container-name>}"
def_file="containers/${container}.def"
out_file="${INFERENCE_BENCH_CONTAINERS_DIR}/${container}.sif"
[ -f "${def_file}" ] || { echo "Definition file not found: ${def_file}" >&2; exit 2; }

export APPTAINER_BIND=""
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${_CONDOR_SCRATCH_DIR:-${HOME}}/apptainer_tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${_CONDOR_SCRATCH_DIR:-${HOME}/.cache}/apptainer}"
mkdir -p "${INFERENCE_BENCH_CONTAINERS_DIR}" "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

if ! flock -n "${APPTAINER_TMPDIR}" true; then
  echo "APPTAINER_TMPDIR must support file locks: ${APPTAINER_TMPDIR}" >&2
  exit 4
fi

fakeroot=()
[ "$(id -u)" -eq 0 ] || fakeroot=(--fakeroot)

echo "Building ${out_file} from ${def_file}"
apptainer build "${fakeroot[@]}" \
  --mksquashfs-args "${APPTAINER_SQUASHFS_OPTS:--no-xattrs -processors 1 -mem 1024M}" \
  "${out_file}" "${def_file}"
(cd "$(dirname "${out_file}")" && sha256sum "$(basename "${out_file}")") > "${out_file}.sha256"
