#!/bin/bash
set -euo pipefail

container="${1:-}"
def_override="${2:-}"
if [ -z "${container}" ]; then
  echo "Usage: $0 <container-name> [def-file]" >&2
  exit 2
fi

export INFERENCE_BENCH_CONTAINERS_DIR=${INFERENCE_BENCH_CONTAINERS_DIR:-${POST_TRAIN_BENCH_CONTAINERS_DIR:-containers}}
export APPTAINER_BIND=""

mkdir -p "${INFERENCE_BENCH_CONTAINERS_DIR}"
out_file="${INFERENCE_BENCH_CONTAINERS_DIR}/${container}.sif"
if [ -n "${def_override}" ]; then
  def_file="${def_override}"
else
  def_file="containers/${container}.def"
fi

if [ -n "${INFERENCE_BENCH_BUILD_LOG_DIR:-}" ]; then
  log_root="${INFERENCE_BENCH_BUILD_LOG_DIR}"
elif [[ "${INFERENCE_BENCH_CONTAINERS_DIR}" = /* ]]; then
  log_root="${INFERENCE_BENCH_CONTAINERS_DIR}/build_logs"
else
  log_root="$(pwd)/${INFERENCE_BENCH_CONTAINERS_DIR}/build_logs"
fi
mkdir -p "${log_root}"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${log_root}/${container}_build_${timestamp}.log"
latest_link="${log_root}/${container}_build.log"

exec > >(tee -a "${log_file}") 2>&1
ln -sfn "$(basename "${log_file}")" "${latest_link}" 2>/dev/null || true
echo "[build_container] build_log=${log_file}"
echo "[build_container] latest_log_link=${latest_link}"

_on_exit() {
  local rc=$?
  if [ "${rc}" -ne 0 ]; then
    echo "[build_container] FAILED (exit=${rc}). See log: ${log_file}" >&2
    echo "[build_container] ----- last 80 log lines -----" >&2
    tail -n 80 "${log_file}" >&2 || true
    echo "[build_container] --------------------------------" >&2
  else
    echo "[build_container] SUCCESS. Log: ${log_file}"
  fi
}
trap _on_exit EXIT

echo "[build_container] pwd=$(pwd)"
echo "[build_container] def_file=${def_file}"
echo "[build_container] out_file=${out_file}"
if [ ! -f "${def_file}" ]; then
  echo "[build_container] ERROR: definition file not found: ${def_file}" >&2
  exit 2
fi

user_name="${USER:-$(id -un 2>/dev/null || echo user)}"

supports_flock() {
  local d="${1:-}"
  mkdir -p "${d}" 2>/dev/null || return 1
  python3 - "$d" <<'PY'
import fcntl
import pathlib
import sys

d = pathlib.Path(sys.argv[1])
p = d / ".ptb_flock_test"
with p.open("w") as f:
    f.write("x")
    f.flush()
    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
try:
    p.unlink()
except Exception:
    pass
PY
}

# Keep this build flow simple: one direct SIF build from def file.
if [ -n "${INFERENCE_BENCH_BUILD_SANDBOX_LOCAL_DIR:-}" ] || [ -n "${INFERENCE_BENCH_BUILD_SANDBOX_STAGE_DIR:-}" ] || [ -n "${INFERENCE_BENCH_PACK_APPTAINER_TMPDIR:-}" ] || [ -n "${INFERENCE_BENCH_CONTAINER_BUILD_MODE:-}" ] || [ -n "${INFERENCE_BENCH_FORCE_MKSQUASHFS_WRAPPER:-}" ]; then
  echo "[build_container] WARN: legacy two-stage/sandbox env vars are ignored; using direct build only."
fi
if [ -z "${APPTAINER_TMPDIR:-}" ] && [ -n "${INFERENCE_BENCH_BUILD_APPTAINER_TMPDIR:-}" ]; then
  echo "[build_container] WARN: INFERENCE_BENCH_BUILD_APPTAINER_TMPDIR is deprecated; use APPTAINER_TMPDIR."
  export APPTAINER_TMPDIR="${INFERENCE_BENCH_BUILD_APPTAINER_TMPDIR}"
fi
if [ -z "${APPTAINER_TMPDIR:-}" ]; then
  export APPTAINER_TMPDIR="${HOME%/}/apptainer_tmp"
fi
if ! supports_flock "${APPTAINER_TMPDIR}"; then
  echo "[build_container] ERROR: APPTAINER_TMPDIR does not support file locks: ${APPTAINER_TMPDIR}" >&2
  echo "[build_container] Choose a lock-capable path (typically under /home)." >&2
  exit 4
fi
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/fast/${user_name}/apptainer_cache}"

mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

# Some cluster filesystems (e.g. NFS w/ nfs4 ACLs) attach xattrs that older
# `mksquashfs` doesn't recognize, producing noisy warnings like:
#   "Unrecognised xattr prefix system.nfs4_acl"
# These xattrs are not needed for our containers, so default to skipping them.
if [ -z "${APPTAINER_SQUASHFS_OPTS:-}" ]; then
  # Also cap mksquashfs memory usage; default can be very high and get killed on login nodes.
  # Use gzip compression to reduce peak memory during final SIF creation.
  export APPTAINER_SQUASHFS_OPTS="-no-xattrs -processors 1 -mem 1024M -comp gzip"
fi
if [ -z "${SINGULARITY_SQUASHFS_OPTS:-}" ]; then
  export SINGULARITY_SQUASHFS_OPTS="${APPTAINER_SQUASHFS_OPTS}"
fi
# Some Apptainer builds ignore *_SQUASHFS_OPTS but honor these mksquashfs env vars.
if [ -z "${APPTAINER_MKSQUASHFS_PROCS:-}" ]; then
  export APPTAINER_MKSQUASHFS_PROCS="1"
fi
if [ -z "${APPTAINER_MKSQUASHFS_MEM:-}" ]; then
  export APPTAINER_MKSQUASHFS_MEM="1024M"
fi
if [ -z "${SINGULARITY_MKSQUASHFS_PROCS:-}" ]; then
  export SINGULARITY_MKSQUASHFS_PROCS="${APPTAINER_MKSQUASHFS_PROCS}"
fi
if [ -z "${SINGULARITY_MKSQUASHFS_MEM:-}" ]; then
  export SINGULARITY_MKSQUASHFS_MEM="${APPTAINER_MKSQUASHFS_MEM}"
fi

echo "[build_container] APPTAINER_TMPDIR=${APPTAINER_TMPDIR}"
echo "[build_container] APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR}"
echo "[build_container] APPTAINER_SQUASHFS_OPTS=${APPTAINER_SQUASHFS_OPTS}"
echo "[build_container] APPTAINER_MKSQUASHFS_PROCS=${APPTAINER_MKSQUASHFS_PROCS}"
echo "[build_container] APPTAINER_MKSQUASHFS_MEM=${APPTAINER_MKSQUASHFS_MEM}"
echo "[build_container] SINGULARITY_MKSQUASHFS_PROCS=${SINGULARITY_MKSQUASHFS_PROCS}"
echo "[build_container] SINGULARITY_MKSQUASHFS_MEM=${SINGULARITY_MKSQUASHFS_MEM}"

fakeroot_flag=""
# On some cluster nodes, `apptainer build --help` can fail if NSS/passwd lookup
# is incomplete for the current UID. Prefer a simple UID-based default.
if [ "$(id -u)" -ne 0 ]; then
  fakeroot_flag="--fakeroot"
fi

mksquashfs_args_supported=0
if apptainer build --help 2>/dev/null | grep -q -- '--mksquashfs-args'; then
  mksquashfs_args_supported=1
fi

if [ "${mksquashfs_args_supported}" -ne 1 ]; then
  echo "[build_container] WARN: this apptainer does not support --mksquashfs-args; relying on env var squashfs settings."
fi

build_cmd=(apptainer build)
if [ -n "${fakeroot_flag}" ]; then
  build_cmd+=("${fakeroot_flag}")
fi
if [ "${mksquashfs_args_supported}" -eq 1 ]; then
  build_cmd+=(--mksquashfs-args "${APPTAINER_SQUASHFS_OPTS}")
fi
build_cmd+=("${out_file}" "${def_file}")
echo "[build_container] command=${build_cmd[*]}"

if ! "${build_cmd[@]}"; then
  echo "[build_container] ERROR: apptainer build failed." >&2
  echo "[build_container] def_file=${def_file}" >&2
  echo "[build_container] out_file=${out_file}" >&2
  echo "[build_container] APPTAINER_TMPDIR=${APPTAINER_TMPDIR}" >&2
  echo "[build_container] APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR}" >&2
  exit 5
fi

if [ ! -f "${out_file}" ]; then
  echo "[build_container] ERROR: build completed but output file is missing: ${out_file}" >&2
  echo "[build_container] Check above build logs for errors." >&2
  exit 3
fi
echo "[build_container] built: $(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${out_file}")"
ls -lh "${out_file}"
