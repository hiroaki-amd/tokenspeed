#!/usr/bin/env bash
# Build the reproduction image if needed, then run a command inside it.
#
#   ./docker/run.sh                       # interactive shell
#   ./docker/run.sh bash scripts/all.sh   # the whole sweep
#
# The kernel is baked into the image from this checkout (see ../docker/Dockerfile),
# so a rebuild is needed after changing tokenspeed-kernel-amd, but not after
# editing a script under scripts/, which stays bind-mounted.
#
# Environment:
#   RULER_DATA   path to the RULER data directory (required for the RULER runs)
#   HF_HOME      Hugging Face cache to reuse; defaults to ~/.cache/huggingface
#   IMAGE        image tag to build/use

set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$BUNDLE_DIR/.." && pwd)"
IMAGE="${IMAGE:-blasst-repro:rocm7.2}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "building $IMAGE (first run only, several minutes)..."
    docker build -t "$IMAGE" -f "$BUNDLE_DIR/docker/Dockerfile" "$REPO_ROOT"
fi

# The container runs as the invoking user (see --user below) so that results/
# comes out owned by them rather than by root. That user has no entry in the
# image's /etc/passwd and therefore no home, so point HOME somewhere writable;
# Triton in particular wants a cache directory and fails hard without one.
mkdir -p "$BUNDLE_DIR/.home/triton" "$BUNDLE_DIR/results"

# That user also has no /etc/passwd entry, and several libraries call
# getpass.getuser() (which is getpwuid, not $USER) during import, so they raise
# before anything runs. Give them an entry to find rather than chasing each
# library with an environment variable. Built from the image's own passwd, not
# the host's: the host's would leak its account list into a shared bundle.
passwd_file="$BUNDLE_DIR/.home/passwd"
{
    docker run --rm --entrypoint cat "$IMAGE" /etc/passwd
    echo "repro:x:$(id -u):$(id -g):repro:/work/bundle/.home:/bin/bash"
} > "$passwd_file"

mounts=(
    -v "$passwd_file:/etc/passwd:ro"
    -v "$BUNDLE_DIR:/work/bundle"
    -v "$HF_HOME:/hf"
)
if [[ -n "${RULER_DATA:-}" ]]; then
    mounts+=(-v "$(cd "$RULER_DATA" && pwd):/work/ruler:ro")
fi

# --device plus group membership is what gives the container the GPU. Resolve
# the groups to numeric GIDs on the host: the container image has no "render"
# group of its own, so passing the name would fail, and the kernel checks the
# numeric GID anyway.
VIDEO_GID="$(stat -c '%g' /dev/dri/card0 2>/dev/null || getent group video | cut -d: -f3)"
RENDER_GID="$(stat -c '%g' /dev/kfd 2>/dev/null || getent group render | cut -d: -f3)"

# -t only when there really is a terminal, so the bundle also runs from CI or
# from a pipe.
tty_args=()
if [[ -t 0 && -t 1 ]]; then
    tty_args=(-it)
fi

# --ipc=host and the seccomp opt-out are the standard ROCm PyTorch settings;
# without them large allocations and HSA queue creation fail.
exec docker run --rm "${tty_args[@]}" \
    --device=/dev/kfd --device=/dev/dri \
    --user "$(id -u):$(id -g)" \
    --group-add "$VIDEO_GID" --group-add "$RENDER_GID" \
    --ipc=host --shm-size=16G \
    --security-opt seccomp=unconfined \
    -e HOME=/work/bundle/.home \
    -e TRITON_CACHE_DIR=/work/bundle/.home/triton \
    -e TORCHINDUCTOR_CACHE_DIR=/work/bundle/.home/inductor \
    -e USER=repro -e LOGNAME=repro \
    -e HF_HOME=/hf \
    -e RULER_DATA="${RULER_DATA:+/work/ruler}" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    "${mounts[@]}" \
    -w /work/bundle \
    "$IMAGE" \
    "${@:-bash}"
