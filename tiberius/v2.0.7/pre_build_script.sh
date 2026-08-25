#!/bin/bash -l
# Download Tiberius build dependencies into ./vendor before 'podman build -f Dockerfile'.
# Pulling outside the build context is faster; compilation still happens in the container.

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true   # bash >= 4.4: propagate errexit into $(...)

[[ -n "${TRACE:-}" ]] && set -x

# ---------------------------------------------------------------- error reporting
CURRENT_STEP="startup"

on_error() {
    local exit_code=$?
    local line_no=$1
    {
        echo
        echo "FAILED during step: ${CURRENT_STEP}"
        echo "  command : ${BASH_COMMAND}"
        echo "  line    : ${BASH_SOURCE[0]}:${line_no}"
        echo "  exit    : ${exit_code}"
        echo "  cwd     : ${PWD}"
    } >&2
    exit "${exit_code}"
}
trap 'on_error ${LINENO}' ERR
trap 'CURRENT_STEP="interrupted"; echo; echo "Interrupted." >&2; exit 130' INT TERM

step() {
    CURRENT_STEP="$1"
    printf '\n==> %s\n' "$1"
}

# ---------------------------------------------------------------- preflight
step "checking required tools"
missing=()
for cmd in git curl tar sha256sum; do
    command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
done
if (( ${#missing[@]} )); then
    echo "Missing required commands: ${missing[*]}" >&2
    exit 127
fi

VENDOR_DIR="${VENDOR_DIR:-${PWD}/vendor}"
mkdir -p "$VENDOR_DIR"
cd "$VENDOR_DIR"
echo "vendor directory: ${VENDOR_DIR}"

DOWNLOADS=()

# ---------------------------------------------------------------- helpers
# clone <url> <ref> [dir]
# <ref> may be a tag, a branch, or a full commit SHA. Idempotent: safe to re-run.
clone() {
    local url=$1 ref=$2 dir=${3:-}
    [[ -n $dir ]] || { dir=${url%/}; dir=${dir##*/}; dir=${dir%.git}; }

    step "clone ${dir} @ ${ref}"
    mkdir -p "$dir"
    git -C "$dir" rev-parse --git-dir >/dev/null 2>&1 || git -C "$dir" init -q
    if git -C "$dir" remote get-url origin >/dev/null 2>&1; then
        git -C "$dir" remote set-url origin "$url"
    else
        git -C "$dir" remote add origin "$url"
    fi
    git -C "$dir" fetch --quiet --depth 1 origin "$ref"
    git -C "$dir" checkout --quiet --detach FETCH_HEAD
    git -C "$dir" submodule update --quiet --init --recursive --depth 1
    echo "    at $(git -C "$dir" rev-parse --short HEAD)"
}

# fetch <url> [sha256]
# -f makes curl exit non-zero on HTTP errors instead of saving an error page.
fetch() {
    local url=$1 sum=${2:-}
    local file=${url##*/}

    step "download ${file}"
    curl --fail --location --silent --show-error \
         --retry 3 --retry-delay 2 --connect-timeout 20 \
         --output "${file}.part" "$url"
    mv -f "${file}.part" "$file"

    if [[ -n $sum ]]; then
        echo "    verifying sha256"
        printf '%s  %s\n' "$sum" "$file" | sha256sum --check --status \
            || { echo "checksum mismatch: ${file} (got $(sha256sum "$file" | cut -d' ' -f1))" >&2; return 1; }
    fi
    DOWNLOADS+=("$file")
}

# extract <file>
extract() {
    step "extract ${1}"
    tar xf "$1"        # GNU tar auto-detects gzip/bzip2
}

# ---------------------------------------------------------------- git sources
clone https://github.com/Gaius-Augustus/Tiberius.git v2.0.7
clone https://github.com/Gaius-Augustus/Augustus.git 220e5a63c0fd546563472ec6d1f9271faddaf569
clone https://github.com/TransDecoder/TransDecoder.git 66d47124da1e536db6d6891a0816fb019aa6049c
clone https://github.com/tomasbruna/miniprothint.git 07cc5abe7fb83d4cbf98a3342e8423beefa81c99
clone https://github.com/lh3/miniprot.git 81f9b93481cec16622eff339fc998e4f38da344e
clone https://github.com/tomasbruna/miniprot-boundary-scorer.git b0d103ef920dc11a567700c1b3dec022fea8de6d

# ---------------------------------------------------------------- release tarballs
# Add the sha256 as a second argument to each fetch once you have pinned them.
fetch   https://github.com/gpertea/stringtie/releases/download/v3.0.3/stringtie-3.0.3.Linux_x86_64.tar.gz
extract stringtie-3.0.3.Linux_x86_64.tar.gz

fetch   https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/3.3.0/sratoolkit.3.3.0-ubuntu64.tar.gz
extract sratoolkit.3.3.0-ubuntu64.tar.gz

fetch   https://github.com/gpertea/gffread/releases/download/v0.12.7/gffread-0.12.7.Linux_x86_64.tar.gz
extract gffread-0.12.7.Linux_x86_64.tar.gz

fetch   https://github.com/lh3/minimap2/releases/download/v2.30/minimap2-2.30_x64-linux.tar.bz2
extract minimap2-2.30_x64-linux.tar.bz2

# ---------------------------------------------------------------- cleanup
step "removing downloaded archives"
if (( ${#DOWNLOADS[@]} )); then
    rm -f "${DOWNLOADS[@]}"
fi
rm -f -- *.part

CURRENT_STEP="done"
printf '\nAll dependencies staged in %s\n' "$VENDOR_DIR"
