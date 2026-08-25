#!/bin/bash -l
# Stage Tiberius build dependencies into ./vendor before 'podman build -f Dockerfile'.
# Tarballs only: nothing here creates a .git directory inside your repo.

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true
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
trap 'echo; echo "Interrupted." >&2; exit 130' INT TERM

step() { CURRENT_STEP="$1"; printf '\n==> %s\n' "$1"; }

# ---------------------------------------------------------------- preflight
step "checking required tools"
missing=()
for cmd in curl tar sha256sum; do
    command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
done
(( ${#missing[@]} == 0 )) || { echo "Missing commands: ${missing[*]}" >&2; exit 127; }

VENDOR_DIR="${VENDOR_DIR:-${PWD}/vendor}"
CACHE_DIR="${CACHE_DIR:-${VENDOR_DIR}/_archives}"
mkdir -p "$VENDOR_DIR" "$CACHE_DIR"
cd "$VENDOR_DIR"
echo "vendor : ${VENDOR_DIR}"
echo "cache  : ${CACHE_DIR}"

# ---------------------------------------------------------------- helpers
# download <url> <local-filename> [sha256]
download() {
    local url=$1 file=$2 sum=${3:-}
    local path="${CACHE_DIR}/${file}"

    if [[ -f $path && -n $sum ]] && printf '%s  %s\n' "$sum" "$path" | sha256sum --check --status; then
        echo "    cached: ${file}"
        return 0
    fi

    curl --fail --location --silent --show-error \
         --retry 3 --retry-delay 2 --connect-timeout 20 \
         --output "${path}.part" "$url"
    mv -f "${path}.part" "$path"

    if [[ -n $sum ]]; then
        printf '%s  %s\n' "$sum" "$path" | sha256sum --check --status \
            || { echo "checksum mismatch: ${file} (got $(sha256sum "$path" | cut -d' ' -f1))" >&2; return 1; }
    else
        echo "    sha256: $(sha256sum "$path" | cut -d' ' -f1)   # paste this into the pin"
    fi
}

# unpack <local-filename> <dest-dir> [strip-components]
unpack() {
    local file=$1 dest=$2 strip=${3:-1}
    rm -rf "$dest"
    mkdir -p "$dest"
    tar xf "${CACHE_DIR}/${file}" -C "$dest" --strip-components="$strip" --no-same-owner
    if [[ -f "${dest}/.gitmodules" ]]; then
        echo "WARNING: ${dest} declares submodules; source tarballs do not include them." >&2
    fi
}

# gh <owner/repo> <ref> <dest-dir> [sha256]
gh() {
    local repo=$1 ref=$2 dest=$3 sum=${4:-}
    local file="${dest}-${ref}.tar.gz"
    step "fetch ${repo} @ ${ref}"
    download "https://github.com/${repo}/archive/${ref}.tar.gz" "$file" "$sum"
    unpack "$file" "$dest"
}

# rel <url> <dest-dir> [strip] [sha256]
rel() {
    local url=$1 dest=$2 strip=${3:-1} sum=${4:-}
    local file=${url##*/}
    step "fetch ${file}"
    download "$url" "$file" "$sum"
    unpack "$file" "$dest" "$strip"
}

# ---------------------------------------------------------------- source trees
# Pinned to commit SHAs. Re-pin with:
#   git ls-remote https://github.com/OWNER/REPO.git HEAD
gh Gaius-Augustus/Tiberius             v2.0.7                                   Tiberius
gh Gaius-Augustus/Augustus             220e5a63c0fd546563472ec6d1f9271faddaf569 Augustus
gh TransDecoder/TransDecoder           66d47124da1e536db6d6891a0816fb019aa6049c TransDecoder
gh tomasbruna/miniprothint             07cc5abe7fb83d4cbf98a3342e8423beefa81c99 miniprothint
gh lh3/miniprot                        81f9b93481cec16622eff339fc998e4f38da344e miniprot
gh tomasbruna/miniprot-boundary-scorer b0d103ef920dc11a567700c1b3dec022fea8de6d miniprot-boundary-scorer

# ---------------------------------------------------------------- prebuilt binaries
rel https://github.com/gpertea/stringtie/releases/download/v3.0.3/stringtie-3.0.3.Linux_x86_64.tar.gz stringtie
rel https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/3.3.0/sratoolkit.3.3.0-ubuntu64.tar.gz                 sratoolkit
rel https://github.com/gpertea/gffread/releases/download/v0.12.7/gffread-0.12.7.Linux_x86_64.tar.gz   gffread
rel https://github.com/lh3/minimap2/releases/download/v2.30/minimap2-2.30_x64-linux.tar.bz2           minimap2

# ---------------------------------------------------------------- cleanup
if [[ -z "${KEEP_ARCHIVES:-}" ]]; then
    step "removing cached archives"
    rm -rf "$CACHE_DIR"
fi

CURRENT_STEP="done"
printf '\nAll dependencies staged in %s\n' "$VENDOR_DIR"
