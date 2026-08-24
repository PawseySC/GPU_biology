# AlphaFold2 on ROCm 7.2.3 with Triton Evoformer attention

Variant of [`alphafold2/rocm7.2.3`](../rocm7.2.3/) that carries the AMD-authored
Triton flash-attention kernel for the Evoformer (`triton-attention.patch`).

The kernel fuses QK<sup>T</sup> / pair-bias / mask / softmax / PV into one pass,
so the pair bias and residue mask never materialise as `[heads, N, N]`
intermediates, and it lets the three call sites
(`MSARowAttentionWithPairBias`, `MSAColumnAttention`, `TriangleAttention`)
skip `mapping.inference_subbatch` at inference time — which is where most of
the speedup comes from.

**It is opt-in and inert by default.** `alphafold/model/triton/__init__.py`
sets `USE_TRITON` only when `triton` *and* `jax_triton` both import **and**
`AF2_USE_TRITON=1` is in the environment. Otherwise every path is the original
XLA one, so this image is a drop-in replacement for
`alphafold2-amd-gpu:v2.3.2_rocm7.2.3`.

## Why a separate recipe

The upstream patch was cut against alphafold **`main`** (the `modules.py` blob
hash in the patch, `1104236`, matches `main` exactly), not the `v2.3.2` tag that
both existing recipes clone. `main` is 96 commits ahead of `v2.3.2` and its
`modules.py` has been reformatted — `key_dim ** (-0.5)` vs `key_dim**(-0.5)`,
multi-line `inference_subbatch(...)` calls — so `git apply` fails on all four
`modules.py` hunks (the new files apply fine).

`triton-attention-v2.3.2.patch` in this directory is that patch rebased onto
`v2.3.2`. The rebase is textual only; the five hunks are identical in meaning.
It has been verified to apply cleanly to both:

- the `v2.3.2` tag (`3f31725`) — what `rocm7.2.3/Dockerfile` clones
- `e9b6848` — what `rocm6.2.4_proteinfold/Dockerfile` pins

The second reason for a separate recipe is the dependency chain:
`jax-triton 0.3.1` requires `jax>=0.8.2`, which requires `numpy>=2.0` and
`scipy>=1.13`. That is a real upgrade to the working ROCm 7.2.3 image
(JAX 0.7.1, numpy 1.26.4), so it lives here rather than being forced on it.

## Version matrix

| Component | `rocm7.2.3` | this recipe | why |
|---|---|---|---|
| JAX / jaxlib | 0.7.1 | **0.8.2** | `jax-triton 0.3.1` requires `jax>=0.8.2` |
| ROCm JAX wheels | `ROCm/jax` @ `rocm-jax-v0.7.1` | `ROCm/rocm-jax` @ `rocm-jax-v0.8.2`, `+rocm7.2.0` | repo moved for the 0.8.x line |
| numpy | 1.26.4 | **2.0.2** | jax 0.8.2 needs `>=2.0`; tensorflow-cpu 2.18.1 needs `<2.1.0` |
| scipy | >=1.12 | **>=1.13** | jax 0.8.2 requirement |
| triton | — | **3.6.0** | PyPI wheel bundles the amdgcn backend; no `pytorch-triton-rocm` needed |
| jax-triton | — | **0.3.1** | installed `--no-deps` |
| Python | 3.11 | 3.11 | `cp311` wheels exist for all four ROCm JAX components |

> The patch author's `scripts/setup_triton_env.sh` uses the `+rocm7.1.1` wheels
> and Python 3.12. This recipe uses `+rocm7.2.0` / `cp311` to match the ROCm
> 7.2.3 base image and the rest of the repo. Fall back with
> `--build-arg ROCM_JAX_VARIANT=rocm7.1.1` if the 7.2.0 build misbehaves —
> that is the combination AMD validated.

## Build

Needs the same ROCm MPICH base as `rocm7.2.3` — see
[`../rocm7.2.3/README.md`](../rocm7.2.3/README.md) for pulling or building it.

```bash
docker build -f alphafold2/rocm7.2.3_triton/Dockerfile -t alphafold2-amd-gpu:v2.3.2_rocm7.2.3_triton alphafold2/rocm7.2.3_triton/
```

Build arguments:

| Argument | Default | Meaning |
|---|---|---|
| `BASE_IMAGE` | `rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04` | ROCm MPICH base image |
| `ROCM_JAX_VARIANT` | `rocm7.2.0` | ROCm build of the JAX wheels; `rocm7.1.1` is AMD's validated combo |
| `ROCM_JAX_TAG` / `JAX_ROCM_VER` | `rocm-jax-v0.8.2` / `0.8.2` | ROCm JAX release |
| `TRITON_VER` / `JAX_TRITON_VER` | `3.6.0` / `0.3.1` | Triton stack |
| `INSTALL_PYMOL` | `0` | as in `rocm7.2.3` |

The build only proves the *fallback* path — `docker build` has no GPU, so it
checks that AlphaFold imports, that `triton`/`jax_triton` import, and that
`USE_TRITON` is off without the env var.

## GO / NO-GO on hardware

Run these on a node with an AMD GPU, from `/app/alphafold` (so
`tests.triton_tests.kernels` resolves), cheapest first:

```bash
python3 -m pytest tests/triton_tests/test_smoke.py -v
```

```bash
python3 -m pytest tests/triton_tests/test_kernel.py -v
```

```bash
AF2_USE_TRITON=1 python3 -m pytest tests/triton_tests/test_integration.py -v
```

`test_smoke.py` is the one that matters: it calls a trivial Triton kernel
through `jax_triton.triton_call`. If that fails, nothing else will — jax-triton
on ROCm is the main unproven link in this stack, since the ROCm `jaxlib` has to
expose the Triton custom call that `jax_triton` dispatches to. All three tests
skip cleanly rather than failing when Triton is unavailable, so **check for
`skipped` in the output, not just a green run.**

## Run

```bash
docker run --rm -it -e AF2_USE_TRITON=1 ...your gpu/rocm and mount flags... alphafold2-amd-gpu:v2.3.2_rocm7.2.3_triton python3 run_alphafold.py ...
```

Everything else — databases, `scripts/alphafold2_docker_run.sh`, bind mounts —
works as in `rocm7.2.3`. Set `ALPHAFOLD2_IMAGE` to this tag when using the host
helper script.

Benchmark with `AF2_USE_TRITON=1` against `AF2_USE_TRITON=0` on the *same
image*: that isolates the kernel from the JAX 0.7.1 → 0.8.2 change.

## Known caveats

- **`config.py` sets `bfloat16: False`** (inherited unchanged from
  `rocm7.2.3`), so q/k/v reach the kernel as fp32. The kernel is correct in
  fp32 but the fused `tl.dot` gains most on bf16. If you want to try bf16, flip
  it in `config.py` and re-validate numerics — don't assume it is safe, the
  flag was turned off deliberately.
- **`TRITON_CACHE_DIR=/tmp/triton_cache`** is set in the image because Triton's
  default (`~/.triton`) is not writable when the container runs under a
  non-root uid. On Setonix, point it at node-local scratch if `/tmp` is small.
- **First call is slow** — Triton JIT-compiles per shape. AF2's shapes vary
  with sequence length, so expect compilation stalls on the first pass of each
  new target.
- `XLA_FLAGS` still contains `--xla_gpu_enable_triton_gemm=false`. That
  disables XLA's *internal* Triton GEMM emitter and is unrelated to the
  jax-triton custom call used here — leave it as is.
- The patch also installs `scripts/setup_triton_env.sh` (a host venv builder).
  It is unused inside the container; harmless.

## Not applicable: `rocm6.2.4_proteinfold`

The patch cannot be applied to [`../rocm6.2.4_proteinfold`](../rocm6.2.4_proteinfold/)
as it stands. That image is JAX 0.4.34 on ROCm 6.2.4 via the `jax-rocm60-*`
wheel line; `jax-triton 0.3.1` needs `jax>=0.8.2`, and the 0.8.x ROCm JAX
release ships **only** `jax_rocm7_*` wheels — there is no ROCm 6 build. Getting
Triton into a ProteinFold-compatible image means rebasing that recipe onto
ROCm 7 first: take this Dockerfile and add the ProteinFold pieces
(`run_msa.py`, `run_predict.py`, the `COPY ... /opt/docker-recipes/` block and
the image labels) from `rocm6.2.4_proteinfold/Dockerfile`. The AlphaFold commit
it pins (`e9b6848`) already accepts the rebased patch, so only the JAX/ROCm
stack has to move.
