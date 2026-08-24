# Testing the Boltz2 ROCm container

This directory ships a self-contained test that confirms the Boltz2 v2.2.1
container (see [`Dockerfile`](Dockerfile)) works correctly on **1–8 GCDs of a
single AMD GPU node**.

A **GCD** (Graphics Compute Die) is one compute die of an AMD MI250X GPU. Each
MI250X exposes **2 GCDs**, and a Pawsey **Setonix GPU node has 4× MI250X = 8
GCDs**, each of which the ROCm stack presents as an independent GPU device. The
test runs one independent `boltz predict` job **per GCD**, pins each job to its
own GCD, and checks that every GCD produces a valid predicted structure.

## Files

| File | Purpose |
|------|---------|
| [`test_boltz_gcd.sh`](test_boltz_gcd.sh) | The test runner. Launches one `boltz predict` per GCD, validates every output, prints a pass/fail summary. Portable — runs anywhere `singularity`/`apptainer` and the GCDs are visible. |
| [`submit_setonix.sbatch`](submit_setonix.sbatch) | SLURM wrapper that requests a **whole Setonix GPU node** (8 GCDs) and runs the 1→8 sweep. |
| [`inputs/test.yaml`](inputs/test.yaml) | Test input: a 4-chain protein complex with **precomputed MSAs** (`inputs/test_*.csv`), so **no MSA server / internet is needed at predict time**. |

## What the test actually checks

For each GCD it verifies **both**:

1. `boltz predict` exits with status `0`, **and**
2. a non-empty predicted structure file (`*.cif` / `*.pdb`) was written under that
   GCD's output directory.

A GCD is `PASS` only if both hold. The script's overall exit status is `0` only
when **every** GCD passed, so it is safe to use in CI or an automated gate.

Before any concurrent runs it does two safety steps:

- **GCD probe** — asks the container `torch.cuda.device_count()` and reports how
  many GCDs it can see (aborts if it sees `0`, clamps the request if it sees
  fewer than asked).
- **Cache warm-up** — if the model-weights cache looks empty, it runs *one*
  prediction first (single process) so the parallel jobs don't race to download
  the same weights. See [Model weights cache](#model-weights-cache).

## Prerequisites

1. **A built container image** (`.sif`). Build the [`Dockerfile`](Dockerfile) and
   convert to Singularity, or pull an existing image, e.g.:
   ```bash
   singularity build boltz2_v2.2.1_rocm6.4.1.sif docker://<your-registry>/boltz2v.2.2.1_rocm6.4.1_ubuntu24.04
   ```
2. **The Boltz2 model weights** in a cache directory (see below).
3. **A GPU allocation** that exposes the GCDs you want to test (on Setonix, a
   whole GPU node = 8 GCDs).

### Model weights cache

Boltz downloads its model checkpoint + CCD data into the `--cache` directory on
first use. The test input already contains its MSAs, so **the cache is the only
thing that needs network access** — and Setonix **compute nodes have no outbound
internet**. Two options:

- **Pre-stage (recommended on Setonix):** point `--cache` at a directory that
  already holds the weights, e.g. the shared `/scratch/references/boltz`.
- **Let the warm-up download them:** run the test somewhere with internet (e.g.
  a login node with the Pawsey proxy, or your workstation) once. The warm-up
  step will populate the cache; subsequent runs reuse it.

## Quick start — Setonix (SLURM)

```bash
cd boltz2/v2.2.1

# Edit submit_setonix.sbatch: set --account, BOLTZ_SIF and BOLTZ_CACHE.
# Then submit (you can also override the account on the command line):
sbatch --account=pawsey0001-gpu submit_setonix.sbatch
```

This requests one full GPU node and runs the **1→8 GCD sweep**. When the job
finishes, check the summary:

```bash
cat $MYSCRATCH/boltz_gcd_test/<jobid>/results.tsv
# and the SLURM log:
cat boltz_gcd_test-<jobid>.out
```

Prefer a single faster all-8 run (still exercises every GCD)? Remove `--sweep`
from `submit_setonix.sbatch` and lower `--time`.

## Quick start — interactive node (`salloc`)

Grab a whole GPU node, load the module, and call the runner directly:

```bash
salloc --account=pawsey0001-gpu --partition=gpu --nodes=1 --gres=gpu:8 \
       --cpus-per-task=64 --time=01:00:00

module load singularity/3.11.4-nompi

cd boltz2/v2.2.1
bash test_boltz_gcd.sh \
    --image  $MYSCRATCH/boltz/boltz2_v2.2.1_rocm6.4.1.sif \
    --cache  /scratch/references/boltz \
    --gcds   8 --sweep
```

## Runner options (`test_boltz_gcd.sh`)

| Flag | Env var | Default | Meaning |
|------|---------|---------|---------|
| `-i, --image PATH` | `BOLTZ_SIF` | — (required) | Container image (`.sif`). |
| `-n, --gcds N` | | `8` | Number of GCDs to test (1–8). |
| `-s, --sweep` | | off | Test **each** level 1,2,…,N (each level run concurrently) instead of one N-GCD run. |
| `-c, --cache DIR` | `BOLTZ_CACHE` | `$MYSCRATCH/boltz/cache` | Boltz model-weights cache. |
| `-o, --outdir DIR` | `BOLTZ_OUTDIR` | `$MYSCRATCH/boltz_gcd_test/<jobid>` | Where outputs, logs and `results.tsv` go. |
| `-y, --yaml FILE` | `BOLTZ_YAML` | `inputs/test.yaml` | Input to fold. |
| `-w, --workdir DIR` | `BOLTZ_WORKDIR` | script dir | Directory containing `inputs/`; used as the in-container working dir so the YAML's relative `msa:` paths resolve. |
| `--quick` | | off | Fewer sampling/recycling steps for a faster functionality-only check. |
| `--kernels` / `--no-kernels` | | `--no-kernels` | The ROCm build ships without the CUDA-only kernels, so `--no_kernels` is the default. |
| `--dry-run` | | off | Print the exact commands and exit — no GPU needed. |
| | `CONTAINER_EXEC` | auto | Force `singularity` or `apptainer`. |

## Interpreting the results

The runner prints a live log and ends with a table (also saved as
`results.tsv` in the output dir):

```
=============================== SUMMARY ==============================
STAGE            GCD  RESULT  TIME(s)   STRUCTURE
sweep_1gcd       0    PASS    412       .../sweep_1gcd/gcd0/boltz_results_test/predictions/test/test_model_0.cif
sweep_2gcd       0    PASS    430       .../sweep_2gcd/gcd0/.../test_model_0.cif
sweep_2gcd       1    PASS    433       .../sweep_2gcd/gcd1/.../test_model_0.cif
...
sweep_8gcd       7    PASS    501       .../sweep_8gcd/gcd7/.../test_model_0.cif
----------------------------------------------------------------------
Runs: 36   Passed: 36   Failed: 0
Outputs & logs under: /scratch/.../boltz_gcd_test/12345678
======================================================================
```

Each GCD's full Boltz output and log live under
`<outdir>/<stage>/gcd<N>/` (the per-run log is `boltz.log`). If a GCD is
`FAIL`, open its `boltz.log` first.

> **Note on timings:** within a stage all GCDs run *concurrently*, so a stage's
> wall-clock time is roughly one prediction, not the sum. In a `--sweep`, later
> stages sharing the node's memory bandwidth may be slightly slower — that is
> expected and does not indicate a failure.

## How GCD pinning works

The batch script (or your `salloc`) hands the runner a node where all 8 GCDs are
visible. For each job the runner sets **`HIP_VISIBLE_DEVICES=<i>`**, which makes
that `boltz predict` process see exactly one GCD (`torch.cuda.device_count() == 1`),
and passes `--devices 1 --accelerator gpu`. Each job also gets its own scratch
dirs (`NUMBA_CACHE_DIR`, `XDG_CACHE_HOME`, `MPLCONFIGDIR`, `TMPDIR`) so the
parallel processes never fight over a shared cache. `WANDB_MODE=disabled`
prevents any accidental network call.

## Running elsewhere (non-Setonix / Docker host)

The runner only needs `singularity` or `apptainer` on `PATH` and the GCDs
visible; it is not Setonix-specific. On a plain ROCm workstation:

```bash
export BOLTZ_SIF=/path/to/boltz2_v2.2.1_rocm6.4.1.sif
bash test_boltz_gcd.sh --gcds 2 --cache $HOME/boltz_cache
```

To sanity-check the commands without a GPU at all:

```bash
bash test_boltz_gcd.sh --image dummy.sif --dry-run
```

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `The container sees 0 GPUs/GCDs` | No GPU allocation, or `/dev/kfd` `/dev/dri` not exposed. Check `--gres=gpu:N`; if devices still aren't visible, add `--rocm` to the `singularity exec` line. |
| Warm-up prediction **FAILED** | Cache is empty **and** the node has no internet. Pre-stage the cache (`--cache /scratch/references/boltz`) or run the warm-up on a node with internet. |
| `Neither 'singularity' nor 'apptainer' … on PATH` | `module load singularity/3.11.4-nompi` (Setonix), or set `CONTAINER_EXEC`. |
| Some GCDs PASS, some FAIL | Open the failing GCD's `boltz.log`. A single bad GCD usually points at a hardware/driver issue on that die; check `rocm-smi`. |
| `Input YAML not found` / MSA errors | Run from `boltz2/v2.2.1` (so `inputs/` resolves), or pass `--workdir` pointing at the directory that contains `inputs/`. |
| Job won't schedule | The whole-node request may be waiting for a free node. Reduce to `--gcds N` without `--sweep`, or check `squeue`. |
| Predictions are slow | Add `--quick` for a functionality-only check (lower sampling/recycling steps). |

## Notes

- The test input uses precomputed MSAs, so results are deterministic in structure
  count — the goal is a **functional / smoke test of every GCD**, not a benchmark
  or an accuracy check.
- To monitor VRAM per GCD while a test runs, see the repo-root
  [`vram_monitoring.sh`](../../vram_monitoring.sh).
