# Boltz-2 benchmark suite — MI250X / ROCm / Setonix

Performance and scalability characterisation for the Boltz-2 containers in this repo.
Targets **AMD MI250X on Setonix** (8 GCDs per node, 64 GiB HBM per GCD) via Singularity.

`--no_kernels` is passed unconditionally — trifast triangular kernels are not usable
on this build, so it is a fixed property of the configuration, not a benchmark axis.

This is a **performance** suite. For a functional check that the container works on
every GCD, use the existing smoke test in
[`../v2.2.1/testing.MD`](../v2.2.1/testing.MD) — the two are complementary, and the
launch model used there (`indep`, below) is one of the configurations benchmarked here.

## What this measures, and what it can't

Boltz's `--devices N` is Lightning DDP **over the input list**. It shards structures
across GPUs; it does **not** shard a single structure. So:

* There is no strong scaling on one prediction. Intra-structure scaling needs context
  parallelism ([Fold-CP](https://arxiv.org/pdf/2603.14806)), which is CUDA-only.
* The scalability result here is **throughput** — predictions/hour vs GCDs and nodes —
  plus **maximum tokens per GCD** as the capability ceiling.

State this in any writeup. It is the first question a reviewer will ask.

## Quick start

```bash
export BENCH_ROOT=$MYSCRATCH/boltz-bench
export BENCH_INPUTS_ROOT=$BENCH_ROOT/inputs
export BENCH_CACHE=/scratch/references/boltz

mkdir -p $BENCH_ROOT && cp *.py *.sh bench_array.sbatch $BENCH_ROOT/
cd $BENCH_ROOT

python3 gen_inputs.py --out $BENCH_INPUTS_ROOT

python3 gen_manifest.py \
  --inputs  $BENCH_INPUTS_ROOT \
  --out     $BENCH_ROOT/manifest.tsv \
  --image   /path/to/boltz2_v2.2.1_rocm6.4.1.sif \
  --image-b /path/to/boltz2_v2.2.1_rocm7.2.3.sif \
  --tiers all --repeats 3

./submit.sh $BENCH_ROOT/manifest.tsv 8      # 8 = max concurrent array tasks
```

Then, once the queue drains:

```bash
python3 collect_results.py --root $BENCH_ROOT --manifest $BENCH_ROOT/manifest.tsv --out $BENCH_ROOT/results.csv
python3 plot_results.py --results $BENCH_ROOT/results.csv --outdir $BENCH_ROOT/figs
```

Run the smoke tier first — it is one short job and it catches every path, module and
cache problem before you commit a 163-job campaign:

```bash
python3 gen_manifest.py --inputs $BENCH_INPUTS_ROOT --out smoke.tsv --image <sif> --tiers smoke && ./submit.sh smoke.tsv
```

## Tiers

Each tier varies **one** axis from a fixed baseline (512 tokens, 3 recycling steps,
200 sampling steps, 1 diffusion sample, MSA depth 1024, no affinity, 1 GCD).

| Tier | Axis | Values | What it tells you |
|---|---|---|---|
| `smoke` | — | 1 tiny run | Plumbing works |
| `tokens` | token count | 128 → 4096 | **Headline curve** + the OOM ceiling per GCD |
| `recycle` | `recycling_steps` | 1, 3, 5, 10 | Cost of the upstream eval standard (10) |
| `samples` | `diffusion_samples` | 1, 5, 25 | Cost of eval standard (5) and AF3 default (25) |
| `steps` | `sampling_steps` | 50, 100, 200 | Diffusion cost, roughly linear |
| `msa` | MSA depth | 1, 256, 1024, 4096, 8192 | Cheapest speed lever; 1 = single-sequence |
| `chains` | chain count | 1, 2, 4 @ 1024 tok | Isolates pairing cost from token count |
| `affinity` | affinity head | off, on | Second model pass; expect roughly 2× |
| `potent` | `--use_potentials` | off, on | Inference-time potentials overhead |
| `cpu` | workers × threads | 2/8 × 8/16 | Preprocessing bound on short targets |
| `image` | container | ROCm 6.4.1 vs 7.2.3 | **Publishable**: does the ROCm bump pay? |
| `batch` | structures/job | 1, 8, 64 | Fixed-overhead amortisation |
| `scale1` | GCDs × launch mode | 1, 2, 4, 8 × `ddp`/`indep` | Intra-node throughput scaling |
| `scaleN` | nodes × 8 GCDs | 1, 2, 4 | Inter-node scaling, Lustre contention |

### Two launch models

`scale1` measures both ways of using N GCDs, because they are genuinely different:

* **`ddp`** — one process; Lightning DDP shards the input list across N GCDs.
* **`indep`** — N independent single-GCD processes, each pinned with
  `HIP_VISIBLE_DEVICES`, each `--devices 1`, over a disjoint shard of the inputs.

`indep` is the model the container's own GCD test already uses. It skips DDP
collectives and process-group setup entirely, and each worker keeps its own caches,
so it is likely the better configuration for embarrassingly-parallel screening — but
that is a hypothesis, so the suite measures it rather than assuming it. Set
`BENCH_LAUNCH_MODE` to force one for an ad-hoc run.

Default `--tiers` omits `image`, `cpu`, `scale1` and `scaleN`. Use `--tiers all` for
the full campaign, or name tiers individually.

## Inputs

`gen_inputs.py` builds **synthetic** proteins at exact token counts with synthetic MSAs
of exact depth. Boltz-2's trunk and diffusion stacks have a fixed layer count and no
early exit, so runtime depends only on token count, MSA depth and sampling parameters —
a synthetic 1024-token monomer times identically to a real one, with none of the
variance. **These are for performance only; their confidence scores are meaningless.**

No network access is used. Never point a benchmark at `--use_msa_server`: it makes
results unreproducible and hammers a public API from an HPC allocation.

### Tier V — accuracy validation (do this too)

Performance numbers alone can't distinguish "fast" from "fast and wrong". Separately:

1. Take 20–30 real targets with precomputed MSAs — the Boltz 541-target PDB holdout or
   [PoseBench](https://github.com/BioinfoMachineLearning/PoseBench)'s preprocessed
   PoseBusters / CASP15 sets.
2. Run them at the upstream comparison settings — `--recycling_steps 10
   --sampling_steps 200 --diffusion_samples 5` — with `BENCH_KEEP_STRUCTURES=1`.
3. Compare pLDDT / ipTM / PAE distributions against the same targets on any NVIDIA GPU
   you can borrow, and score structures with OpenStructure 2.8 as upstream does.

That comparison is what makes the ROCm port defensible.

## Methodology notes

**Repeats and statistics.** Three repeats minimum; `plot_results.py` reports the
**median** with a min/max band, never the mean. One contended or cold-cache run wrecks
a 3-sample mean, and the band shows the reader how noisy the measurement actually was.

**Warm-up.** The first invocation on a node pays MIOpen/Triton JIT and autotuning.
`run_one.sh` pins those caches to node-local storage so *every* run pays the same cost
rather than the first one subsidising the rest. Discard the smoke run.

**Phase split.** `bench_wrap.py` runs inside the container and attaches a Lightning
callback (by patching `Trainer.__init__`, since Boltz builds its own Trainer) to
separate import → setup → inference. Reporting only end-to-end wall time makes small
targets look terrible: at 128 tokens the fixed cost dominates completely. Both numbers
are recorded; `fig_overhead.png` is the argument for batching.

**Memory.** `torch.cuda.max_memory_allocated()` from inside the process is the
authoritative peak. `rocm-smi`/`amd-smi` see the whole GCD including HIP context
overhead and any co-tenant, so they over-report — both are recorded and plotted.

**Energy.** `gpu_monitor.py` samples power at 1 Hz on the host and
`collect_results.py` integrates it trapezoidally into J/prediction. Almost nobody
publishes this, and it is a genuinely good look for MI250X.

**Failures are data.** OOM runs are kept in `results.csv` with `status=oom`, not
dropped. The token at which a single GCD runs out is a headline capability number;
silently dropping those rows would misreport the ladder.

## Setonix / ROCm gotchas encoded here

* **Compute nodes have no outbound internet.** An unpopulated weights cache does not
  download slowly, it hangs or dies mid-sweep. `run_one.sh` preflights for a `*.ckpt`
  under `BENCH_CACHE` and exits 3 with an explanation rather than burning the array.
  Pre-stage the cache (see `testing.MD`).
* **MIOpen, Triton and numba caches default to `$HOME`** (or a read-only path inside
  the `.sif`). On Lustre with a large array that is slow and a metadata-server hazard,
  and it pollutes timings — the first run pays for the tuning DB and the rest
  free-ride. `run_one.sh` redirects all of them, plus `MPLCONFIGDIR` and `TMPDIR`, to
  per-process node-local scratch so every run pays the same cost.
* **`WANDB_MODE=disabled`.** Boltz depends on wandb; an accidental network call on an
  internet-less compute node stalls the run and corrupts the measurement.
* **Module is `singularity/3.11.4-nompi`.** Nothing here uses MPI — each boltz process
  is independent. Override with `SINGULARITY_MODULE`.
* **Lightning + SLURM.** Lightning auto-detects SLURM and then expects one task per
  device. We run one task per *node* and let Lightning spawn per-GCD workers, so
  `run_one.sh` sets `SLURM_JOB_NAME=bash` inside the container for multi-GCD runs —
  the documented escape hatch from its SLURM environment plugin.
* **Mixed resource shapes.** A SLURM array shares one `--gres`/`--nodes`. `submit.sh`
  splits the manifest by shape and submits one array per shape. Do not `sbatch
  bench_array.sbatch` directly.
* **SLURM does not expand shell variables in `#SBATCH` directives.** A line like
  `#SBATCH --account=${PAWSEY_PROJECT}-gpu` is taken literally and the job is rejected
  — worth checking in any existing job scripts. `submit.sh` passes `--account` on the
  command line instead, where the shell does expand it.
* **CPU binding.** `--cpus-per-task` is set to `8 × devices`, matching the Setonix GPU
  node layout (64 cores / 8 GCDs). Confirm against current Pawsey guidance.
* **`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`** is on by default; it usually
  shifts the OOM ceiling upward. Override with `BENCH_ALLOC_CONF` to measure the delta.

## Environment overrides

| Variable | Default | Effect |
|---|---|---|
| `BENCH_ROOT` | — | Working tree (scripts, results, logs). Required. |
| `BENCH_INPUTS_ROOT` | — | Input root, bind-mounted into the container. Required. |
| `BENCH_CACHE` | `/scratch/references/boltz` | Boltz weight/CCD cache |
| `BENCH_ACCOUNT` | `${PAWSEY_PROJECT}-gpu` | SLURM account |
| `BENCH_PARTITION` | `gpu` | SLURM partition |
| `SINGULARITY_MODULE` | `singularity/3.11.4-nompi` | Module to load |
| `BENCH_KEEP_STRUCTURES` | `0` | Keep `.cif`/`.npz`. Set `1` for Tier V. |
| `BENCH_LAUNCH_MODE` | `ddp` | `ddp` or `indep` (see above) |
| `BENCH_MONITOR` | `1` | Power/utilisation sampling |
| `BENCH_MONITOR_INTERVAL` | `1.0` | Sampler period, seconds |
| `BENCH_ALLOC_CONF` | `expandable_segments:True` | `PYTORCH_HIP_ALLOC_CONF` |

## Files

| File | Role |
|---|---|
| `gen_inputs.py` | Build the synthetic token / MSA-depth ladder and batch dirs |
| `gen_manifest.py` | Expand tiers into a TSV manifest; validates input paths exist |
| `submit.sh` | Split manifest by resource shape, submit one SLURM array per shape |
| `bench_array.sbatch` | Array driver: manifest row → environment → `srun run_one.sh` |
| `run_one.sh` | One run: caches, sharding, monitor, `singularity exec`, metadata |
| `bench_wrap.py` | In-container: phase timings, torch peak VRAM, OOM classification |
| `gpu_monitor.py` | 1 Hz `amd-smi`/`rocm-smi` sampler → `gpu.csv` |
| `collect_results.py` | Join `meta.json` + `bench.json` + `gpu.csv` → `results.csv` |
| `plot_results.py` | `results.csv` → seven figures |

## Cost

The full `--tiers all --repeats 3` campaign is 163 runs. Dominated by the top of the
token ladder and the 25-sample runs; budget roughly 40–60 GCD-hours, less if you cap
`--tokens` below the OOM ceiling once you know where it is.
