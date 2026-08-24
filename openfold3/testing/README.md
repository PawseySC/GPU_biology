# OpenFold3 container tests

Two things this answers: **is the container producing correct output**, and
**how fast is it**. It is not a research benchmark — it will not tell you how
OF3 compares to AlphaFold3. See "What this is not" at the bottom.

## Setup

Edit the three settings at the top of [`env.sh`](env.sh):

```bash
CONTAINER=openfold3_rocm.sif       # or docker://quay.io/pawsey/openfold3:...
OPENFOLD_CACHE=/scratch/references/openfold3   # model weights
LOCAL_SCRATCH=/tmp/${USER}/of3_$$              # must be node-local
```

`score.py` and `report_perf.py` need numpy, which a bare compute node will not
necessarily have. Either load a python module, or run them through the
container — `env.sh` defines a helper for that:

```bash
source ./env.sh
of3_python score.py validity results/correctness/hipblaslt0
```

The `./score.py ...` forms below assume you have numpy on the host.

Everything else in `env.sh` is deliberate; read the comments before changing it.
Two settings in particular matter more than they look:

- **`TRITON_CACHE_DIR` must be node-local.** The OF3 AMD path runs Triton
  kernels that are JIT-compiled on first use. Pointed at Lustre, every array
  task compiles into the same directory over a shared filesystem — slow at
  best, corrupted cache entries and spurious failures at worst.
- **`TORCH_BLAS_PREFER_HIPBLASLT` is pinned to 0.** See the A/B below.

## Order to run things

### 1. Determinism — run this first

```bash
sbatch check_determinism.sh
```

Same input, same seed, twice. If this fails, stop: golden-output comparison is
meaningless against a nondeterministic model, and every other test here becomes
noise. A failure is usually a nondeterministic reduction or kernel autotuning
varying between runs, not a wrong answer.

The queries pin `seeds: [42]` at the top of each JSON. That is the field the
predict path reads — `InferenceQuerySet.seeds`. Setting a seed in
`runner_bench.yaml` instead will *not* do what you want.

### 2. Correctness

```bash
sbatch run_correctness.sh
./score.py validity results/correctness/hipblaslt0
```

Eight targets, one modality each: protein monomer, multimer, protein-ligand
(CCD and SMILES), RNA, DNA with modified residues, protein-DNA, protein-RNA.
A Triton kernel port can break one modality and leave the others clean, so the
coverage matters more than the target count.

The targets are small and easy **on purpose**. Held-out post-cutoff data exists
to measure model quality; that is the OF3 team's problem, not yours. Here you
want targets the model should nail, so a bad result is unambiguously the
container rather than a hard target.

`score.py validity` checks backbone bond lengths, steric clashes, and CA
chirality consistency. It does *not* compute LDDT or TM-score, and that is the
point — see below.

### 3. The hipBLASLt A/B

```bash
TORCH_BLAS_PREFER_HIPBLASLT=1 sbatch run_correctness.sh
./score.py compare results/correctness/hipblaslt0 results/correctness/hipblaslt1
```

The OF3-preview2 technical report's errata records hipBLASLt on MI300A
producing structures with *decreased chemical validity*, corrected by falling
back to rocBLAS. Nothing in the OF3 source pins the backend, so it inherits
whatever the PyTorch ROCm build defaults to — which makes this yours to control.

Two things worth being precise about:

- **LDDT and TM-score would not catch this.** They measure positional
  agreement. A structure can score 0.9 LDDT with broken bond lengths and
  inverted stereocentres. That is why the checks here are geometric.
- **The errata is on MI300A; Setonix is gfx90a.** hipBLASLt kernel coverage
  differs by architecture, so it may not reproduce for you at all. Measuring is
  cheap. Assuming is not.

Expect large coordinate differences between backends regardless. What matters
is whether the bond and clash counts get *worse*.

### 4. Freeze golden outputs

Once a build passes 1–3 and you trust it:

```bash
./score.py freeze results/correctness/hipblaslt0
git add golden/ && git commit -m "Freeze OF3 golden outputs"
```

Then every later build is one command:

```bash
./score.py golden results/correctness/hipblaslt0
```

This is far more sensitive than any threshold against experimental structures —
it catches drift a "> 0.8 LDDT" gate would wave straight through. It costs
nothing and it is the check that will actually catch a bad rebuild.

### 5. Performance

```bash
sbatch run_perf.sh
./report_perf.py > PERFORMANCE.md
```

A ubiquitin copy-number ladder (76 → 2432 tokens) plus two mixed-modality
points. Emits a markdown table: wall clock, peak VRAM, GPU utilisation, and
pass/fail per size.

**Expect the top of the ladder to OOM.** That is the point — the ceiling is the
number users actually ask about, so the report keeps failed rows rather than
dropping them.

The mixed points exist because OF3 is all-atom: memory scales on tokens
*including* ligand and nucleic atoms, so a protein-only ladder will not tell
you where a protein-ligand job falls over.

## Measuring the MSA stage

Every query here sets `use_msas: false`, which makes the suite hermetic — no
MSA databases, no network, no alignment cost — so timings are model-only and
golden outputs are stable. That is right for build validation and wrong for
telling a user how long their job will take.

For end-to-end numbers, precompute alignments and time that stage separately:

```bash
run_openfold align_msa_server --query_json=... --output_dir=alignments/
```

then point `main_msa_file_paths` / `paired_msa_file_paths` in the query chains
at the results and flip `use_msas` to true. On a shared filesystem the
alignment stage frequently dominates total runtime — quote it separately rather
than folding it into one number.

## What this is not

- **Not LDDT or TM-score against experimental structures.** Those want
  OpenStructure, which needs its own environment (it conflicts with the OF3
  dependencies — the OF3 team separate them too). That is a second container,
  not something to bolt into this one. The reference harness, if you want to go
  further, is
  [aqlaboratory/openfold-3-benchmarking](https://github.com/aqlaboratory/openfold-3-benchmarking).
- **Not the paper's benchmark suite.** Runs N' Poses, FoldBench and the
  antibody-antigen scaling curve answer "is OF3 as good as AF3". The
  antibody-antigen curve alone is 1,000 MSA seeds per target — research-scale
  compute that tells you nothing about your build.
- **Not run against a real container yet.** The scoring and reporting logic is
  tested against synthetic structures with known-good and known-broken
  geometry, and the CLI flags match `run_openfold predict` as of the current
  `main`. The SLURM plumbing and the actual container invocation are untested
  on Setonix.

## Files

| file | what it does |
|---|---|
| [`env.sh`](env.sh) | shared setup: container, caches, BLAS backend, `of3_predict` |
| [`runner_bench.yaml`](runner_bench.yaml) | pinned inference config — frozen; changing it invalidates golden outputs |
| [`gen_inputs.py`](gen_inputs.py) | regenerates everything under `inputs/` |
| [`check_determinism.sh`](check_determinism.sh) | same input twice, then compare |
| [`run_correctness.sh`](run_correctness.sh) | 8-target modality sweep |
| [`run_perf.sh`](run_perf.sh) | token ladder with VRAM sampling |
| [`monitor_vram.sh`](monitor_vram.sh) | rocm-smi sampler, header-driven CSV |
| [`score.py`](score.py) | validity / determinism / compare / freeze / golden |
| [`report_perf.py`](report_perf.py) | perf results → markdown table |

### A note on `monitor_vram.sh`

It replaces the `rocm-smi | awk '{print $9}'` approach in
[`../../vram_monitoring.sh`](../../vram_monitoring.sh), which is not safe to
reuse here for two reasons:

1. **Positional columns move between rocm-smi releases.** These containers span
   ROCm 6.2 to 7.2, so `$9` is not the same field on every node, and the
   mismatch is silent — you get a plausible number from the wrong column.
   This version looks columns up by header name.
2. **A 60s poll misses the peak.** Diffusion sampling spikes over seconds, so
   the sampled maximum is close to arbitrary. Default here is 2s.

For a true high-water mark, `torch.cuda.max_memory_allocated()` inside the run
beats any external sampler. The sampler is for when you cannot instrument.
