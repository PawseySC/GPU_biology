#!/usr/bin/env python3
"""Build the run manifest (one TSV row per benchmark run) consumed by the SLURM array.

Each tier varies ONE axis from a fixed baseline so that a difference in the result
is attributable to that axis. Tiers are selected with --tiers; the default set is
everything that fits on a single GCD.

  smoke     1 run, tiny, ~2 min. Proves the container, cache and paths work.
  tokens    Token-count ladder. The headline curve, and where the OOM ceiling is found.
  recycle   recycling_steps 1/3/5/10 (10 == the upstream eval standard).
  samples   diffusion_samples 1/5/25 (5 == upstream eval standard, 25 == AF3 default).
  steps     sampling_steps 50/100/200.
  msa       MSA depth 1/256/1024/4096/8192 (1 == single-sequence mode).
  chains    Same total tokens split over 1/2/4 chains.
  affinity  Structure only vs structure+affinity.
  potent    --use_potentials off/on.
  cpu       num_workers x preprocessing-threads.
  image     Baseline ladder subset re-run on a second container (ROCm 6.4.1 vs 7.2.3).
  batch     1 / 8 / 64 structures per job on one GCD: fixed-overhead amortisation.
  scale1    1/2/4/8 GCDs on one node over a fixed 64-structure batch.
  scaleN    1/2/4 nodes x 8 GCDs. Requires --image-b unset; uses the primary image.

`--no_kernels` is always passed (hard requirement on this build) and is therefore
not an axis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

COLUMNS = [
    "run_id", "tier", "axis", "axis_value", "input", "batch_n", "tokens",
    "recycling_steps", "sampling_steps", "diffusion_samples", "max_parallel_samples",
    "affinity", "use_potentials", "devices", "nodes", "launch_mode", "num_workers",
    "preproc_threads", "image", "repeat",
]

# Baseline. Every tier overrides exactly one field (or one tightly coupled pair).
BASE = {
    "batch_n": 1,
    "recycling_steps": 3,
    "sampling_steps": 200,
    "diffusion_samples": 1,
    "max_parallel_samples": 5,
    "affinity": 0,
    "use_potentials": 0,
    "devices": 1,
    "nodes": 1,
    "launch_mode": "ddp",
    "num_workers": 2,
    "preproc_threads": 8,
    "repeat": 0,
}

ALL_TIERS = ["smoke", "tokens", "recycle", "samples", "steps", "msa", "chains",
             "affinity", "potent", "cpu", "image", "batch", "scale1", "scaleN"]
DEFAULT_TIERS = ["smoke", "tokens", "recycle", "samples", "steps", "msa", "chains",
                 "affinity", "potent", "batch"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", type=Path, required=True,
                    help="Input root created by gen_inputs.py.")
    ap.add_argument("--out", type=Path, required=True, help="Manifest TSV to write.")
    ap.add_argument("--image", required=True,
                    help="Primary container .sif (absolute path).")
    ap.add_argument("--image-b", default="",
                    help="Second container for the 'image' tier, e.g. the ROCm 7.2.3 build.")
    ap.add_argument("--tiers", nargs="+", default=DEFAULT_TIERS,
                    choices=[*ALL_TIERS, "all"])
    ap.add_argument("--repeats", type=int, default=3,
                    help="Repeats per configuration. Report the median; 3 is the minimum "
                         "that lets you see a flyer.")
    ap.add_argument("--base-tokens", type=int, default=512)
    ap.add_argument("--base-msa", type=int, default=1024)
    ap.add_argument("--tokens", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 1536, 2048, 3072, 4096])
    ap.add_argument("--msa-depths", type=int, nargs="+",
                    default=[1, 256, 1024, 4096, 8192])
    args = ap.parse_args()

    tiers = ALL_TIERS if "all" in args.tiers else args.tiers
    if "image" in tiers and not args.image_b:
        sys.stderr.write("error: --image-b is required for the 'image' tier\n")
        return 2

    syn = (args.inputs / "synthetic").resolve()
    bat = (args.inputs / "batch").resolve()

    def case(tokens: int, chains: int = 1, msa: int | None = None,
             ligand: bool = False) -> Path:
        depth = args.base_msa if msa is None else msa
        name = f"t{tokens}_c{chains}_m{depth}" + ("_lig" if ligand else "")
        return syn / f"{name}.yaml"

    rows: list[dict] = []

    def add(tier: str, axis: str, axis_value: object, inp: Path,
            tokens: int, reps: int | None = None, **over: object) -> None:
        n = args.repeats if reps is None else reps
        for r in range(n):
            row = dict(BASE)
            row.update(over)
            row.update({
                "tier": tier, "axis": axis, "axis_value": axis_value,
                "input": str(inp), "tokens": tokens, "repeat": r,
                "image": over.get("image", args.image),
            })
            row["run_id"] = (
                f"{tier}__{axis}-{axis_value}__t{tokens}"
                f"__d{row['devices']}n{row['nodes']}{row['launch_mode']}__r{r}"
            )
            rows.append(row)

    bt, bm = args.base_tokens, args.base_msa

    if "smoke" in tiers:
        add("smoke", "none", "-", case(128), 128, reps=1,
            recycling_steps=1, sampling_steps=50)

    if "tokens" in tiers:
        for t in args.tokens:
            add("tokens", "tokens", t, case(t), t)

    if "recycle" in tiers:
        for v in (1, 3, 5, 10):
            add("recycle", "recycling_steps", v, case(bt), bt, recycling_steps=v)

    if "samples" in tiers:
        for v in (1, 5, 25):
            add("samples", "diffusion_samples", v, case(bt), bt, diffusion_samples=v)

    if "steps" in tiers:
        for v in (50, 100, 200):
            add("steps", "sampling_steps", v, case(bt), bt, sampling_steps=v)

    if "msa" in tiers:
        for v in args.msa_depths:
            add("msa", "msa_depth", v, case(bt, msa=v), bt)

    if "chains" in tiers:
        for c in (1, 2, 4):
            add("chains", "chains", c, case(1024, chains=c), 1024)

    if "affinity" in tiers:
        # Both cases are `bt` tokens in total; the ligand case spends 15 of them on
        # the ligand's heavy atoms rather than on residues, so the comparison is
        # like-for-like on token count.
        add("affinity", "affinity", 0, case(bt), bt)
        add("affinity", "affinity", 1, case(bt, ligand=True), bt, affinity=1)

    if "potent" in tiers:
        for v in (0, 1):
            add("potent", "use_potentials", v, case(bt), bt, use_potentials=v)

    if "cpu" in tiers:
        for nw in (2, 8):
            for pt in (8, 16):
                add("cpu", "workers_threads", f"{nw}x{pt}", case(bt), bt,
                    num_workers=nw, preproc_threads=pt)

    if "image" in tiers:
        for t in (256, 1024, 2048):
            for label, img in (("A", args.image), ("B", args.image_b)):
                add("image", "image", label, case(t), t, image=img)

    if "batch" in tiers:
        for n in (1, 8, 64):
            add("batch", "batch_n", n, bat / f"n{n}", bt, batch_n=n)

    if "scale1" in tiers:
        # Two launch models, because they are genuinely different machines:
        #   ddp   - one process, Lightning DDP shards the input list over N GCDs
        #   indep - N independent single-GCD processes pinned via HIP_VISIBLE_DEVICES
        # `indep` is what the container's own GCD test uses and it skips DDP
        # collectives entirely. Measure both rather than assuming.
        for mode in ("ddp", "indep"):
            for d in (1, 2, 4, 8):
                add("scale1", "devices", d, bat / "n64", bt, batch_n=64,
                    devices=d, launch_mode=mode)

    if "scaleN" in tiers:
        for n in (1, 2, 4):
            add("scaleN", "nodes", n, bat / "n64", bt, batch_n=64,
                devices=8, nodes=n)

    # Fail before submission, not 151 array tasks later. A tier can reference a case
    # gen_inputs.py was not asked to build (mismatched --tokens / --msa-depths).
    missing = sorted({r["input"] for r in rows if not Path(r["input"]).exists()})
    if missing:
        sys.stderr.write(
            f"error: {len(missing)} input path(s) do not exist. Re-run gen_inputs.py "
            f"with matching --tokens/--msa-depths/--batch-sizes:\n"
        )
        for path in missing[:10]:
            sys.stderr.write(f"  {path}\n")
        if len(missing) > 10:
            sys.stderr.write(f"  ... and {len(missing) - 10} more\n")
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")

    by_tier: dict[str, int] = {}
    for row in rows:
        by_tier[row["tier"]] = by_tier.get(row["tier"], 0) + 1
    print(f"Wrote {len(rows)} runs to {args.out}")
    for tier in tiers:
        if tier in by_tier:
            print(f"  {tier:<10} {by_tier[tier]:>4} runs")
    # submit.sh, not sbatch: rows have mixed resource shapes and must be split into
    # one array per (nodes, devices) before submission.
    print(f"\nSubmit with:  ./submit.sh {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
