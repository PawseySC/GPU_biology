#!/usr/bin/env python3
"""Generate a controlled Boltz-2 benchmark input set (token ladder + MSA depth ladder).

Two kinds of input are produced:

  synthetic/  Random sequences at exact target token counts, with synthetic MSAs of
              exact depth. Boltz-2 cost is a function of token count, MSA depth and
              sampling parameters only -- the trunk and diffusion stacks have a fixed
              layer count and no early exit -- so synthetic inputs give *identical*
              timing to real targets of the same size, with none of the variance.
              These are for PERFORMANCE ONLY. The confidence scores are meaningless.

  batch/      Directories of N identical t512 monomers, for throughput / multi-GPU
              scaling. Identical inputs are deliberate: they remove load imbalance
              across DDP ranks so parallel efficiency measures the machine, not the
              work distribution.

Real targets for the accuracy/validation tier are NOT generated here -- supply those
yourself with precomputed MSAs (see README.md, "Tier V").

No network access is required or performed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Natural amino-acid background frequencies (UniProt, rounded). Content is irrelevant
# to runtime; using a realistic background just avoids pathological RDKit/tokeniser paths.
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_WEIGHTS = [
    8.25, 1.38, 5.46, 6.72, 3.86, 7.07, 2.27, 5.91, 5.80, 9.65,
    2.41, 4.06, 4.74, 3.93, 5.53, 6.65, 5.36, 6.86, 1.10, 2.92,
]

# Ibuprofen: 15 heavy atoms => 15 ligand tokens. Small, unambiguous, no stereo.
LIGAND_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
LIGAND_TOKENS = 15


def random_sequence(length: int, rng: random.Random) -> str:
    return "".join(rng.choices(AA, weights=AA_WEIGHTS, k=length))


def make_msa(query: str, depth: int, rng: random.Random) -> str:
    """Build a Boltz `key,sequence` MSA CSV of exactly `depth` rows.

    Row 0 is the query itself (matching the convention in the upstream example
    inputs). Remaining rows are the query with ~30% substitutions and ~12% gaps,
    which is representative of a real alignment's column occupancy.
    """
    lines = ["key,sequence", f"0,{query}"]
    for i in range(1, depth):
        chars = []
        for c in query:
            r = rng.random()
            if r < 0.12:
                chars.append("-")
            elif r < 0.42:
                chars.append(rng.choices(AA, weights=AA_WEIGHTS, k=1)[0])
            else:
                chars.append(c)
        lines.append(f"{i}," + "".join(chars))
    return "\n".join(lines) + "\n"


def chain_ids(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def write_case(
    root: Path,
    tokens: int,
    chains: int,
    msa_depth: int,
    ligand: bool,
    rng: random.Random,
) -> tuple[str, int]:
    """Write one YAML (+ its MSA CSVs) and return (name, actual_token_count)."""
    lig_tokens = LIGAND_TOKENS if ligand else 0
    protein_tokens = tokens - lig_tokens
    if protein_tokens < chains * 16:
        msg = f"{tokens} tokens is too few for {chains} chains"
        raise ValueError(msg)

    # Distribute residues across chains, remainder onto the first chain.
    per = protein_tokens // chains
    lengths = [per] * chains
    lengths[0] += protein_tokens - per * chains

    name = f"t{tokens}_c{chains}_m{msa_depth}" + ("_lig" if ligand else "")
    msa_dir = root / "msa" / name
    msa_dir.mkdir(parents=True, exist_ok=True)

    seq_blocks = []
    for cid, length in zip(chain_ids(chains), lengths):
        seq = random_sequence(length, rng)
        if msa_depth <= 1:
            msa_ref = "empty"
        else:
            msa_path = msa_dir / f"{cid}.csv"
            msa_path.write_text(make_msa(seq, msa_depth, rng))
            msa_ref = str(msa_path.resolve())
        seq_blocks.append(
            "  - protein:\n"
            f"      id: {cid}\n"
            f"      sequence: {seq}\n"
            f"      msa: {msa_ref}\n"
        )

    body = "version: 1\nsequences:\n" + "".join(seq_blocks)

    if ligand:
        lig_id = chain_ids(chains + 1)[-1]
        body += f"  - ligand:\n      id: {lig_id}\n      smiles: '{LIGAND_SMILES}'\n"
        body += f"properties:\n  - affinity:\n      binder: {lig_id}\n"

    yaml_dir = root / "synthetic"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    (yaml_dir / f"{name}.yaml").write_text(body)
    return name, sum(lengths) + lig_tokens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output root. Must be on a filesystem visible to the compute "
                         "nodes at the same absolute path (e.g. $MYSCRATCH/boltz-bench/inputs).")
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--tokens", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 1536, 2048, 3072, 4096],
                    help="Token ladder for the size sweep.")
    ap.add_argument("--msa-depths", type=int, nargs="+",
                    default=[1, 256, 1024, 4096, 8192],
                    help="MSA depth ladder (1 == single-sequence mode).")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 64],
                    help="Batch directories to build for throughput/scaling tiers.")
    ap.add_argument("--base-msa", type=int, default=1024,
                    help="MSA depth used for every case outside the MSA-depth ladder.")
    ap.add_argument("--base-tokens", type=int, default=512,
                    help="Token count used for every case outside the token ladder.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)

    index: dict[str, int] = {}

    def emit(tokens: int, chains: int, depth: int, ligand: bool = False) -> None:
        name = f"t{tokens}_c{chains}_m{depth}" + ("_lig" if ligand else "")
        if name in index:
            return
        _, actual = write_case(root, tokens, chains, depth, ligand, rng)
        index[name] = actual

    # Token ladder (single chain, fixed MSA depth).
    for t in args.tokens:
        emit(t, 1, args.base_msa)

    # MSA-depth ladder at the baseline size.
    for d in args.msa_depths:
        emit(args.base_tokens, 1, d)

    # Chain-count axis at a fixed total token budget: isolates MSA pairing and
    # cross-chain attention cost from raw token count.
    for c in (1, 2, 4):
        emit(1024, c, args.base_msa)

    # Affinity axis: same protein, with and without a ligand + affinity request.
    emit(args.base_tokens, 1, args.base_msa, ligand=True)

    # Batch directories for throughput and multi-GPU scaling.
    base_name = f"t{args.base_tokens}_c1_m{args.base_msa}"
    base_yaml = (root / "synthetic" / f"{base_name}.yaml").read_text()
    for n in args.batch_sizes:
        bdir = root / "batch" / f"n{n}"
        bdir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (bdir / f"{base_name}_{i:03d}.yaml").write_text(base_yaml)

    (root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {len(index)} cases to {root / 'synthetic'}")
    for name in sorted(index, key=lambda k: index[k]):
        print(f"  {name:<28} {index[name]:>6} tokens")
    print(f"Batch dirs: {', '.join('n' + str(n) for n in args.batch_sizes)} "
          f"under {root / 'batch'}")


if __name__ == "__main__":
    main()
