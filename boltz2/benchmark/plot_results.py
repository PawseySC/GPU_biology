#!/usr/bin/env python3
"""Turn results.csv into the figures for the writeup.

Aggregates repeats to the median with a min/max band -- never the mean. A single
cold-cache or contended run skews a 3-sample mean badly, and the band shows the
reader how noisy the measurement was instead of hiding it.

Produces (only for tiers present in the data):
  fig_tokens.png      runtime and peak VRAM vs token count, with the OOM ceiling
  fig_params.png      runtime vs recycling / diffusion samples / sampling steps
  fig_msa.png         runtime vs MSA depth
  fig_overhead.png    where the time goes: import / setup / predict, vs batch size
  fig_scaling.png     throughput and parallel efficiency vs GCDs (and nodes)
  fig_energy.png      energy per structure vs token count
  fig_image.png       ROCm 6.4.1 vs 7.2.3
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.stderr.write("matplotlib is required: pip install matplotlib\n")
    raise SystemExit(1) from None


def load(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def num(row: dict, key: str) -> float | None:
    val = row.get(key)
    if val in (None, "", "None"):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def agg(rows: list[dict], xkey: str, ykey: str) -> tuple[list, list, list, list]:
    """Group by x, return (xs, medians, lows, highs) sorted numerically by x."""
    buckets: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        x, y = num(row, xkey), num(row, ykey)
        if x is not None and y is not None:
            buckets[x].append(y)
    xs = sorted(buckets)
    med, lo, hi = [], [], []
    for x in xs:
        vals = sorted(buckets[x])
        mid = len(vals) // 2
        med.append(vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2)
        lo.append(vals[0])
        hi.append(vals[-1])
    return xs, med, lo, hi


def band(ax, xs, med, lo, hi, label, marker="o") -> None:  # noqa: ANN001
    line, = ax.plot(xs, med, marker=marker, label=label)
    ax.fill_between(xs, lo, hi, alpha=0.18, color=line.get_color())


def ok(rows: list[dict], tier: str) -> list[dict]:
    return [r for r in rows if r["tier"] == tier and r["status"] == "ok"]


def fig_tokens(rows: list[dict], out: Path) -> None:
    good = ok(rows, "tokens")
    if not good:
        return
    oomed = [num(r, "tokens") for r in rows
             if r["tier"] == "tokens" and r["status"] == "oom"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    xs, med, lo, hi = agg(good, "tokens", "predict_s")
    band(ax1, xs, med, lo, hi, "predict phase")
    xs2, med2, lo2, hi2 = agg(good, "tokens", "wall_s")
    band(ax1, xs2, med2, lo2, hi2, "end-to-end wall", marker="s")
    ax1.set_xlabel("tokens")
    ax1.set_ylabel("seconds")
    ax1.set_title("Runtime vs token count (1 GCD, MI250X)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend()

    xs3, med3, lo3, hi3 = agg(good, "tokens", "peak_alloc_gib")
    band(ax2, xs3, med3, lo3, hi3, "torch peak allocated")
    xs4, med4, _, _ = agg(good, "tokens", "gpu_peak_vram_gib")
    if xs4:
        ax2.plot(xs4, med4, marker="^", linestyle="--", label="GCD VRAM in use")
    ax2.axhline(64, color="crimson", linestyle=":", label="MI250X GCD capacity (64 GiB)")
    if oomed:
        ax2.axvline(min(oomed), color="crimson", alpha=0.5)
        ax2.annotate(f"OOM from\n{int(min(oomed))} tokens", xy=(min(oomed), 32),
                     xytext=(6, 0), textcoords="offset points",
                     color="crimson", fontsize=9, va="center")
    ax2.set_xlabel("tokens")
    ax2.set_ylabel("GiB")
    ax2.set_title("Peak memory vs token count")
    ax2.set_xscale("log", base=2)
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_params(rows: list[dict], out: Path) -> None:
    panels = [("recycle", "recycling_steps", "recycling_steps"),
              ("samples", "diffusion_samples", "diffusion_samples"),
              ("steps", "sampling_steps", "sampling_steps")]
    panels = [p for p in panels if ok(rows, p[0])]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (tier, col, label) in zip(axes, panels):
        xs, med, lo, hi = agg(ok(rows, tier), col, "predict_s")
        band(ax, xs, med, lo, hi, "predict phase")
        ax.set_xlabel(label)
        ax.set_ylabel("seconds")
        ax.set_title(f"Runtime vs {label}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_msa(rows: list[dict], out: Path) -> None:
    good = ok(rows, "msa")
    if not good:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    xs, med, lo, hi = agg(good, "axis_value", "predict_s")
    band(ax, xs, med, lo, hi, "predict phase")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("MSA depth (sequences)")
    ax.set_ylabel("seconds")
    ax.set_title("Runtime vs MSA depth")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_overhead(rows: list[dict], out: Path) -> None:
    good = ok(rows, "batch")
    if not good:
        return
    def med_of(sub: list[dict], key: str) -> float:
        vals = sorted(v for v in (num(r, key) for r in sub) if v is not None)
        return vals[len(vals) // 2] if vals else 0.0

    xs = sorted({num(r, "batch_n") for r in good if num(r, "batch_n")})
    parts = ["import_s", "setup_s", "predict_s", "other_s"]
    labels = {"import_s": "python+torch import", "setup_s": "preprocess + weight load",
              "predict_s": "inference", "other_s": "container start + write-out"}
    stacks: dict[str, list[float]] = {p: [] for p in parts}
    wall_per_struct, predict_per_struct = [], []

    for x in xs:
        sub = [r for r in good if num(r, "batch_n") == x]
        wall = med_of(sub, "wall_s")
        vals = {p: med_of(sub, p) for p in parts[:3]}
        # Whatever wall time the in-container phases do not account for is singularity
        # start, image page-in and output write. Naming it keeps the bars honest.
        vals["other_s"] = max(0.0, wall - sum(vals.values()))
        total = sum(vals.values()) or 1.0
        for p in parts:
            stacks[p].append(100.0 * vals[p] / total)
        wall_per_struct.append(wall / x if x else 0.0)
        predict_per_struct.append(vals["predict_s"] / x if x else 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Share of wall time, not absolute seconds: at n=64 the fixed cost is a sliver of a
    # very tall bar, and the whole point of the panel is that the sliver is shrinking.
    bottom = [0.0] * len(xs)
    pos = list(range(len(xs)))
    for p in parts:
        ax1.bar(pos, stacks[p], bottom=bottom, label=labels[p])
        bottom = [b + v for b, v in zip(bottom, stacks[p])]
    ax1.set_xticks(pos)
    ax1.set_xticklabels([str(int(x)) for x in xs])
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("structures per job")
    ax1.set_ylabel("share of wall time (%)")
    ax1.set_title("Where the wall time goes")
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(alpha=0.3, axis="y")

    # Must be driven by WALL time. predict_s excludes the fixed cost by construction,
    # so plotting it here would show a flat line and hide the effect being measured.
    ax2.plot(xs, wall_per_struct, marker="o", label="wall / structure")
    ax2.plot(xs, predict_per_struct, marker="s", linestyle="--",
             label="inference only (floor)")
    ax2.set_xlabel("structures per job")
    ax2.set_ylabel("seconds per structure")
    ax2.set_title("Fixed-overhead amortisation")
    ax2.set_xscale("log", base=2)
    ax2.set_ylim(bottom=0)
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_scaling(rows: list[dict], out: Path) -> None:
    intra = ok(rows, "scale1")
    multi = ok(rows, "scaleN")
    if not intra and not multi:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    def throughput(sub: list[dict], key: str) -> tuple[list, list]:
        """Structures/hour vs `key`, medianed over repeats.

        A multi-node run emits one row per rank, so a run's throughput is the SUM
        over its ranks (they work concurrently on disjoint shards). Group by run_id
        first, sum ranks, then take the median across repeats of the same x.
        """
        per_run: dict[str, tuple[float, float]] = {}
        for row in sub:
            x = num(row, key)
            n = num(row, "n_inputs") or 0
            t = num(row, "predict_s") or num(row, "wall_s")
            if not (x and n and t):
                continue
            rid = row["run_id"]
            prev = per_run.get(rid, (x, 0.0))
            per_run[rid] = (x, prev[1] + n / t * 3600.0)

        buckets: dict[float, list[float]] = defaultdict(list)
        for x, rate in per_run.values():
            buckets[x].append(rate)
        xs = sorted(buckets)
        med = []
        for x in xs:
            vals = sorted(buckets[x])
            mid = len(vals) // 2
            med.append(vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2)
        return xs, med

    for sub, key, label, ax in ((intra, "devices", "GCDs (1 node)", ax1),
                                (multi, "nodes", "nodes (8 GCDs each)", ax2)):
        if not sub:
            ax.set_visible(False)
            continue
        # ddp and indep are different launch models, not repeats of one another.
        modes = sorted({r.get("launch_mode") or "ddp" for r in sub})
        drew = False
        for mode in modes:
            xs, ys = throughput([r for r in sub
                                 if (r.get("launch_mode") or "ddp") == mode], key)
            if not xs:
                continue
            drew = True
            ax.plot(xs, ys, marker="o", label=f"{mode}")
            ideal = [ys[0] * (x / xs[0]) for x in xs]
            if len(modes) == 1:
                ax.plot(xs, ideal, linestyle="--", color="grey", label="linear")
            for x, y, i in zip(xs, ys, ideal):
                if i:
                    ax.annotate(f"{100 * y / i:.0f}%", (x, y),
                                textcoords="offset points", xytext=(0, -14),
                                fontsize=8, ha="center")
        if not drew:
            ax.set_visible(False)
            continue
        if len(modes) > 1:
            xs0, ys0 = throughput([r for r in sub
                                   if (r.get("launch_mode") or "ddp") == modes[0]], key)
            ax.plot(xs0, [ys0[0] * (x / xs0[0]) for x in xs0],
                    linestyle="--", color="grey", label="linear")
        ax.set_xlabel(label)
        ax.set_ylabel("structures / hour")
        ax.set_title(f"Throughput scaling: {label}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_energy(rows: list[dict], out: Path) -> None:
    good = [r for r in ok(rows, "tokens") if num(r, "energy_j_per_structure")]
    if not good:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    xs, med, lo, hi = agg(good, "tokens", "energy_j_per_structure")
    band(ax, xs, [m / 1000 for m in med], [v / 1000 for v in lo],
         [v / 1000 for v in hi], "energy per structure")
    ax.set_xlabel("tokens")
    ax.set_ylabel("kJ per structure")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Energy cost per prediction (1 GCD)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_image(rows: list[dict], out: Path) -> None:
    good = ok(rows, "image")
    if not good:
        return
    images = sorted({r["image_name"] for r in good})
    if len(images) < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for img in images:
        xs, med, lo, hi = agg([r for r in good if r["image_name"] == img],
                              "tokens", "predict_s")
        band(ax, xs, med, lo, hi, img)
    ax.set_xlabel("tokens")
    ax.set_ylabel("seconds (predict phase)")
    ax.set_xscale("log", base=2)
    ax.set_title("Container / ROCm version comparison")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    rows = load(args.results)
    args.outdir.mkdir(parents=True, exist_ok=True)

    for name, fn in (("fig_tokens.png", fig_tokens), ("fig_params.png", fig_params),
                     ("fig_msa.png", fig_msa), ("fig_overhead.png", fig_overhead),
                     ("fig_scaling.png", fig_scaling), ("fig_energy.png", fig_energy),
                     ("fig_image.png", fig_image)):
        path = args.outdir / name
        fn(rows, path)
        print(f"{'wrote' if path.exists() else 'skipped (no data)'} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
