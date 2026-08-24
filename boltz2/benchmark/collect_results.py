#!/usr/bin/env python3
"""Walk the results tree and flatten every run into one tidy CSV.

Joins three sources per run:
  meta.json   host-side wall time (includes singularity start + image page-in)
  bench.json  in-container phase split and torch peak VRAM
  gpu.csv     1 Hz power/utilisation samples -> mean power and integrated energy

Runs that OOMed or failed are kept with status set, not dropped: the OOM ceiling on
the token ladder is a result, and silently dropping rows would misreport it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

OUT_COLUMNS = [
    "run_id", "tier", "axis", "axis_value", "repeat", "status", "exit_code",
    "tokens", "batch_n", "n_inputs", "nodes", "devices", "launch_mode", "rank",
    "recycling_steps", "sampling_steps", "diffusion_samples", "max_parallel_samples",
    "affinity", "use_potentials", "num_workers", "preproc_threads", "image_name",
    "wall_s", "import_s", "setup_s", "predict_s", "total_s",
    "per_structure_median_s", "s_per_structure", "structures_per_gpu_hour",
    "peak_alloc_gib", "peak_reserved_gib", "gpu_peak_vram_gib",
    "mean_power_w", "peak_power_w", "energy_kj", "energy_j_per_structure",
    "mean_util_pct", "hostname", "torch", "hip", "device_name",
]


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def summarise_gpu(path: Path) -> dict:
    """Mean/peak power, integrated energy and mean utilisation from the 1 Hz samples.

    Energy is trapezoidal over each GPU's own series, then summed across GPUs, so a
    dropped sample stretches an interval rather than losing its contribution.
    """
    if not path.exists():
        return {}
    series: dict[str, list[tuple[float, float]]] = {}
    utils: list[float] = []
    vram: list[float] = []
    try:
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    t = float(row["t_s"])
                except (TypeError, ValueError, KeyError):
                    continue
                gpu = row.get("gpu", "0")
                if row.get("power_w"):
                    try:
                        series.setdefault(gpu, []).append((t, float(row["power_w"])))
                    except ValueError:
                        pass
                for key, sink in (("util_pct", utils), ("vram_used_mib", vram)):
                    if row.get(key):
                        try:
                            sink.append(float(row[key]))
                        except ValueError:
                            pass
    except OSError:
        return {}

    energy_j = 0.0
    powers: list[float] = []
    for samples in series.values():
        samples.sort()
        powers.extend(p for _, p in samples)
        for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
            energy_j += (p0 + p1) / 2.0 * (t1 - t0)

    out: dict = {}
    if powers:
        out["mean_power_w"] = round(sum(powers) / len(powers), 2)
        out["peak_power_w"] = round(max(powers), 2)
        out["energy_kj"] = round(energy_j / 1000.0, 4)
    if utils:
        out["mean_util_pct"] = round(sum(utils) / len(utils), 2)
    if vram:
        out["gpu_peak_vram_gib"] = round(max(vram) / 1024.0, 3)
    return out


def median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="BENCH_ROOT")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    manifest: dict[str, dict] = {}
    with args.manifest.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            manifest[row["run_id"]] = row

    results_root = args.root / "results"
    rows: list[dict] = []
    missing = 0

    for run_id, man in manifest.items():
        run_dir = results_root / run_id
        if not run_dir.exists():
            missing += 1
            continue
        # Multi-node runs write one subdirectory per rank; `indep` launch mode writes
        # one per pinned GCD. Both are concurrent workers of a single run.
        rank_dirs = sorted(
            d for d in run_dir.glob("rank*") if d.is_dir()
        ) or sorted(
            d for d in run_dir.glob("gcd*") if d.is_dir()
        ) or [run_dir]

        for rank_dir in rank_dirs:
            meta = read_json(rank_dir / "meta.json")
            bench = read_json(rank_dir / "bench.json")
            # One power sampler runs per node. Multi-node ranks each have their own
            # gpu.csv; `indep` per-GCD workers share the node-level one a directory up,
            # so its energy is attributed to the whole run, not to one worker.
            gpu_csv = rank_dir / "gpu.csv"
            shared_gpu = not gpu_csv.exists() and rank_dir != run_dir
            if shared_gpu:
                gpu_csv = run_dir / "gpu.csv"
            gpu = summarise_gpu(gpu_csv)
            if shared_gpu:
                # Node-level energy would be double-counted once per worker.
                gpu.pop("energy_kj", None)
            phases = bench.get("phases", {})
            env = bench.get("env", {})

            n_inputs = int(meta.get("n_inputs") or 1)
            devices = int(meta.get("devices") or man.get("devices") or 1)
            predict_s = phases.get("predict_s")
            wall_s = meta.get("wall_s")

            per_struct = median(bench.get("per_structure_s") or [])
            # Throughput uses the predict phase where we have it (excludes one-off
            # container + weight-load overhead) and falls back to wall time.
            basis = predict_s if predict_s else wall_s
            s_per_structure = (basis / n_inputs) if basis and n_inputs else None
            gpu_hour = None
            if s_per_structure and s_per_structure > 0:
                gpu_hour = 3600.0 / (s_per_structure * devices)

            energy_kj = gpu.get("energy_kj")
            e_per_struct = (
                energy_kj * 1000.0 / n_inputs if energy_kj and n_inputs else None
            )

            status = bench.get("status") or (
                "missing" if not bench else "unknown"
            )
            if meta.get("exit_code") not in (0, None) and status == "unknown":
                status = "error"

            rows.append({
                "run_id": run_id,
                "tier": man.get("tier"),
                "axis": man.get("axis"),
                "axis_value": man.get("axis_value"),
                "repeat": man.get("repeat"),
                "status": status,
                "exit_code": meta.get("exit_code"),
                "tokens": man.get("tokens"),
                "batch_n": man.get("batch_n"),
                "n_inputs": n_inputs,
                "nodes": meta.get("nodes", man.get("nodes")),
                "devices": devices,
                "launch_mode": meta.get("launch_mode", man.get("launch_mode")),
                "rank": meta.get("rank", 0),
                "recycling_steps": man.get("recycling_steps"),
                "sampling_steps": man.get("sampling_steps"),
                "diffusion_samples": man.get("diffusion_samples"),
                "max_parallel_samples": man.get("max_parallel_samples"),
                "affinity": man.get("affinity"),
                "use_potentials": man.get("use_potentials"),
                "num_workers": man.get("num_workers"),
                "preproc_threads": man.get("preproc_threads"),
                "image_name": Path(man.get("image", "")).name,
                "wall_s": round(float(wall_s), 3) if wall_s else None,
                "import_s": phases.get("import_s"),
                "setup_s": phases.get("setup_s"),
                "predict_s": predict_s,
                "total_s": phases.get("total_s"),
                "per_structure_median_s": per_struct,
                "s_per_structure": round(s_per_structure, 3) if s_per_structure else None,
                "structures_per_gpu_hour": round(gpu_hour, 2) if gpu_hour else None,
                "peak_alloc_gib": bench.get("peak_alloc_gib"),
                "peak_reserved_gib": bench.get("peak_reserved_gib"),
                "gpu_peak_vram_gib": gpu.get("gpu_peak_vram_gib"),
                "mean_power_w": gpu.get("mean_power_w"),
                "peak_power_w": gpu.get("peak_power_w"),
                "energy_kj": energy_kj,
                "energy_j_per_structure": round(e_per_struct, 1) if e_per_struct else None,
                "mean_util_pct": gpu.get("mean_util_pct"),
                "hostname": meta.get("hostname"),
                "torch": env.get("torch"),
                "hip": env.get("hip"),
                "device_name": env.get("device_name"),
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    oom = sum(1 for r in rows if r["status"] == "oom")
    bad = len(rows) - ok - oom
    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"  ok={ok}  oom={oom}  failed/other={bad}  not-yet-run={missing}")

    if oom:
        ceiling = [
            int(r["tokens"]) for r in rows
            if r["status"] == "oom" and str(r.get("tokens", "")).isdigit()
        ]
        if ceiling:
            print(f"  OOM observed from {min(ceiling)} tokens upward "
                  f"(single-GCD capability ceiling)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
