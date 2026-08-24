#!/usr/bin/env python3
"""Sample AMD GPU power / utilisation / VRAM at 1 Hz into a CSV, on the host.

Runs on the compute node, outside the container, alongside the Boltz process.
Prefers `amd-smi` (ROCm 6.2+, stable JSON) and falls back to `rocm-smi --json`.

Purpose is energy-per-prediction and utilisation. Peak VRAM should be taken from
bench_wrap.py's torch.cuda.max_memory_allocated(); rocm-smi/amd-smi report the whole
GCD, including HIP context overhead and any co-tenant, so they over-report.

Usage:
    gpu_monitor.py --out gpu.csv --interval 1.0 &
    MON=$!
    ... run workload ...
    kill -TERM $MON
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import signal
import subprocess
import sys
import time

RUNNING = True


def _stop(*_a: object) -> None:
    global RUNNING  # noqa: PLW0603
    RUNNING = False


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def _first_number(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if m:
            return float(m.group())
    if isinstance(value, dict):
        for key in ("value", "Value"):
            if key in value:
                return _first_number(value[key])
    return None


def _dig(obj: object, *needles: str) -> float | None:
    """Depth-first search for the first key whose lowercased name contains all needles."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            kl = str(key).lower()
            if all(n in kl for n in needles):
                num = _first_number(val)
                if num is not None:
                    return num
        for val in obj.values():
            found = _dig(val, *needles)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for val in obj:
            found = _dig(val, *needles)
            if found is not None:
                return found
    return None


def sample_amd_smi() -> list[dict] | None:
    raw = _run(["amd-smi", "metric", "--json"])
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    gpus = data if isinstance(data, list) else [data]
    rows = []
    for idx, gpu in enumerate(gpus):
        rows.append({
            "gpu": gpu.get("gpu", idx) if isinstance(gpu, dict) else idx,
            "power_w": _dig(gpu, "socket", "power") or _dig(gpu, "power"),
            "util_pct": _dig(gpu, "gfx", "activity") or _dig(gpu, "gfx_activity"),
            "vram_used_mib": _dig(gpu, "vram", "used"),
            "sclk_mhz": _dig(gpu, "gfx", "clk") or _dig(gpu, "sclk"),
            "temp_c": _dig(gpu, "edge") or _dig(gpu, "temperature"),
        })
    return rows


def sample_rocm_smi() -> list[dict] | None:
    raw = _run(["rocm-smi", "--showpower", "--showuse", "--showmemuse",
                "--showmeminfo", "vram", "--showtemp", "--json"])
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    rows = []
    for key, gpu in sorted(data.items()):
        if not key.startswith("card"):
            continue
        used = _dig(gpu, "vram", "used")
        rows.append({
            "gpu": key.replace("card", ""),
            "power_w": _dig(gpu, "power"),
            "util_pct": _dig(gpu, "gpu", "use"),
            "vram_used_mib": (used / 2**20) if used and used > 2**20 else used,
            "sclk_mhz": _dig(gpu, "sclk"),
            "temp_c": _dig(gpu, "temperature"),
        })
    return rows or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    backend = None
    if sample_amd_smi() is not None:
        backend = sample_amd_smi
        backend_name = "amd-smi"
    elif sample_rocm_smi() is not None:
        backend = sample_rocm_smi
        backend_name = "rocm-smi"
    else:
        sys.stderr.write("gpu_monitor: neither amd-smi nor rocm-smi usable; no samples\n")
        # Still write a header so downstream parsing has a well-formed empty file.
        with open(args.out, "w", newline="") as fh:
            csv.writer(fh).writerow(
                ["t_s", "backend", "gpu", "power_w", "util_pct",
                 "vram_used_mib", "sclk_mhz", "temp_c"])
        return 0

    t0 = time.time()
    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t_s", "backend", "gpu", "power_w", "util_pct",
                         "vram_used_mib", "sclk_mhz", "temp_c"])
        while RUNNING:
            tick = time.time()
            for row in backend() or []:
                writer.writerow([
                    round(tick - t0, 3), backend_name, row["gpu"], row["power_w"],
                    row["util_pct"], row["vram_used_mib"], row["sclk_mhz"], row["temp_c"],
                ])
            fh.flush()
            time.sleep(max(0.0, args.interval - (time.time() - tick)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
