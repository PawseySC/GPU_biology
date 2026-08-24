#!/usr/bin/env python3
"""Collate perf runs into a markdown table.

    ./report_perf.py [results/perf] > PERFORMANCE.md

Reads timing.json and vram.csv from each run directory. Rows that failed are
kept, not dropped - an OOM at a given size is the most useful number in the
table, and silently omitting it is how a benchmark starts lying.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

GIB = 1024 ** 3


def peak_vram(csv_path: Path) -> tuple[float | None, float | None, float | None]:
    """(peak_used_gib, total_gib, mean_busy_pct) from a monitor_vram.sh log."""
    if not csv_path.exists():
        return None, None, None
    used, total, busy = [], [], []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            for key, acc in (("vram_used_bytes", used),
                             ("vram_total_bytes", total),
                             ("gpu_busy_pct", busy)):
                try:
                    acc.append(float(str(row.get(key, "")).strip().rstrip("%")))
                except (TypeError, ValueError):
                    pass
    return (
        max(used) / GIB if used else None,
        max(total) / GIB if total else None,
        sum(busy) / len(busy) if busy else None,
    )


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/perf")
    if not root.exists():
        print(f"no results at {root}", file=sys.stderr)
        return 1

    runs = []
    for d in sorted(root.iterdir()):
        tj = d / "timing.json"
        if not tj.exists():
            continue
        t = json.loads(tj.read_text())
        pk, total, busy = peak_vram(d / "vram.csv")
        t.update(peak_gib=pk, total_gib=total, busy=busy)
        runs.append(t)

    if not runs:
        print(f"no timing.json under {root}", file=sys.stderr)
        return 1

    gpu = next((r.get("gpu") for r in runs if r.get("gpu")), "unknown")
    capacity = next((r["total_gib"] for r in runs if r.get("total_gib")), None)

    print("# OpenFold3 container performance\n")
    print(f"- GPU: `{gpu.strip() if isinstance(gpu, str) else gpu}`"
          + (f" ({capacity:.0f} GiB)" if capacity else ""))
    print("- MSAs disabled in these queries, so timings are model-only. "
          "Add the alignment stage separately for end-to-end numbers.")
    print("- 1 model seed x 1 diffusion sample, 10 recycles, fp32.\n")

    print("| run | tokens | wall clock | peak VRAM | GPU busy | result |")
    print("|---|---|---|---|---|---|")
    for r in runs:
        name = r.get("name", "?")
        tok = _tokens(name)
        secs = r.get("wall_seconds")
        wall = _hms(secs) if isinstance(secs, (int, float)) else "-"
        pk = f"{r['peak_gib']:.1f} GiB" if r.get("peak_gib") else "-"
        busy = f"{r['busy']:.0f}%" if r.get("busy") else "-"
        ok = r.get("exit_code", 1) == 0
        print(f"| `{name}` | {tok} | {wall} | {pk} | {busy} | "
              f"{'ok' if ok else '**failed**'} |")

    failed = [r for r in runs if r.get("exit_code", 1) != 0]
    if failed:
        print(f"\n{len(failed)} run(s) failed - check `predict.log` in each "
              "directory. For the top of the ladder this is normally the "
              "memory ceiling, which is worth quoting explicitly.")
    return 0


def _tokens(name: str) -> str:
    for part in name.split("_"):
        if part.endswith("tok") and part[:-3].isdigit():
            return part[:-3]
    return "-"


def _hms(s: float) -> str:
    s = int(round(s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


if __name__ == "__main__":
    sys.exit(main())
