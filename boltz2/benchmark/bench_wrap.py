#!/usr/bin/env python3
"""In-container wrapper around `boltz predict` that emits machine-readable timings.

Runs inside the Boltz container. Everything after `--` is passed verbatim to the
Boltz CLI, so this stays valid across Boltz versions.

Why a wrapper rather than just `time boltz predict`:

  * torch.cuda.max_memory_allocated() is the only unambiguous peak-VRAM number.
    rocm-smi sees the whole GCD including other tenants and the HIP context
    overhead, so it over-reports; we record both and report both.
  * Total wall time on a small target is dominated by interpreter import, weight
    load and CCD/molecule-cache read. Reporting only end-to-end makes the GPU look
    slow. We split: import -> setup (preprocess + weight load) -> predict -> teardown.
  * The predict phase is isolated by attaching a Lightning callback. Boltz builds its
    own Trainer, so we patch Trainer.__init__ to append the callback. Guarded: if the
    patch fails (upstream refactor), we still get import/total and say so.

Output: one JSON object to the path given by --bench-json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback

T0 = time.perf_counter()

RESULT: dict = {
    "schema": 1,
    "phases": {},
    "per_structure_s": [],
    "callback_attached": False,
    "status": "unknown",
}


def _now() -> float:
    return time.perf_counter() - T0


def _attach_callback() -> None:
    """Append a timing callback to every Lightning Trainer boltz constructs."""
    import pytorch_lightning as pl

    class BenchTimer(pl.Callback):
        def on_predict_start(self, trainer, pl_module) -> None:  # noqa: ANN001
            RESULT["phases"]["setup_s"] = _now() - RESULT["phases"]["import_s"]
            self._predict_t0 = time.perf_counter()

        def on_predict_batch_start(self, *a, **kw) -> None:  # noqa: ANN002, ANN003
            self._batch_t0 = time.perf_counter()

        def on_predict_batch_end(self, *a, **kw) -> None:  # noqa: ANN002, ANN003
            RESULT["per_structure_s"].append(
                round(time.perf_counter() - self._batch_t0, 4)
            )

        def on_predict_end(self, trainer, pl_module) -> None:  # noqa: ANN001
            RESULT["phases"]["predict_s"] = round(
                time.perf_counter() - self._predict_t0, 4
            )

    orig_init = pl.Trainer.__init__

    def patched_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        cbs = list(kwargs.get("callbacks") or [])
        cbs.append(BenchTimer())
        kwargs["callbacks"] = cbs
        return orig_init(self, *args, **kwargs)

    pl.Trainer.__init__ = patched_init
    RESULT["callback_attached"] = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-json", required=True)
    ap.add_argument("--run-id", default=os.environ.get("BENCH_RUN_ID", ""))
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    RESULT["run_id"] = args.run_id
    RESULT["argv"] = rest

    import torch

    RESULT["phases"]["import_s"] = round(_now(), 4)
    RESULT["env"] = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "hostname": platform.node(),
    }
    for var in (
        "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "PYTORCH_HIP_ALLOC_CONF",
        "MIOPEN_USER_DB_PATH", "TRITON_CACHE_DIR", "OMP_NUM_THREADS",
        "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_NNODES",
    ):
        if var in os.environ:
            RESULT["env"][var] = os.environ[var]

    try:
        _attach_callback()
    except Exception as exc:  # noqa: BLE001 - never fail the run over instrumentation
        RESULT["callback_error"] = f"{type(exc).__name__}: {exc}"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    rc = 0
    try:
        from boltz.main import cli

        cli.main(args=rest, prog_name="boltz", standalone_mode=False)
        RESULT["status"] = "ok"
    except SystemExit as exc:
        rc = int(exc.code or 0)
        RESULT["status"] = "ok" if rc == 0 else "exit"
    except torch.cuda.OutOfMemoryError:
        rc = 2
        RESULT["status"] = "oom"
        RESULT["error"] = "torch.cuda.OutOfMemoryError"
    except Exception as exc:  # noqa: BLE001
        rc = 1
        # HIP surfaces some OOMs as a generic RuntimeError.
        text = f"{type(exc).__name__}: {exc}"
        RESULT["status"] = "oom" if "out of memory" in text.lower() else "error"
        RESULT["error"] = text
        RESULT["traceback"] = traceback.format_exc()

    RESULT["phases"]["total_s"] = round(_now(), 4)
    if torch.cuda.is_available():
        RESULT["peak_alloc_gib"] = round(
            torch.cuda.max_memory_allocated() / 2**30, 4
        )
        RESULT["peak_reserved_gib"] = round(
            torch.cuda.max_memory_reserved() / 2**30, 4
        )

    with open(args.bench_json, "w") as fh:
        json.dump(RESULT, fh, indent=2)
        fh.write("\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
