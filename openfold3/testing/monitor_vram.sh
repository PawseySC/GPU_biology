#!/bin/bash
# Sample VRAM and GPU utilisation to CSV until killed.
#
#   ./monitor_vram.sh <out.csv> [interval_seconds]
#
# Replaces the older vram_monitoring.sh approach of `rocm-smi | awk '{print $9}'`.
# Two reasons that one cannot be trusted here:
#
#   1. Positional columns move between rocm-smi releases. These containers span
#      ROCm 6.2 through 7.2, so $9 is not the same field everywhere and the
#      mismatch is silent - you get a plausible number from the wrong column.
#      --csv with header lookup is stable across versions.
#   2. A 60s poll misses the peak. Diffusion sampling spikes over seconds, so
#      the sampled maximum is close to arbitrary. Default here is 2s.
#
# For the true high-water mark, torch.cuda.max_memory_allocated() inside the
# run beats any external sampler. This is for when you cannot instrument.

set -u
OUT="${1:?usage: monitor_vram.sh <out.csv> [interval]}"
INTERVAL="${2:-2}"

echo "timestamp,card,vram_used_bytes,vram_total_bytes,gpu_busy_pct" > "${OUT}"

while true; do
  TS=$(date +%s.%N)
  rocm-smi --showmeminfo vram --showuse --csv 2>/dev/null | awk -v ts="${TS}" '
    NR == 1 {
      # Locate columns by header name rather than by position.
      for (i = 1; i <= NF; i++) {
        h = tolower($i)
        # Note the !~ /used/ guard on total: rocm-smi names the columns
        # "VRAM Total Memory" and "VRAM Total Used Memory", so a naive
        # /total/ match silently binds total to the used column.
        if (h ~ /vram/ && h ~ /used/)                used  = i
        if (h ~ /vram/ && h ~ /total/ && h !~ /used/) total = i
        if (h ~ /gpu/  && h ~ /use/)                 busy  = i
        if (h ~ /^device|^card|^gpu$/)               card  = i
      }
      next
    }
    NF > 1 {
      printf "%s,%s,%s,%s,%s\n", ts,
        (card  ? $card  : "card?"),
        (used  ? $used  : ""),
        (total ? $total : ""),
        (busy  ? $busy  : "")
    }
  ' FS=, OFS=, >> "${OUT}"
  sleep "${INTERVAL}"
done
