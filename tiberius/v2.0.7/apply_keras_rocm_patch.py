#!/usr/bin/env python3
"""Patch Keras 3 so unmasked LSTM layers run on ROCm/MIOpen.

MIOpen only implements the "packed" RNN path, which TensorFlow selects only
when CudnnRNNV3 is called with time_major=True and full-length sequences.
Keras 3 always calls it batch-major (time_major=False), so every cuDNN LSTM
fails on ROCm with:

    INVALID_ARGUMENT: ROCm MIOpen only supports packed input output.

This patch makes _cudnn_lstm transpose unmasked batch-major inputs to
time-major before the call and transpose the outputs back afterwards.
Masked (variable-length) LSTMs are left untouched and remain unsupported
by MIOpen.

Usage: apply_keras_rocm_patch.py [path/to/keras/src/backend/tensorflow/rnn.py]
"""

import py_compile
import sys
import sysconfig
from pathlib import Path

MARKER = "rocm_batch_major"

FUNC_START = "def _cudnn_lstm("

ANCHOR_PRE = (
    "    if not time_major and sequence_lengths is None:\n"
    "        inputs = tf.transpose(inputs, perm=(1, 0, 2))\n"
)
PATCH_PRE = (
    "    # ROCm MIOpen only supports packed IO: time-major, full-length\n"
    "    # sequences. Feed unmasked batch-major inputs as time-major.\n"
    "    rocm_batch_major = (\n"
    "        mask is None\n"
    "        and not time_major\n"
    '        and tf.sysconfig.get_build_info()["is_rocm_build"]\n'
    "    )\n"
    "    if rocm_batch_major:\n"
    "        inputs = tf.transpose(inputs, perm=(1, 0, 2))\n"
    "        time_major = True\n"
    "\n"
)

ANCHOR_POST = "    # Match CPU return format\n"
PATCH_POST = (
    "    if rocm_batch_major:\n"
    "        # Restore the caller's batch-major layout\n"
    "        time_major = False\n"
    "        if return_sequences:\n"
    "            outputs = tf.transpose(outputs, perm=(1, 0, 2))\n"
    "\n"
)


def insert_before(body, anchor, patch):
    if body.count(anchor) != 1:
        sys.exit(
            f"ERROR: expected exactly one match in _cudnn_lstm for:\n{anchor}\n"
            "Keras source has changed; review apply_keras_rocm_patch.py."
        )
    return body.replace(anchor, patch + anchor)


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = (
            Path(sysconfig.get_paths()["purelib"])
            / "keras/src/backend/tensorflow/rnn.py"
        )
    source = path.read_text()

    if MARKER in source:
        print(f"Already patched: {path}")
        return

    start = source.find(FUNC_START)
    if start == -1:
        sys.exit(f"ERROR: {FUNC_START} not found in {path}")
    end = source.find("\ndef ", start + 1)
    if end == -1:
        end = len(source)

    body = source[start:end]
    body = insert_before(body, ANCHOR_PRE, PATCH_PRE)
    body = insert_before(body, ANCHOR_POST, PATCH_POST)

    path.write_text(source[:start] + body + source[end:])
    py_compile.compile(str(path), doraise=True)
    print(f"Patched _cudnn_lstm for ROCm packed IO: {path}")


if __name__ == "__main__":
    main()
