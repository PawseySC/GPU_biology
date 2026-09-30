#!/usr/bin/env python3
"""Patch Keras 3 so unmasked LSTM layers run on ROCm/MIOpen.

MIOpen only implements the "packed" RNN path, which TensorFlow selects only
when CudnnRNNV3 is called with time_major=True and full-length sequences.
Keras 3 always calls it batch-major (time_major=False), so every cuDNN LSTM
fails on ROCm with:

    INVALID_ARGUMENT: ROCm MIOpen only supports packed input output.

Calling CudnnRNNV3 time-major is not enough either: the V3 kernel always
builds its IO descriptors through the sequence-lengths overload, which
MIOpen does not implement ("CreateRnnSequenceTensorDescriptor is
unimplemented").

This patch makes _cudnn_lstm, on ROCm builds with no mask, transpose
batch-major inputs to time-major, call the plain CudnnRNN op (packed IO, no
sequence lengths) instead of CudnnRNNV3, and transpose the outputs back.
Masked (variable-length) LSTMs are left untouched and remain unsupported
by MIOpen.

Usage: apply_keras_rocm_patch.py [path/to/keras/src/backend/tensorflow/rnn.py]
"""

import py_compile
import sys
import sysconfig
from pathlib import Path

MARKER = "rocm_packed"

FUNC_START = "def _cudnn_lstm("

ANCHOR_PRE = (
    "    if not time_major and sequence_lengths is None:\n"
    "        inputs = tf.transpose(inputs, perm=(1, 0, 2))\n"
)
PATCH_PRE = (
    "    # ROCm MIOpen only supports packed IO: time-major, full-length\n"
    "    # sequences, via the plain CudnnRNN op. Feed unmasked batch-major\n"
    "    # inputs as time-major.\n"
    "    rocm_packed = (\n"
    '        mask is None and tf.sysconfig.get_build_info()["is_rocm_build"]\n'
    "    )\n"
    "    rocm_batch_major = rocm_packed and not time_major\n"
    "    if rocm_batch_major:\n"
    "        inputs = tf.transpose(inputs, perm=(1, 0, 2))\n"
    "        time_major = True\n"
    "\n"
)

CALL_V3 = (
    "    outputs, h, c, _, _ = tf.raw_ops.CudnnRNNV3(\n"
    "        input=inputs,\n"
    "        input_h=init_h,\n"
    "        input_c=init_c,\n"
    "        params=params,\n"
    "        is_training=True,\n"
    '        rnn_mode="lstm",\n'
    "        sequence_lengths=sequence_lengths,\n"
    "        time_major=time_major,\n"
    "    )\n"
)
CALL_PATCHED = (
    "    if rocm_packed:\n"
    "        outputs, h, c, _ = tf.raw_ops.CudnnRNN(\n"
    "            input=inputs,\n"
    "            input_h=init_h,\n"
    "            input_c=init_c,\n"
    "            params=params,\n"
    "            is_training=True,\n"
    '            rnn_mode="lstm",\n'
    "        )\n"
    "    else:\n"
    + "".join("    " + line for line in CALL_V3.splitlines(keepends=True))
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
    if body.count(CALL_V3) != 1:
        sys.exit(
            "ERROR: CudnnRNNV3 call in _cudnn_lstm not found as expected.\n"
            "Keras source has changed; review apply_keras_rocm_patch.py."
        )
    body = body.replace(CALL_V3, CALL_PATCHED)

    path.write_text(source[:start] + body + source[end:])
    py_compile.compile(str(path), doraise=True)
    print(f"Patched _cudnn_lstm for ROCm packed IO: {path}")


if __name__ == "__main__":
    main()
