"""The production GPU image must not silently fall back to CPU."""

import os
import sys

import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Start perception with --gpus all and NVIDIA drivers.")
os.execv("/app/scripts/entrypoint.sh", ["/app/scripts/entrypoint.sh", *sys.argv[1:]])
