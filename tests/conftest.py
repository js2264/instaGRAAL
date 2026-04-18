"""Pytest configuration and fixtures for instaGRAAL tests.

Mocks CUDA/GPU-related modules so that tests can run without GPU hardware.
"""

import sys
from unittest.mock import MagicMock

# Mock pycuda and related CUDA modules before any instagraal imports,
# so tests can run without GPU hardware.
_cuda_mocks = [
    "pycuda",
    "pycuda.autoinit",
    "pycuda.driver",
    "pycuda.compiler",
    "pycuda.tools",
    "pycuda.characterize",
    "pycuda.gpuarray",
    "cgen",
    "codepy",
    "codepy.bpl",
    "codepy.cuda",
    "codepy.jit",
    "codepy.toolchain",
]

for _mod_name in _cuda_mocks:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()
