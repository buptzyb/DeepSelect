"""DeepSelect bfa4507 + 3 local patches, trimmed to the GLM-5.2 DSA indexer top-k configs.

``CONFIGS`` (configs.json) lists the compiled kernel configurations; ``BUILD_INFO``
(build_info.json) records the DeepSelect commit, patch hashes, toolchain and build job.
``topk(x, k, end, cfg_id, ...)`` keeps the call signature of the h2h-validated ``ds_tune2``
build, so adapters that look a config up by name (``f_t512_b2_2k_d3``) work unchanged.

Input contract (checked on the host by the extension): x is [rows, width] fp32 or bf16 with
stride(1) == 1 and stride(0) * itemsize a multiple of 1024 B; width < 2^23; end is int32
[rows] (valid prefix length per row); outputs are [rows, k] with row strides a multiple of
32 B; 0 < k <= the config's max_topk. Results are unsorted (sorted_index=False) or sorted by
index (sorted_index=True); padding slots get idx_oob_fill_value / value_oob_fill_value. With
abort_when_nan_found=False a row containing NaN is left with the 0x3F3F3F3F sentinel in
idx[r, 0] (callers repair such rows).
"""

import json
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "build_info.json")) as _f:
    BUILD_INFO = json.load(_f)
__version__ = BUILD_INFO["version"]

if str(torch.__version__) != BUILD_INFO["torch"]:
    raise ImportError(
        f"ds_tune {__version__} at {_HERE} was built against torch {BUILD_INFO['torch']} "
        f"(container {BUILD_INFO['container']}); this interpreter has torch {torch.__version__}. "
        "Rebuild the overlay for this torch (see PROVENANCE.md) instead of loading a mismatched ABI."
    )

from . import ds_tune_cuda as _C  # noqa: E402

with open(os.path.join(_HERE, "configs.json")) as _f:
    CONFIGS = json.load(_f)
CONFIGS_BY_NAME = {c["name"]: c for c in CONFIGS}
INPUT_STRIDE_ALIGNMENT_BYTES, OUTPUT_STRIDE_ALIGNMENT_BYTES = _C.get_alignment_requirement()
SUPPORTED_ARCHS = tuple(BUILD_INFO["cuda_archs"])  # e.g. ("sm_100a", "sm_103a")
NAN_ROW_SENTINEL = 0x3F3F3F3F
_DTYPES = {torch.float32: "fp32", torch.bfloat16: "bf16"}
_ARCH_OK = {}


def config_for(dtype, k):
    """The compiled config for ``dtype`` (torch.float32 / torch.bfloat16) with the smallest
    max_topk >= k: the tuned max_topk 2048 tier for k <= 2048, the stock 4096 tier above."""
    name = _DTYPES.get(dtype)
    if name is None:
        raise ValueError(f"ds_tune supports fp32 and bf16 scores, got {dtype}")
    fits = [c for c in CONFIGS if c["dtype"] == name and k <= c["max_topk"]]
    if k <= 0 or not fits:
        raise ValueError(f"ds_tune has no {name} config for k={k} (max {max(c['max_topk'] for c in CONFIGS)})")
    return min(fits, key=lambda c: c["max_topk"])


def check_device(device=None):
    """Raise unless the CUDA device has an architecture this build contains (cached per device)."""
    idx = torch.cuda.current_device() if device is None else torch.device(device).index
    if idx is None:
        idx = torch.cuda.current_device()
    ok = _ARCH_OK.get(idx)
    if ok is None:
        major, minor = torch.cuda.get_device_capability(idx)
        ok = f"sm_{major}{minor}a" in SUPPORTED_ARCHS
        _ARCH_OK[idx] = ok
        if not ok:
            raise RuntimeError(f"ds_tune {__version__} has kernels for {SUPPORTED_ARCHS}; "
                               f"device {idx} is sm_{major}{minor}")
    return ok


def _out_alloc(n, k, dtype, device):
    itemsize = torch.empty((), dtype=dtype).element_size()
    align = OUTPUT_STRIDE_ALIGNMENT_BYTES // itemsize
    pad = (k + align - 1) // align * align
    return torch.empty((n, pad), device=device, dtype=dtype)[:, :k]


def topk(x, k, end, cfg_id, sorted_index=False, out_val=None, out_idx=None,
         idx_oob_fill_value=-1, value_oob_fill_value=float("-inf"), abort_when_nan_found=False):
    """Top-k of every row of ``x[:, :end[r]]`` with compiled config ``cfg_id`` (int id or name).

    Returns ``(values [rows, k] x.dtype, indices [rows, k] int32)``; pass ``out_val``/``out_idx``
    to write into preallocated (e.g. CUDA-graph-static) buffers.
    """
    if isinstance(cfg_id, str):
        cfg_id = CONFIGS_BY_NAME[cfg_id]["id"]
    check_device(x.device)
    n = x.shape[0]
    if out_val is None:
        out_val = _out_alloc(n, k, x.dtype, x.device)
    if out_idx is None:
        out_idx = _out_alloc(n, k, torch.int32, x.device)
    _C.topk_cfg(x, k, end, out_val, out_idx, cfg_id, sorted_index,
                idx_oob_fill_value, value_oob_fill_value, abort_when_nan_found)
    return out_val, out_idx
