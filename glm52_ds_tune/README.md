# ds_tune: DeepSelect for the GLM-5.2 DSA indexer top-k

`ds_tune 0.1.0+bfa4507.p3` is the DeepSelect build that the GLM-5.2 GB300 recipes of Megatron-LM
(`examples/moe_recipes/glm5_2`, [NVIDIA/Megatron-LM#6944](https://github.com/NVIDIA/Megatron-LM/pull/6944))
use for `dsa_indexer_topk_backend: deepselect`. It is DeepSelect
`bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18`, the parent of the commit that adds this directory,
with three patches, built for four kernel configurations only.

This directory holds the build inputs. Except `build.sh` and this README, every file is byte for
byte an input of the measured build. `build.sh` is that build's procedure without its
cluster-specific Slurm and path settings and without the steps that do not affect the extension:
the hash checks and SASS comparisons against two unpublished reference builds, some diagnostic
listings, one import assertion for an internal adapter, and the staging of the measured build's
directory tree. Nothing outside this directory differs from DeepSelect `bfa4507`.

## Patches

The patches apply in order to DeepSelect `bfa4507` (`patch -p1` from the DeepSelect root).
`gen_pkg.py` applies them to a copy of `csrc/`; the checkout itself stays unmodified.

| File | SHA256 | Change |
| --- | --- | --- |
| `0001-ds_tune-allow-max-topk-2048.patch` | `4a223c5a938b4d4832a6ac801f5a94aec748bfb0f53cca179e37f1a26a054dc9` | Allows `MAX_TOPK == 2048` in the `static_assert` of the bf16 (`v3/`) and fp32 (`v3_fp32/`) `topk_select.cuh`; upstream allows 512, 1024 and 4096. |
| `0002-ds_tune-per-config-window-table-capacity.patch` | `a134233f4d0961d8bda64ce11ec04e1590ba8dbbd6fccae398c99040da200a06` | Replaces the fixed window-delta-table capacity check in `common_parts.cuh` with a per-configuration `MAX_VOCAB_THIS_CFG` (at least 2^21 columns) and adds a host check of `vocab_size` in both kernel launchers. This allows configurations with `B2 + B < 8192`. Of the four configurations below only configuration 1 (`B = 4096`, `B2 = 2048`) needs it; it accepts at most 6,291,456 columns, while the others keep the stock limit of fewer than 2^23 (the `width < 2^23` in `ds_tune/__init__.py` holds only for them). |
| `0003-ds_tune-private-kernel-namespaces.patch` | `bed6f6cb1e8b44d0361ee1138bb8e5fc41704e4d60af4159dba46daea3d50387` | Renames the kernel namespaces `topk_select_{common,fp32,bf16_normal}` to `tune_{common,fp32,bf16}`, so the extension never aliases the symbols of a stock `deep_select` build. |

`make_patches.py` records how the patches were derived from the generator of an earlier,
GPU-validated tuning build; it is provenance and is not needed to build. The copied files keep
their original comments, which name files, directories and Slurm jobs of the measured build and
of the earlier tuning work that are not in this repository (`build-ds-pkg.sbatch`,
`PROVENANCE.md`, `glm52-topk-efficiency-20261019/...`, `ds_tune2`); `build.sh` stands in for
`build-ds-pkg.sbatch` and this README for `PROVENANCE.md`.

## Kernel configurations

`gen_pkg.py` instantiates each configuration with int32 indices, `return_value=True` and
`sorted_value=False`, for `sorted_index` false and true (eight kernels), for `sm_100a` and
`sm_103a`. `api_tune.cpp.in` is the host entry point, which selects a configuration by id.

| Id | Name | Scores | `max_topk` | Threads / occupancy / B / B2 / TMA depth |
| ---: | --- | --- | ---: | --- |
| 0 | `f_t512_b2_2k_d3` | fp32 | 2048 | 512 / 1 / 8192 / 2048 / 3 |
| 1 | `h_t256_o2_b2_2k_d3` | bf16 | 2048 | 256 / 2 / 4096 / 2048 / 3 |
| 2 | `f_t256_b2_4k_d3_k4096` | fp32 | 4096 | 256 / 1 / 4096 / 4096 / 3 |
| 3 | `h_t512_b2_4k_d3_k4096` | bf16 | 4096 | 512 / 1 / 8192 / 4096 / 3 |

Configurations 2 and 3 are stock DeepSelect's tiers for top-k in (1024, 4096].

## Build

Use a pristine DeepSelect `bfa4507` checkout with its cutlass submodule
(`ae6bccf341fb4410241f696ba06873023d5ce4ed`), and run `build.sh` inside the GLM-5.2 recipes'
container. The build needs no GPU. Run these commands from the root of a checkout of this branch;
set `DS_TUNE_CONTAINER` and `DS_TUNE_BUILD_JOB` to record the container and job in
`build_info.json` (they are empty otherwise).

```bash
git clone https://github.com/deepseek-ai/DeepSelect.git DeepSelect-bfa4507
git -C DeepSelect-bfa4507 checkout --detach bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18
git -C DeepSelect-bfa4507 submodule update --init csrc/3rdparty/cutlass
glm52_ds_tune/build.sh "$PWD/DeepSelect-bfa4507" "$PWD/ds-tune-build"
```

`build.sh` checks the inputs, generates the build tree (`gen_pkg.py`: the patched `csrc` and
the eight instantiations), builds the wheel with `SOURCE_DATE_EPOCH` set to the DeepSelect commit
time, installs it with `pip install --target` into `site/`, and runs a CPU import check. The
recipes put `site/` on `PYTHONPATH`.

The measured build ran with this toolchain:

| Tool | Version |
| --- | --- |
| Architecture | aarch64 |
| nvcc | 13.2 V13.2.78 |
| g++ | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) |
| Python | 3.12.3 |
| torch | 2.12.0a0+0291f960b6.nv26.04.48445190 (CUDA 13.2) |
| setuptools / wheel / pybind11 / ninja | 81.0.0 / 0.46.3 / 3.0.3 / 1.13.0 |

`import ds_tune` refuses a torch version other than the one it was built with.

## The measured build

| File | SHA256 |
| --- | --- |
| `ds_tune-0.1.0+bfa4507.p3-cp312-cp312-linux_aarch64.whl` | `4e9d5e819708ba032283b109578e9719761ac2be0c98a0b924a37dc78b13e51b` |
| `ds_tune/ds_tune_cuda.cpython-312-aarch64-linux-gnu.so` (22,281,832 bytes) | `d393ce337024901a1d0599f7c532acb29207b1d1d877592dcc8cdba837ef7033` |

A rebuild in another directory cannot reproduce these bytes: the `.so` embeds the absolute paths of
the build tree and of the DeepSelect checkout (`__FILE__` in host assertions, `-lineinfo`). A
rebuild with `build.sh` in the same container (2026-11-01) gave a different `.so` whose SASS matched
the measured build's for every kernel, on `sm_103a` and on `sm_100a`. To compare a rebuild's kernels
with the measured `.so` (`site/ds_tune/` of the recipes' `GLM52_DEEPSELECT` tree), run
`python3 sass_compare.py <new .so> <measured .so> <sm_103a or sm_100a> <report.json>`; its summary
must count all eight `topk_kernels` as `identical`.

The GLM-5.2 runs on GB300 (`sm_103a`) used this build. Its `sm_100a` (GB200) kernels have not
run on a GPU.
