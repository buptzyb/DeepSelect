"""Generate the trimmed ``ds_tune`` build tree: pristine DeepSelect csrc + patches/*.patch +
explicit kernel instantiations for the configurations the GLM-5.2 DSA top-k adapter uses.

Derived from glm52-topk-efficiency-20261019/deepselect/tune/gen_tune.py (the generator of the
h2h-validated 18-config ``ds_tune2`` build). Differences:
  * the 3 DeepSelect edits come from ``patches/000{1,2,3}-*.patch`` applied with ``patch -p1``
    (make_patches.py proved they reproduce gen_tune.py's csrc_tune byte for byte);
  * 4 configs instead of 18: fp32/bf16 x int32 indices x max_topk 2048/4096 (table below);
  * per-config max_topk on the host (the 4096 tiers), a get_alignment_requirement() export.

Every instantiation is TopkSelectConfig<ValueT, int32_t, sorted_value=false, sorted_index in
{false, true}, return_value=true, MAX_TOPK, THREADS, OCC, B, B2, TMA_DEPTH, 512, 1>.

Usage: python3 gen_pkg.py <DeepSelect bfa4507 root> <pkg dir (holds patches/, ds_tune/, *.in)> <out build dir>
"""

import json
import os
import re
import shutil
import subprocess
import sys

# (name, dtype, max_topk, threads, occupancy, B, B2, tma_depth, provenance)
CONFIGS = [
    ("f_t512_b2_2k_d3", "float", 2048, 512, 1, 8192, 2048, 3,
     "h2h winner ds_tuned (ds_tune2 cfg 8); jobs 4627869/4628005/4643155"),
    ("h_t256_o2_b2_2k_d3", "nv_bfloat16", 2048, 256, 2, 4096, 2048, 3,
     "fastest bf16 tuning point (ds_tune2 cfg 14, S3 4.04 ms); job 4628005; not in the h2h"),
    ("f_t256_b2_4k_d3_k4096", "float", 4096, 256, 1, 4096, 4096, 3,
     "stock DeepSelect fp32 big-topk tier (topk in (1024, 4096]), = h2h ds_stock at k=2048"),
    ("h_t512_b2_4k_d3_k4096", "nv_bfloat16", 4096, 512, 1, 8192, 4096, 3,
     "stock DeepSelect bf16 big-topk tier (topk in (1024, 4096])"),
]
NS = {"float": "tune_fp32", "nv_bfloat16": "tune_bf16"}
KERNEL_HDR = {"float": "v3_fp32", "nv_bfloat16": "v3"}
PATCHES = (
    "0001-ds_tune-allow-max-topk-2048.patch",
    "0002-ds_tune-per-config-window-table-capacity.patch",
    "0003-ds_tune-private-kernel-namespaces.patch",
)


def copy_pristine(ds_root, dst):
    """csrc without 3rdparty/cutlass (used in place), instantiations, ascend_kernels, v3_cluster, api.cpp."""
    def ignore(d, names):
        drop = []
        if os.path.basename(d) == "3rdparty":
            drop.append("cutlass")
        drop += [n for n in names if n in ("instantiations", "ascend_kernels", "v3_cluster")]
        return drop
    shutil.copytree(os.path.join(ds_root, "csrc"), os.path.join(dst, "csrc"), ignore=ignore)
    os.remove(os.path.join(dst, "csrc", "api.cpp"))


def main(ds_root, pkg, out):
    head = subprocess.run(["git", "-C", ds_root, "rev-parse", "HEAD"], stdout=subprocess.PIPE,
                          text=True, check=True).stdout.strip()
    if head != "bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18":
        raise SystemExit(f"DeepSelect HEAD is {head}, expected bfa4507")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; generate into a fresh directory")
    os.makedirs(out)
    stage = os.path.join(out, "_stage")
    copy_pristine(ds_root, stage)
    for name in PATCHES:
        subprocess.run(["patch", "-p1", "--forward", "--batch", "--no-backup-if-mismatch", "-s",
                        "-d", stage, "-i", os.path.join(pkg, "patches", name)], check=True)
    rejects = [os.path.join(d, f) for d, _, fs in os.walk(stage) for f in fs if f.endswith((".rej", ".orig"))]
    if rejects:
        raise SystemExit(f"patch left {rejects}")
    os.rename(os.path.join(stage, "csrc"), os.path.join(out, "csrc_tune"))
    os.rmdir(stage)
    for root, _, files in os.walk(os.path.join(out, "csrc_tune", "cuda_kernels")):
        for f in files:
            with open(os.path.join(root, f)) as h:
                if re.search(r"topk_select_(common|fp32|bf16_normal)\b", h.read()):
                    raise SystemExit(f"stale namespace in {f}")

    shutil.copytree(os.path.join(pkg, "ds_tune"), os.path.join(out, "ds_tune"),
                    ignore=shutil.ignore_patterns("__pycache__", "configs.json", "build_info.json"))
    shutil.copy(os.path.join(pkg, "setup.py"), out)
    gen = os.path.join(out, "gen")
    os.makedirs(gen)
    sources, cases, meta = [], [], []
    for cid, (name, dtype, mk, th, occ, b, b2, d, why) in enumerate(CONFIGS):
        assert b == th * 16, name
        for si in (0, 1):
            cfg = (f"TopkSelectConfig<{dtype}, int32_t, false, {'true' if si else 'false'}, true, "
                   f"{mk}, {th}, {occ}, {b}, {b2}, {d}, 512, 1>")
            fn = os.path.join(gen, f"{name}_si{si}.cu")
            with open(fn, "w") as f:
                f.write(f'#include "cuda_kernels/{KERNEL_HDR[dtype]}/topk_select.cuh"\n\n'
                        f"namespace {NS[dtype]} {{\ntemplate void run_topk_select_kernel<\n    {cfg}\n>(const TopkSelectArgs &args);\n}}\n")
            sources.append(os.path.relpath(fn, out))
            cases.append(f"        case {cid * 2 + si}: {NS[dtype]}::run_topk_select_kernel<{cfg}>(args); break;")
        meta.append(dict(id=cid, name=name, dtype="fp32" if dtype == "float" else "bf16", index_dtype="int32",
                         max_topk=mk, threads=th, occupancy=occ, B=b, B2=b2, tma_depth=d,
                         sorted_value=False, sorted_index=[False, True], return_value=True, provenance=why))
    with open(os.path.join(pkg, "api_tune.cpp.in")) as f:
        api = f.read()
    api = api.replace("@@CASES@@", "\n".join(cases)).replace("@@NUM_CFG@@", str(len(CONFIGS)))
    api = api.replace("@@FP32FLAGS@@", ", ".join("true" if c[1] == "float" else "false" for c in CONFIGS))
    api = api.replace("@@MAXTOPK@@", ", ".join(str(c[2]) for c in CONFIGS))
    if "@@" in api:
        raise SystemExit("unfilled placeholder in api_tune.cpp.in")
    with open(os.path.join(out, "api_tune.cpp"), "w") as f:
        f.write(api)
    with open(os.path.join(out, "ds_tune", "configs.json"), "w") as f:
        json.dump(meta, f, indent=1)
        f.write("\n")
    with open(os.path.join(out, "sources.json"), "w") as f:
        json.dump(["api_tune.cpp"] + sources, f, indent=1)
        f.write("\n")
    print(f"generated {len(sources)} instantiations for {len(CONFIGS)} configs in {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
