"""Derive the 3 local DeepSelect patches as standalone .patch files (one-time, provenance).

The h2h-validated ``ds_tune2`` build got its patched kernel sources from
``glm52-topk-efficiency-20261019/deepselect/tune/gen_tune.py``, which edits a copy of
DeepSelect ``csrc`` with string substitutions. This script replays exactly those edits one
patch at a time on a pristine bfa4507 ``csrc`` copy and writes each step as a unified diff
(``a/csrc/...`` -> ``b/csrc/...``, apply with ``patch -p1`` or ``git apply`` in a DeepSelect
checkout). It then applies the three diffs in order with ``patch -p1`` to a fresh pristine copy
and requires the result to be byte-identical to the validated ``csrc_tune`` tree, so the
.patch files (not this script) are the source of truth used by ``gen_pkg.py``.

Usage: python3 make_patches.py <DeepSelect bfa4507 root> <validated csrc_tune dir> <out patches dir> <scratch dir>
"""

import difflib
import filecmp
import os
import re
import shutil
import subprocess
import sys

PATCHES = (
    ("0001-ds_tune-allow-max-topk-2048.patch",
     "Allow MAX_TOPK == 2048 in the v3 (bf16) and v3_fp32 kernels; upstream lists 512/1024/4096 only."),
    ("0002-ds_tune-per-config-window-table-capacity.patch",
     "Replace the compile-time 'window delta table holds 2^23/512 windows' static_assert by a per-config "
     "capacity constant (still >= 2^21 columns) plus a host-side vocab_size check, so B2 + B < 8192 configs "
     "(B2 = 2048 tier, bf16 occupancy 2) compile."),
    ("0003-ds_tune-private-kernel-namespaces.patch",
     "Rename topk_select_{common,fp32,bf16_normal} to tune_{common,fp32,bf16} so the ds_tune .so can never "
     "alias stock deep_select symbols."),
)
KERNEL_DIRS = {"v3_fp32": "topk_select_fp32", "v3": "topk_select_bf16_normal"}
NEW_NS = {"v3_fp32": "tune_fp32", "v3": "tune_bf16"}


def sub_once(text, old, new, path):
    if text.count(old) != 1:
        raise SystemExit(f"patch anchor not unique/absent in {path}: {old[:80]!r} (count={text.count(old)})")
    return text.replace(old, new)


def edit(root, rel, fn):
    p = os.path.join(root, rel)
    with open(p) as f:
        t = f.read()
    t2 = fn(t, p)
    if t2 == t:
        raise SystemExit(f"edit of {rel} changed nothing")
    with open(p, "w") as f:
        f.write(t2)


def copy_pristine(ds_root, dst):
    """Same file set as gen_tune.py: csrc without 3rdparty/cutlass, instantiations, ascend_kernels,
    v3_cluster, and api.cpp."""
    def ignore(d, names):
        drop = []
        if os.path.basename(d) == "3rdparty":
            drop.append("cutlass")
        drop += [n for n in names if n in ("instantiations", "ascend_kernels", "v3_cluster")]
        return drop
    shutil.copytree(os.path.join(ds_root, "csrc"), os.path.join(dst, "csrc"), ignore=ignore)
    os.remove(os.path.join(dst, "csrc", "api.cpp"))


def step1(root):
    for sub in KERNEL_DIRS:
        edit(root, f"csrc/cuda_kernels/{sub}/topk_select.cuh", lambda t, p: sub_once(
            t, "static_assert(MAX_TOPK == 512 || MAX_TOPK == 1024 || MAX_TOPK == 4096);",
            "static_assert(MAX_TOPK == 512 || MAX_TOPK == 1024 || MAX_TOPK == 2048 || MAX_TOPK == 4096);  // [ds_tune patch 1]",
            p))


def step2(root):
    edit(root, "csrc/cuda_kernels/common_parts.cuh", lambda t, p: sub_once(
        t,
        """    static_assert(NUM_EXTRA_SLOTS * sizeof(uint64_t) / sizeof(uint32_t) >= MAX_VOCAB_SIZE / NUM_ELEMS_PER_SEG,
                "the index sort's window delta table must fit in the extra pairs region");""",
        """    // [ds_tune patch 2] per-config capacity of the sorted-index window delta table, checked on the host.
    static constexpr uint64_t WINDOW_DELTA_CAPACITY_COLS =
        (uint64_t)(NUM_EXTRA_SLOTS * sizeof(uint64_t) / sizeof(uint32_t)) * NUM_ELEMS_PER_SEG;
    static constexpr uint32_t MAX_VOCAB_THIS_CFG =
        WINDOW_DELTA_CAPACITY_COLS < MAX_VOCAB_SIZE ? (uint32_t)WINDOW_DELTA_CAPACITY_COLS : MAX_VOCAB_SIZE;
    static_assert(MAX_VOCAB_THIS_CFG >= (1u << 21),
                "the index sort's window delta table must hold at least 2^21 columns");""",
        p))
    anchor = "    KU_ASSERT(args.topk <= Kernel::MAX_TOPK, \"topk is too large. Maximum allowed: %d\\n\", Kernel::MAX_TOPK);"
    for sub in KERNEL_DIRS:
        edit(root, f"csrc/cuda_kernels/{sub}/topk_select.cuh", lambda t, p: sub_once(
            t, anchor,
            anchor + "\n    KU_ASSERT(args.vocab_size <= Kernel::MAX_VOCAB_THIS_CFG, \"vocab_size exceeds this config's window table\");  // [ds_tune patch 2]",
            p))


def step3(root):
    for rel in ("csrc/cuda_kernels/common_parts.cuh", "csrc/cuda_kernels/bit_utils.cuh"):
        edit(root, rel, lambda t, p: t.replace("namespace topk_select_common", "namespace tune_common"))
    for sub, old_ns in KERNEL_DIRS.items():
        for fn in ("topk_select.cuh", "topk_select.h"):
            def ren(t, p, old_ns=old_ns, sub=sub):
                t = t.replace(f"namespace {old_ns}", f"namespace {NEW_NS[sub]}").replace(f"// {old_ns}", f"// {NEW_NS[sub]}")
                return t.replace("topk_select_common::", "tune_common::")
            edit(root, f"csrc/cuda_kernels/{sub}/{fn}", ren)
    for d, _, files in os.walk(os.path.join(root, "csrc", "cuda_kernels")):
        for f in files:
            with open(os.path.join(d, f)) as h:
                if re.search(r"topk_select_(common|fp32|bf16_normal)\b", h.read()):
                    raise SystemExit(f"stale namespace in {f}")


def tree_files(root):
    out = []
    for d, _, files in os.walk(root):
        out += [os.path.relpath(os.path.join(d, f), root) for f in files]
    return sorted(out)


def unified(a_root, b_root):
    """Deterministic unified diff (no timestamps) of every changed file under csrc/."""
    chunks = []
    files = sorted(set(tree_files(a_root)) | set(tree_files(b_root)))
    for rel in files:
        pa, pb = os.path.join(a_root, rel), os.path.join(b_root, rel)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            raise SystemExit(f"patches only modify files; {rel} added/removed")
        if filecmp.cmp(pa, pb, shallow=False):
            continue
        with open(pa) as f:
            a = f.read()
        with open(pb) as f:
            b = f.read()
        if not (a.endswith("\n") and b.endswith("\n")):
            raise SystemExit(f"{rel}: no trailing newline, difflib output would not apply")
        chunks.append(f"diff -u a/{rel} b/{rel}\n")
        chunks += difflib.unified_diff(a.splitlines(True), b.splitlines(True), f"a/{rel}", f"b/{rel}", n=3)
    return "".join(chunks)


def main(ds_root, validated, out, scratch):
    head = subprocess.run(["git", "-C", ds_root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != "bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18":
        raise SystemExit(f"DeepSelect HEAD is {head}, expected bfa4507")
    if os.path.exists(scratch):
        shutil.rmtree(scratch)
    os.makedirs(out, exist_ok=True)
    stages = [os.path.join(scratch, f"s{i}") for i in range(4)]
    copy_pristine(ds_root, stages[0])
    for i, step in enumerate((step1, step2, step3)):
        shutil.copytree(stages[i], stages[i + 1])
        step(stages[i + 1])
        body = unified(stages[i], stages[i + 1])
        name, why = PATCHES[i]
        hdr = (f"ds_tune patch {i + 1}/3 against DeepSelect bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18 "
               f"(https://github.com/deepseek-ai/DeepSelect).\n{why}\n"
               "Apply in order from the DeepSelect root: patch -p1 < 000N-*.patch (or git apply).\n\n")
        with open(os.path.join(out, name), "w") as f:
            f.write(hdr + body)
    # re-apply the written patches with patch(1) to a fresh pristine copy; must equal the validated tree
    fresh = os.path.join(scratch, "apply")
    copy_pristine(ds_root, fresh)
    for name, _ in PATCHES:
        subprocess.run(["patch", "-p1", "--forward", "--batch", "--no-backup-if-mismatch", "-s", "-d", fresh,
                        "-i", os.path.join(out, name)], check=True)
    got = tree_files(os.path.join(fresh, "csrc"))
    want = tree_files(validated)
    if got != want:
        raise SystemExit(f"file sets differ: {sorted(set(got) ^ set(want))[:10]}")
    bad = [r for r in got if not filecmp.cmp(os.path.join(fresh, "csrc", r), os.path.join(validated, r), shallow=False)]
    if bad:
        raise SystemExit(f"patched tree != validated csrc_tune: {bad}")
    print(f"OK: 3 patches reproduce the validated csrc_tune ({len(got)} files byte-identical)")


if __name__ == "__main__":
    main(*sys.argv[1:5])
