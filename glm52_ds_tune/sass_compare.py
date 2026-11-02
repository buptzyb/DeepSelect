"""Compare the SASS of every kernel in a built ds_tune .so with a reference .so (CPU only).

For each kernel of ``new_so`` (cuobjdump -sass -arch <arch>), find the reference kernel with the
same demangled name after mapping the DeepSelect namespaces to canonical ones
(tune_fp32 == topk_select_fp32, tune_bf16 == topk_select_bf16_normal, tune_common ==
topk_select_common), and compare the full instruction listing (offsets, opcodes, encodings).
Identical SASS means the kernel the GPU runs is the one the reference build ran.

Usage: python3 sass_compare.py <new_so> <ref_so> <arch> <out.json>
Prints one summary line; exit 0 always (the caller decides which kernels must match).
"""

import json
import os
import re
import shutil
import subprocess
import sys

NS_CANON = {
    "tune_fp32": "NSFP32", "topk_select_fp32": "NSFP32",
    "tune_bf16": "NSBF16", "topk_select_bf16_normal": "NSBF16",
    "tune_common": "NSCOMMON", "topk_select_common": "NSCOMMON",
}
_MANGLED_NS = re.compile(r"\d+(tune_fp32|tune_bf16|tune_common|topk_select_fp32|topk_select_bf16_normal|topk_select_common)")
_PLAIN_NS = re.compile(r"\b(tune_fp32|tune_bf16|tune_common|topk_select_fp32|topk_select_bf16_normal|topk_select_common)\b")


def _demangler():
    for cand in ("cu++filt", "/usr/local/cuda/bin/cu++filt", "c++filt"):  # kernel names are Itanium-mangled
        path = shutil.which(cand) or (cand if os.path.isfile(cand) else None)
        if path:
            return path
    raise SystemExit("no cu++filt / c++filt found")


DEMANGLER = _demangler()


def canon(text):
    text = _MANGLED_NS.sub(lambda m: NS_CANON[m.group(1)], text)
    return _PLAIN_NS.sub(lambda m: NS_CANON[m.group(1)], text)


def kernels(so, arch):
    out = subprocess.run(["cuobjdump", "-sass", "-arch", arch, so], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, check=True).stdout
    funcs, cur, body = {}, None, []
    for line in out.splitlines():
        m = re.match(r"\s+Function : (\S+)", line)
        if m or line.startswith("Fatbin") or re.match(r"\s+\.{10,}", line):
            if cur is not None:
                funcs.setdefault(cur, []).append("\n".join(body))
            cur, body = (m.group(1) if m else None), []
            continue
        if cur is not None:
            body.append(line.rstrip())
    if cur is not None:
        funcs.setdefault(cur, []).append("\n".join(body))
    names = sorted(funcs)
    dem = subprocess.run([DEMANGLER], input="\n".join(names) + "\n", stdout=subprocess.PIPE, text=True,
                         check=True).stdout.splitlines()
    res = {}
    if len(dem) != len(names):
        raise SystemExit(f"demangler returned {len(dem)} lines for {len(names)} names")
    for mangled, d in zip(names, dem):
        # A kernel can occur in several cubins (one per translation unit, e.g. cub's EmptyKernel);
        # compare the sorted list of its bodies.
        bodies = sorted(canon(b) for b in funcs[mangled])
        res[canon(d)] = dict(mangled=mangled, demangled=d, sass=bodies, copies=len(bodies),
                             n_instr=sum(1 for ln in bodies[0].splitlines() if re.match(r"\s+/\*[0-9a-f]{4,}\*/", ln)))
    return res


def is_topk_kernel(key):
    return "NSFP32::" in key or "NSBF16::" in key


def main(new_so, ref_so, arch, out):
    new, ref = kernels(new_so, arch), kernels(ref_so, arch)
    rows = []
    for key, k in sorted(new.items()):
        r = ref.get(key)
        status = "missing_in_ref" if r is None else ("identical" if r["sass"] == k["sass"] else "different")
        rows.append(dict(kernel=k["demangled"], ref_kernel=r["demangled"] if r else None, status=status,
                         topk_kernel=is_topk_kernel(key), copies=k["copies"], ref_copies=r["copies"] if r else None,
                         n_instr=k["n_instr"], ref_n_instr=r["n_instr"] if r else None))
    summary = {}
    for scope, sel in (("topk_kernels", lambda x: x["topk_kernel"]), ("other_kernels", lambda x: not x["topk_kernel"])):
        summary[scope] = {s: sum(1 for x in rows if sel(x) and x["status"] == s)
                          for s in ("identical", "different", "missing_in_ref")}
    with open(out, "w") as f:
        json.dump(dict(new_so=new_so, ref_so=ref_so, arch=arch, demangler=DEMANGLER, summary=summary, kernels=rows), f, indent=1)
        f.write("\n")
    print(f"sass_compare {arch} new={len(new)} ref={len(ref)} {summary} -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
