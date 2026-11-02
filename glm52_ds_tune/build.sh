#!/usr/bin/env bash
# Build ds_tune 0.1.0+bfa4507.p3: DeepSelect bfa4507 plus the three patches in patches/, trimmed
# to the four GLM-5.2 DSA top-k kernel configurations listed in gen_pkg.py.
#
# This is the procedure of the measured build (a CPU-only Slurm job, 4643789) without its
# cluster-specific Slurm, path and reference-binary settings. Run it in the GLM-5.2 recipes'
# container (aarch64, nvcc 13.2, g++ 13.3, Python 3.12, torch 2.12.0a0+0291f960b6); no GPU is
# needed. See README.md.
#
# usage: build.sh <pristine DeepSelect bfa4507 checkout> <new output directory>
set -euo pipefail
usage='usage: build.sh <pristine DeepSelect bfa4507 checkout> <new output directory>'
DS_SRC=$(realpath "${1:?${usage}}")
OUT=$(realpath -m "${2:?${usage}}")
PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 1) inputs: DeepSelect commit with a clean csrc, the cutlass pin, the patch hashes
[[ $(git -C "${DS_SRC}" rev-parse HEAD) == bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18 ]]
[[ -z $(git -C "${DS_SRC}" status --porcelain --ignore-submodules=all --untracked-files=no) ]]
[[ $(git -C "${DS_SRC}/csrc/3rdparty/cutlass" rev-parse HEAD) == ae6bccf341fb4410241f696ba06873023d5ce4ed ]]
[[ -z $(git -C "${DS_SRC}/csrc/3rdparty/cutlass" status --porcelain -- include) ]]
(cd "${PKG}/patches" && sha256sum -c "${PKG}/PATCHES.sha256")
[[ ! -e ${OUT} ]] || { echo "${OUT} exists; give a new directory" >&2; exit 1; }
mkdir -p "${OUT}"
cd "${OUT}"
export DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA PYTHONNOUSERSITE=1 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} \
    MAX_JOBS=${MAX_JOBS:-56} NVCC_THREADS=${NVCC_THREADS:-2} PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
export TMPDIR=${OUT}/tmp
mkdir -p "${TMPDIR}"

# 2) toolchain
{ uname -m; nvcc --version | tail -2; g++ --version | head -1; ld --version | head -1;
  python3 --version; python3 -c 'import torch,setuptools,wheel,pybind11;print("torch",torch.__version__,"cuda",torch.version.cuda,"setuptools",setuptools.__version__,"wheel",wheel.__version__,"pybind11",pybind11.__version__)' 2>/dev/null;
  python3 -c 'import ninja; print("ninja", ninja.__version__)'; patch --version | head -1; git --version; } > toolchain.txt 2>&1
cat toolchain.txt

# 3) generate the build tree from pristine csrc + patches, write build_info.json
#    (DS_TUNE_CONTAINER and DS_TUNE_BUILD_JOB fill its container and build_job fields)
python3 "${PKG}/gen_pkg.py" "${DS_SRC}" "${PKG}" tree
PKG=${PKG} DS_SRC=${DS_SRC} python3 - <<'PY'
import json, os, subprocess, torch
pkg, ds = os.environ['PKG'], os.environ['DS_SRC']
sh = lambda *c: subprocess.run(c, stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
patches = {}
for line in open(f'{pkg}/PATCHES.sha256'):
    h, n = line.split()
    patches[n] = h
info = dict(
    package='ds_tune', version='0.1.0+bfa4507.p3',
    deepselect=dict(url='https://github.com/deepseek-ai/DeepSelect', commit=sh('git', '-C', ds, 'rev-parse', 'HEAD'),
                    cutlass_commit=sh('git', '-C', f'{ds}/csrc/3rdparty/cutlass', 'rev-parse', 'HEAD')),
    patches=patches,
    torch=str(torch.__version__), torch_cuda=torch.version.cuda,
    nvcc=sh('nvcc', '--version').splitlines()[-2], gxx=sh('g++', '--version').splitlines()[0],
    cuda_archs=['sm_100a', 'sm_103a'], container=os.environ.get('DS_TUNE_CONTAINER', ''),
    build_job=os.environ.get('DS_TUNE_BUILD_JOB', ''),
    configs=[c['name'] for c in json.load(open('tree/ds_tune/configs.json'))],
)
json.dump(info, open('tree/ds_tune/build_info.json', 'w'), indent=1)
open('tree/ds_tune/build_info.json', 'a').write('\n')
print(json.dumps(info))
PY

# 4) the wheel in ONE setup.py process, then install it into site/
export SOURCE_DATE_EPOCH=$(git -C "${DS_SRC}" log -1 --format=%ct)
( cd tree && DS_SRC=${DS_SRC} python3 setup.py bdist_wheel -d "${OUT}/wheel" ) > build.log 2>&1 || { tail -50 build.log; exit 1; }
python3 -m pip install --no-deps --no-index --no-compile --target "${OUT}/site" "${OUT}"/wheel/*.whl > pip-install.log 2>&1
rm -rf "${OUT}/site/bin"
SO=$(ls "${OUT}"/site/ds_tune/ds_tune_cuda*.so)
cp tree/build/lib*/ds_tune/spills.json spills.json
cuobjdump -res-usage "${SO}" > res-usage.txt 2>&1

# 5) CPU import check of site/ (the CUDA driver stub stands in when the node has no driver)
export PYTHONPATH=${OUT}/site
if ! python3 -c 'import ctypes; ctypes.CDLL("libcuda.so.1")' 2>/dev/null; then
  mkdir -p "${TMPDIR}/cudastub" && ln -sf "${CUDA_HOME}/targets/sbsa-linux/lib/stubs/libcuda.so" "${TMPDIR}/cudastub/libcuda.so.1"
  export LD_LIBRARY_PATH=${TMPDIR}/cudastub${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  echo 'import test: no driver libcuda.so.1 on this node, using the CUDA stub' > import-test.txt
fi
python3 - >> import-test.txt 2>&1 <<'PY'
import torch, ds_tune
print('ds_tune', ds_tune.__version__, ds_tune.__file__)
print('alignment', ds_tune.INPUT_STRIDE_ALIGNMENT_BYTES, ds_tune.OUTPUT_STRIDE_ALIGNMENT_BYTES, 'archs', ds_tune.SUPPORTED_ARCHS)
for c in ds_tune.CONFIGS:
    print('cfg', c['id'], c['name'], c['dtype'], c['max_topk'])
assert ds_tune.config_for(torch.float32, 2048)['name'] == 'f_t512_b2_2k_d3'
assert ds_tune.config_for(torch.bfloat16, 2048)['name'] == 'h_t256_o2_b2_2k_d3'
assert ds_tune.config_for(torch.float32, 4096)['name'] == 'f_t256_b2_4k_d3_k4096'
assert ds_tune.config_for(torch.bfloat16, 2049)['name'] == 'h_t512_b2_4k_d3_k4096'
assert (ds_tune.INPUT_STRIDE_ALIGNMENT_BYTES, ds_tune.OUTPUT_STRIDE_ALIGNMENT_BYTES) == (1024, 32)
for bad in ((torch.float32, 4097), (torch.float16, 64)):
    try:
        ds_tune.config_for(*bad)
        raise SystemExit(f'config_for{bad} did not raise')
    except ValueError as e:
        print('expected ValueError:', e)
print('IMPORT_OK')
PY
cat import-test.txt
grep -q IMPORT_OK import-test.txt
rm -rf "${TMPDIR}"
sha256sum wheel/*.whl site/ds_tune/ds_tune_cuda*.so | tee hashes.txt
echo DS_TUNE_BUILD_OK
