"""ds_tune: DeepSelect bfa4507 + 3 local patches, trimmed to the GLM-5.2 DSA top-k configs.

Build from the tree written by gen_pkg.py, in one setup.py process (see build-ds-pkg.sbatch):
    DS_SRC=<DeepSelect bfa4507 root> python3 setup.py bdist_wheel -d <wheel dir>
Compile flags are the ones of the validated ds_tune2 build (glm52-topk-efficiency-20261019
deepselect/tune/setup.py) plus an sm_100a gencode (stock DeepSelect builds sm_100a + sm_103a).
"""

import json
import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

VERSION = "0.1.0+bfa4507.p3"
HERE = Path(__file__).resolve().parent
DS_SRC = Path(os.environ["DS_SRC"]).resolve()  # pristine DeepSelect clone (for cutlass + kernelkit)
sys.path.insert(0, str(DS_SRC))
import tests.kernelkit as kk  # noqa: E402

sources = json.load(open(HERE / "sources.json"))
cc_flag = [
    "-gencode", "arch=compute_100a,code=sm_100a",  # GB200 (built, not GPU-validated)
    "-gencode", "arch=compute_103a,code=sm_103a",  # GB300 (validated)
]


class ReportSpillBuild(BuildExtension):
    """Build, then run DeepSelect's own spill check and RECORD (not fail on) spills."""

    def run(self):
        super().run()
        for ext in self.extensions:
            so = self.get_ext_fullpath(ext.name)
            spills = kk.check_kernel_reg_spill_in_artifact(so, stack_baseline=8)
            (Path(so).parent / "spills.json").write_text(json.dumps(spills, indent=1) + "\n")


setup(
    name="ds_tune",
    version=VERSION,
    description="DeepSelect bfa4507 + 3 local patches (GLM-5.2 DSA indexer top-k, trimmed)",
    packages=["ds_tune"],
    package_data={"ds_tune": ["configs.json", "build_info.json"]},
    ext_modules=[CUDAExtension(
        name="ds_tune.ds_tune_cuda",
        sources=sources,
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20", "-DNDEBUG", "-Wno-deprecated-declarations",
                    "-DKERUTILS_IS_BUILD_ON_CUDA", "-DDEEP_SELECT_IS_BUILD_ON_CUDA"],
            "nvcc": ["-O3", "-std=c++20", "-DDEEP_SELECT_IS_BUILD_ON_CUDA", "-Wno-deprecated-declarations",
                     "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                     "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                     "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math", "--ftz=false",
                     "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-double-precision-use",
                     "-lineinfo", "--threads", os.environ.get("NVCC_THREADS", "2")] + cc_flag,
        },
        include_dirs=[HERE / "csrc_tune", DS_SRC / "csrc" / "3rdparty" / "cutlass" / "include",
                      HERE / "csrc_tune" / "3rdparty" / "kerutils" / "include",
                      Path(CUDA_HOME) / "targets" / "sbsa-linux" / "include" / "cccl"],
        extra_link_args=[f'-L{Path(CUDA_HOME) / "targets" / "sbsa-linux" / "lib" / "stubs"}', "-lcuda"],
    )],
    cmdclass={"build_ext": ReportSpillBuild},
    zip_safe=False,
)
