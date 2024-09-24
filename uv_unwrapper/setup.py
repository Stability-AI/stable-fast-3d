import torch
import glob
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
)

library_name = "uv_unwrapper"


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    is_mac = True if torch.backends.mps.is_available() else False
    use_native_arch = not is_mac and os.getenv("USE_NATIVE_ARCH", "1") == "1"
    extension = CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            ("-Xclang " if is_mac else "") + "-fopenmp",
        ] + ["-march=native"] if use_native_arch else [],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["cxx"].append("-UNDEBUG")
        extra_link_args.extend(["-O0", "-g"])

    define_macros = []
    extensions = []

    this_dir = os.path.dirname(os.path.curdir)
    sources = glob.glob(
        os.path.join(this_dir, library_name, "csrc", "**", "*.cpp"), recursive=True
    )

    if len(sources) == 0:
        print("No source files found for extension, skipping extension compilation")
        return None

    extensions.append(
        extension(
            name=f"{library_name}._C",
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=[
                "c10",
                "torch",
                "torch_cpu",
                "torch_python"
            ] + ["omp"] if is_mac else [],
        )
    )

    print(extensions)

    return extensions


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=[],
    description="Box projection based UV unwrapper",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    cmdclass={"build_ext": BuildExtension},
)
