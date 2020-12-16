from setuptools import setup, Extension
import sysconfig

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()

if "-Wstrict-prototypes" in extra_compile_args: extra_compile_args.remove("-Wstrict-prototypes")

extra_compile_args += ["-std=c++11", "-fopenmp"]

setup(
    ext_modules=[Extension('preprocess', ['preprocess.cpp'], include_dirs = ['./include'], extra_compile_args=extra_compile_args),],
)
