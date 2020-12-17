import sys
import os
import json
import subprocess
from utils import read_config_json
from setuptools import setup, Extension
import sysconfig
import shutil


USE_CUDA = "ON" #ON/OFF
if USE_CUDA == "ON":
    make_nom_proc = "4"
elif USE_CUDA == "OFF":
    make_nom_proc = "$(nproc)"

config = read_config_json("config.txt")
log_file = "boxhed_setup_log.txt"

from contextlib import redirect_stdout

def _write_mode(filename):
    if os.path.exists(filename):
        return 'a' # append if already exists
    else:
        return 'w' # make a new file if not

def run_cmd (cmd, cwd):

    with open(log_file, _write_mode(log_file)) as f:

        p = subprocess.Popen(cmd.split(), 
                cwd=cwd, 
                stdout = f, stderr = f)
        p.wait()

def log (msg):
    print (msg)
    with open(log_file, _write_mode(log_file)) as f:
        f.write ("~~"*10 + " "*5 + msg + " "*5 + "~~"*10 + '\n')

#######    setting up boxhed     #######

#### installing xgb

if __name__ == "__main__":
    try:
        os.remove(log_file)
    except OSError:
        pass


    log ("creating build directory for boxhed ...")
    run_cmd("mkdir -p build", 
            config["boxhed_addr"])

    log ("running cmake for boxhed ...")
    run_cmd("cmake .. -DUSE_CUDA=%s"%USE_CUDA,
            os.path.join(config["boxhed_addr"], "build"))


    log ("running make for boxhed ...")
    run_cmd("make -j%s"%make_nom_proc,
            os.path.join(config["boxhed_addr"], "build"))

    log ("setting up boxhed for python ... ")
    run_cmd ("python setup.py install",
        os.path.join(config["boxhed_addr"], "python-package"))

    log ("boxhed installed successfully")


    ####### setting up preprocessing #######
    log ("setting up the preprocessor ... ")
    try:
        shutil.rmtree("./build")
    except OSError:
        pass

    extra_compile_args = sysconfig.get_config_var('CFLAGS').split()

    if "-Wstrict-prototypes" in extra_compile_args: extra_compile_args.remove("-Wstrict-prototypes")

    extra_compile_args += ["-std=c++11", "-fopenmp"]

    with open(log_file, _write_mode(log_file)) as f:
        with redirect_stdout(f):
            setup(
                ext_modules=[Extension('preprocess', ['preprocess.cpp'], include_dirs = ['./include'], extra_compile_args=extra_compile_args),],
            )

