import sys
import os
import json
import subprocess
from utils import read_config_json
from setuptools import setup, Extension
import sysconfig


USE_CUDA = "ON" #ON/OFF
if USE_CUDA == "ON":
    make_nom_proc = "4"
elif USE_CUDA == "OFF":
    make_nom_proc = "$(nproc)"

config = read_config_json("config.txt")

from contextlib import redirect_stdout
boxhed_setup_log = open("boxhed_setup_log.txt", "w")

def run_cmd (cmd, cwd):
    p = subprocess.Popen(cmd.split(), 
            cwd=cwd, 
            stdout = boxhed_setup_log, stderr = boxhed_setup_log)
    p.wait()

def log (msg):
    print (msg)
    print (msg, file = boxhed_setup_log)

#######    setting up boxhed     #######

#### installing xgb

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
print ("setting up the preprocessor ... ")
import shutil
try:
    shutil.rmtree("./build")
except OSError:
    pass

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()

if "-Wstrict-prototypes" in extra_compile_args: extra_compile_args.remove("-Wstrict-prototypes")

extra_compile_args += ["-std=c++11", "-fopenmp"]

with redirect_stdout(boxhed_setup_log):
    setup(
        ext_modules=[Extension('preprocess', ['preprocess.cpp'], include_dirs = ['./include'], extra_compile_args=extra_compile_args),],
    )

boxhed_setup_log.close()
