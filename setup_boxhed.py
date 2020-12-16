import os
import json
import subprocess
from utils import read_config_json

USE_CUDA = "ON" #ON/OFF
if USE_CUDA == "ON":
    make_nom_proc = "4"
elif USE_CUDA == "OFF":
    make_nom_proc = "$(nproc)"

#TODO: logging instead of just printing

config = read_config_json("config.txt")
#######    setting up boxhed     #######

#### installing xgb

boxhed_setup_log = open("boxhed_setup_log.txt", "w")


print ("creating build directory for boxhed ...")
p = subprocess.Popen(['mkdir', '-p', 'build'], 
        cwd=config["boxhed_addr"],
        stdout = boxhed_setup_log, stderr = boxhed_setup_log)
p.wait()

print ("running cmake for boxhed ...")
p = subprocess.Popen(['cmake', '..', "-DUSE_CUDA=%s"%USE_CUDA], 
        cwd=os.path.join(config["boxhed_addr"], "build"),
        stdout = boxhed_setup_log, stderr = boxhed_setup_log)
p.wait()

print ("running make for boxhed ...")
p = subprocess.Popen(['make', "-j%s"%make_nom_proc], 
        cwd=os.path.join(config["boxhed_addr"], "build"),
        stdout = boxhed_setup_log, stderr = boxhed_setup_log)
p.wait()

print ("setting up boxhed for python ... ")
p = subprocess.Popen(['python', "setup.py", "install"], 
        cwd=os.path.join(config["boxhed_addr"], "python-package"),
        stdout = boxhed_setup_log, stderr = boxhed_setup_log)
p.wait()

print ("boxhed installed successfully")
####### setting up preprocessing #######

