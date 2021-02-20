#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
OPEN="~ ~ ~ ~ ~ ~ > "
CLOSE="  ..."
setup_log="${DIR}/setup_log.txt"

check_success () {
    if [ $? -eq 0 ]; then
        echo " -- successful"
    else
        echo " -- unsuccessful, check ${setup_log}"
        exit 1
    fi
}

rm -f setup_log

echo "${OPEN}creating build directory for boxhed2.0 in ${DIR}/BoXHED2.0/${CLOSE}"
cd "${DIR}/BoXHED2.0/"
mkdir -p build &> ${setup_log}
check_success

echo "${OPEN}running cmake for boxhed in ${DIR}/BoXHED2.0/build/${CLOSE}"
cd "${DIR}/BoXHED2.0/build/"
cmake .. -DUSE_CUDA=OFF >> ${setup_log} 2>&1
check_success

echo "${OPEN}running make for boxhed in ${DIR}/BoXHED2.0/build/${CLOSE}"
cd "${DIR}/BoXHED2.0/build/"
make -j4 >> ${setup_log} 2>&1
check_success

echo "${OPEN}setting up boxhed for python in ${DIR}/BoXHED2.0/python-package/${CLOSE}"
cd "${DIR}/BoXHED2.0/python-package/"
python setup.py install >> ${setup_log} 2>&1
check_success


echo "${OPEN}boxhed installed successfully${CLOSE}"

####### setting up preprocessing #######

echo "${OPEN}creating build directory for preprocessor in ${DIR}/build/${CLOSE}"
cd "${DIR}"
mkdir -p build >> ${setup_log} 2>&1
check_success

echo "${OPEN}running cmake for preprocessor in ${DIR}/build/${CLOSE}"
cd "${DIR}/build/"
cmake ../preprocessor_installer  >> ${setup_log} 2>&1
check_success

echo "${OPEN}running cmake --build for preprocessor in ${DIR}/build/${CLOSE}"
cmake --build .  >> ${setup_log} 2>&1
check_success


echo "${OPEN}preprocessor installed successfully${CLOSE}"
