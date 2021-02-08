#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
OPEN="~ ~ ~ ~ ~ ~ > "
CLOSE="  ..."

check_success () {
    if [ $? -eq 0 ]; then
        echo " -- successful"
    else
        echo " -- unsuccessful"
        exit 1
    fi
}

echo "${OPEN}creating build directory for boxhed2.0 in ${DIR}/../xgboost/${CLOSE}"
cd "${DIR}/../xgboost/"
mkdir -p build
check_success

echo "${OPEN}running cmake for boxhed in ${DIR}/../xgboost/build/${CLOSE}"
cd "${DIR}/../xgboost/build/"
cmake .. -DUSE_CUDA=OFF
check_success

echo "${OPEN}running make for boxhed in ${DIR}/../xgboost/build/${CLOSE}"
cd "${DIR}/../xgboost/build/"
make -j4
check_success

echo "${OPEN}setting up boxhed for python in ${DIR}/../xgboost/python-package/${CLOSE}"
cd "${DIR}/../xgboost/python-package/"
python setup.py install
check_success


echo "${OPEN}boxhed installed successfully${CLOSE}"

####### setting up preprocessing #######

echo "${OPEN}creating build directory for preprocessor in ${DIR}/build/${CLOSE}"
cd "${DIR}"
mkdir -p build
check_success

echo "${OPEN}running cmake for preprocessor in ${DIR}/build/${CLOSE}"
cd "${DIR}/build/"
cmake ../preprocessor_installer
check_success

echo "${OPEN}running cmake --build for preprocessor in ${DIR}/build/${CLOSE}"
cmake --build .
check_success


echo "${OPEN}preprocessor installed successfully${CLOSE}"
