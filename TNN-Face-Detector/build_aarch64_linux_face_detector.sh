#!/bin/bash

CC=aarch64-linux-gnu-gcc
CXX=aarch64-linux-gnu-g++
TNN_LIB_PATH=./TNN/scripts/build_aarch64_linux/

rm -r build
mkdir build
cd build
cmake .. \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DCMAKE_BUILD_TYPE=Release \
    -DTNN_LIB_PATH=$TNN_LIB_PATH

make -j4
