#!/bin/bash
set -e

cd /io

mkdir -p dependencies
cd dependencies

# we need wget and cmake, cmake is actually easier to install via pip on centos
yum install -y wget
# yum install -y tbb-devel
export PATH=/opt/python/cp36-cp36m/bin:$PATH
pip install cmake

if [[ -d ./opencv ]]
then
    echo "Found OpenCV cache."
else
    echo "OpenCV cache missing. Rebuilding..."

    wget -q -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
    unzip -q opencv.zip
    cd opencv-4.2.0
    mkdir -p build
    cd build
    cmake ..\
        -DCMAKE_BUILD_TYPE=Release\
        -DCMAKE_INSTALL_PREFIX=../../opencv\
        -DBUILD_LIST=core,highgui,videoio,imgcodecs,imgproc,video\
        -DBUILD_opencv_world=ON\
        -DBUILD_EXAMPLES=OFF\
        -DBUILD_DOCS=OFF\
        -DBUILD_PERF_TESTS=OFF\
        -DBUILD_TESTS=OFF\
        -DBUILD_opencv_java=OFF\
        -DBUILD_opencv_python=OFF\
        -DWITH_OPENMP=ON\
        -DWITH_IPP=ON\
        -DWITH_CSTRIPES=ON\
        -DWITH_OPENCL=ON\
        -DWITH_TBB=OFF\
        -DWITH_CUDA=OFF
    make -j2 && make install
    cd ../..
    rm -rf opencv.zip
    rm -rf opencv-4.2.0
fi

if [[ -d ./eigen3 ]]
then
    echo "Found eigen3 cache."
else
    echo "Eigen3 cache missing. Downloading..."
    wget -q -O eigen.zip https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip
    unzip -q eigen.zip
    cd eigen-3.3.7
    mkdir -p build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../../eigen3
    make && make install
    cd ../..
    rm -rf eigen-3.3.7
    rm -rf eigen.zip
fi
