::!/bin/bash
set -e

if NOT EXIST dependencies (
	mkdir dependencies
)

:: Opencv
if EXIST dependencies/opencv (
	echo "CLEARING OPENCV CACHE..."
	RMDIR /s dependencies/opencv
)

echo "Checking OpenCV cache..."
if EXIST dependencies/opencv (
    echo "Found OpenCV cache. Build configuration:"
    dependencies/opencv/x64/vc15/bin/opencv_version.exe -v
) ELSE (
    cd dependencies
    echo "OpenCV cache missing. Rebuilding..."
	curl.exe https://github.com/opencv/opencv/archive/4.2.0.zip -L -o opencv.zip
    7z x opencv.zip
    cd opencv-4.2.0
    mkdir build
    cd build
    :: MSMF: see https://github.com/skvark/opencv-python/issues/263
    :: CUDA/TBB: turned off because not easy to install on Windows and we cannot easily
    :: ship this with the wheel.
    cmake .. ^
	-G"Visual Studio 15 2017 Win64" ^
	-DCMAKE_BUILD_TYPE=Release ^
	-DCMAKE_INSTALL_PREFIX=../../opencv ^
	-DBUILD_LIST=core,highgui,videoio,imgcodecs,imgproc,video ^
	-DBUILD_opencv_world=ON ^
	-DBUILD_EXAMPLES=OFF ^
	-DBUILD_DOCS=OFF ^
	-DBUILD_PERF_TESTS=OFF ^
	-DBUILD_TESTS=OFF ^
	-DBUILD_opencv_java=OFF ^
	-DBUILD_opencv_python=OFF ^
	-DWITH_OPENMP=ON ^
	-DWITH_IPP=ON ^
	-DWITH_CSTRIPES=ON ^
	-DWITH_OPENCL=ON ^
	-DWITH_CUDA=OFF ^
	-DWITH_TBB=OFF ^
	-DWITH_MSMF=OFF
    cmake --build . --target INSTALL --config Release --parallel
    cd ../..
    DEL opencv.zip /s /f /q
    RMDIR /s /q opencv-4.2.0
    cd ..
)

:: Eigen3

if EXIST dependencies/eigen3 (
	echo "CLEARING EIGEN CACHE..."
	RMDIR dependencies/eigen3
)

if EXIST dependencies/eigen3 (
    echo "Found eigen3 cache."
) ELSE (
    echo "Eigen3 cache missing. Downloading..."
    cd dependencies
    curl.exe https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip -L -o eigen.zip
    7z x eigen.zip
    cd eigen-3.3.7
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../../eigen3
    cmake --build . --target INSTALL --config Release --parallel
    cd ../..
    RMDIR /s /q eigen-3.3.7
    DEL eigen.zip /s /f /q
    cd ..
)
