#!/bin/bash
rm -rf ./build
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=DEBUG -DTEST_MODE_NO_HAILO_LINK:BOOL=ON ../ 
make

