#!/bin/bash
rm -rf ./build
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=DEBUG -D TEST_MODE_NO_HAILO_LINK:BOOL=OFF ../ 
make

