#!/bin/bash
rm -rf ./build
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=DEBUG -D UDP_AI_DETECTION:BOOL=ON -D HAILO_AI_DETECTION:BOOL=OFF -D DDEBUG=OFF ../ 
make

