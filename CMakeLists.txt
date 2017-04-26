cmake_minimum_required(VERSION 3.0)

project(gnns)

find_package(CUDA REQUIRED)

# options for NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)

# Specify the include directories
include_directories(
    src
    src/algorithms
    src/util
    src/io
    src/evaluation
    )

set (CMAKE_CXX_FLAGS "--std=c++11")
set (CUDA_PROPAGATE_HOST_FLAGS ON)

cuda_add_executable(gnns examples/main.cu)
