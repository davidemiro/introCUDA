cmake_minimum_required(VERSION 3.27)
project(introCUDA CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(introCUDA main.cu)

set_target_properties(introCUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
