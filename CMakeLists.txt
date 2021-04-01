cmake_minimum_required(VERSION 3.12)
project(examples VERSION 1.0.0)

find_package(HPX REQUIRED)

set (program matmul)

add_hpx_executable(${program}
    ESSENTIAL
    SOURCES ${program}.cpp
)
