cmake_minimum_required(VERSION 3.11)
project(surface_normal CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include)
link_libraries(${OpenCV_LIBS})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-O3 -Wall -Wpedantic -Wno-sign-compare)

include(FetchContent)
FetchContent_Declare(pybind11 GIT_REPOSITORY https://github.com/pybind/pybind11)
FetchContent_MakeAvailable(pybind11)
pybind11_add_module(${PROJECT_NAME} src/python.cpp src/surface_normal.cpp)
