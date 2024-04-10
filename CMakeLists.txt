cmake_minimum_required(VERSION 3.0)

project(ZCTC)

set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED PYTHON_EXECUTABLE OR NOT DEFINED CMAKE_INSTALL_PREFIX)
    message(FATAL_ERROR "Please specify PYTHON_EXECUTABLE and CMAKE_INSTALL_PREFIX")
endif()

add_compile_options(-Wall -pedantic -fPIC)


include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.10.4
)
FetchContent_MakeAvailable(pybind11)
# add_directory(pybind11)

pybind11_add_module(_zctc SHARED ${CMAKE_SOURCE_DIR}/src/binding.cpp)
include_directories(${CMAKE_SOURCE_DIR}/include ${pybind11_INCLUDE_DIR})
# add_executable(zctc ${CMAKE_SOURCE_DIR}/src/main.cpp)

install(TARGETS _zctc DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
# install(TARGETS zctc DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
