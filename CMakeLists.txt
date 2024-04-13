cmake_minimum_required(VERSION 3.5)

project(ZCTC)

set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED PYTHON_EXECUTABLE OR NOT DEFINED CMAKE_INSTALL_PREFIX OR NOT DEFINED PYTHON_INCLUDE_DIR)
    message(FATAL_ERROR "Please specify PYTHON_EXECUTABLE and CMAKE_INSTALL_PREFIX and PYTHON_INCLUDE_DIR")
endif()

add_compile_options(-Wall -pedantic -fPIC)


include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.10.4
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_zctc SHARED ${CMAKE_SOURCE_DIR}/src/binding.cpp)
include_directories(${CMAKE_SOURCE_DIR}/include ${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR})
link_libraries(_zctc ${PYTHON_LIBRARIES})
install(TARGETS _zctc DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

set_target_properties(_zctc PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    add_executable(zctc ${CMAKE_SOURCE_DIR}/src/main.cpp)
    install(TARGETS zctc DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endif()
