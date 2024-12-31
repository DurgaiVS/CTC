cmake_minimum_required(VERSION 3.5)

project(ZCTC)

set(CMAKE_CXX_STANDARD 17)
set(BUILD_SHARED_LIBS ON)

if(NOT DEFINED PYTHON_EXECUTABLE OR NOT DEFINED CMAKE_INSTALL_PREFIX OR NOT DEFINED PYTHON_INCLUDE_DIR OR NOT DEFINED FST_DIR)
    message(FATAL_ERROR "Please specify PYTHON_EXECUTABLE and CMAKE_INSTALL_PREFIX and PYTHON_INCLUDE_DIR and FST_DIR")
endif()

set(KENLM_MAX_ORDER 4)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -fPIC -DKENLM_MAX_ORDER=${KENLM_MAX_ORDER}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

find_package(Boost REQUIRED)
find_library(PTHREAD NAMES pthread REQUIRED)
find_library(DL NAMES dl REQUIRED)
find_library(UTIL NAMES util REQUIRED)
find_library(Z_LIB NAMES z)
find_library(BZ2_LIB NAMES bz2)
find_library(LZMA_LIB NAMES lzma)

include(FetchContent)

FetchContent_Declare(
    kenlm_external
    GIT_REPOSITORY https://github.com/kpu/kenlm.git
    GIT_TAG e504a4d61c413873e7d5f13d8ac4890bd4dae36c
)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.10.4
)
FetchContent_Declare(
    threadpool_external
    GIT_REPOSITORY https://github.com/progschj/ThreadPool.git
    GIT_TAG 9a42ec13
)

FetchContent_MakeAvailable(pybind11 kenlm_external threadpool_external)

file(GLOB FST_SOURCES ${FST_DIR}/src/lib/*.cc ${FST_DIR}/src/script/*.cc)

pybind11_add_module(_zctc SHARED ${CMAKE_SOURCE_DIR}/src/binding.cpp ${FST_SOURCES})
include_directories(${CMAKE_SOURCE_DIR}/include ${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR} ${FST_DIR}/src/include ${FETCHCONTENT_BASE_DIR}/kenlm_external-src ${FETCHCONTENT_BASE_DIR}/threadpool_external-src)

target_link_libraries(_zctc PUBLIC ${PYTHON_LIBRARIES} kenlm_filter kenlm_builder kenlm_util kenlm pthread dl util)
if(TARGET z)
    target_link_libraries(_zctc PUBLIC z)
endif()
if(TARGET bz2)
    target_link_libraries(_zctc PUBLIC bz2)
endif()
if(TARGET lzma)
    target_link_libraries(_zctc PUBLIC lzma)
endif()

install(TARGETS _zctc LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set_target_properties(_zctc kenlm_filter kenlm_builder kenlm_util kenlm PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_executable(zctc ${CMAKE_SOURCE_DIR}/src/main.cpp ${FST_SOURCES})
    target_link_libraries(zctc PUBLIC ${PYTHON_LIBRARIES} kenlm_filter kenlm_builder kenlm_util kenlm pthread dl util)
    if(TARGET z)
        target_link_libraries(zctc PUBLIC z)
    endif()
    if(TARGET bz2)
        target_link_libraries(zctc PUBLIC bz2)
    endif()
    if(TARGET lzma)
        target_link_libraries(zctc PUBLIC lzma)
    endif()

    install(TARGETS zctc RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
    set_target_properties(zctc PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY})
endif()
