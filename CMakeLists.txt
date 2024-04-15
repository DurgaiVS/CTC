cmake_minimum_required(VERSION 3.5)

project(ZCTC)

set(CMAKE_CXX_STANDARD 17)
set(KENLM_MAX_ORDER 8)
set(BUILD_SHARED_LIBS ON)

if(NOT DEFINED PYTHON_EXECUTABLE OR NOT DEFINED CMAKE_INSTALL_PREFIX OR NOT DEFINED PYTHON_INCLUDE_DIR OR NOT DEFINED FST_DIR)
    message(FATAL_ERROR "Please specify PYTHON_EXECUTABLE and CMAKE_INSTALL_PREFIX and PYTHON_INCLUDE_DIR and FST_DIR")
endif()

add_compile_options(-Wall -pedantic -fPIC -DKENLM_MAX_ORDER=6)
find_package(Boost REQUIRED)
include(FetchContent)

FetchContent_Declare(
    kenlm_external
    GIT_REPOSITORY https://github.com/kpu/kenlm.git
)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.10.4
)

FetchContent_MakeAvailable(pybind11 kenlm_external)

file(GLOB FST_SOURCES ${FST_DIR}/src/lib/*.cc ${FST_DIR}/src/script/*.cc)

pybind11_add_module(_zctc SHARED ${CMAKE_SOURCE_DIR}/src/binding.cpp ${FST_SOURCES})
include_directories(${CMAKE_SOURCE_DIR}/include ${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR} ${FST_DIR}/src/include ${FETCHCONTENT_BASE_DIR}/kenlm_external-src)

target_link_libraries(_zctc PUBLIC ${PYTHON_LIBRARIES} kenlm_filter kenlm_builder kenlm_util kenlm z bz2 lzma pthread dl util)
install(TARGETS _zctc DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

add_executable(zctc ${CMAKE_SOURCE_DIR}/src/main.cpp ${FST_SOURCES})
target_link_libraries(zctc PUBLIC ${PYTHON_LIBRARIES} kenlm_filter kenlm_builder kenlm_util kenlm z bz2 lzma pthread dl util)
install(TARGETS zctc DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

set_target_properties(_zctc zctc kenlm_filter kenlm_builder kenlm_util kenlm PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
