cmake_minimum_required(VERSION 2.8)

project(nonlocal)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

option(USE_BLAS "Use BLAS and LAPACK" OFF)
if (USE_BLAS)
    find_package(BLAS REQUIRED)
    include_directories( ${BLAS_INCLUDE_DIRS} )
    find_package(LAPACK REQUIRED)
    include_directories( ${LAPACK_INCLUDE_DIRS} )
    add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACK)
endif()

option(USE_SPECTRA "Use SPECTRA for eigendecomposition" OFF)
if (USE_SPECTRA)
    add_definitions(-DUSE_SPECTRA)
endif()

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

# option(USE_OPENMP "Use OpenMP" OFF)
# if (USE_OPENMP)
#     find_package(OpenMP)
#     # if(OpenMP_CXX_FOUND)
#     #     target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
#     # endif()
# endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message("Release mode")
    add_compile_options(-march=native)
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode")
    add_compile_options(-ggdb3)
endif()

add_executable(${PROJECT_NAME} "src/main.cpp" "include/filter.hpp")
add_compile_options(-Wall -Wextra -Wpedantic)
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} Eigen3::Eigen)
