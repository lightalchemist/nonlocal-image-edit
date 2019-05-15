cmake_minimum_required(VERSION 2.8)

project(nonlocal)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

add_compile_options(-Wall -Wextra -Wpedantic -ggdb3)
add_executable(${PROJECT_NAME} "src/main.cpp" "include/filter.hpp")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} Eigen3::Eigen)

