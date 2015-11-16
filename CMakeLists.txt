cmake_minimum_required(VERSION 2.8.3)
project(image_converter)

## Find catkin and any catkin packages
find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(--std=c++0x -Wall -Wextra -pedantic)

cs_add_library(${PROJECT_NAME} src/kinect_ir_16-to-8-bit.cpp)

cs_add_executable(converter src/kinect_ir_16-to-8-bit.cpp)

target_link_libraries(converter ${PROJECT_NAME} ${catkin_LIBRARIES})

cs_install()
cs_export() 
