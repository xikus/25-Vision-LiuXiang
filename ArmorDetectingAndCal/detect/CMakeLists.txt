# cmake_minimum_required(VERSION 3.10)

# set(CMAKE_CXX_STANDARD 11)

# SET(CMAKE_C_COMPILER "g++")
# SET(CMAKE_CXX_COMPILER "g++")

# project(detect)

# add_executable(detect_test test_light_descriptor.cpp)

# add_subdirectory(detect_light)
# add_subdirectory(LightDescriptor)

# find_package(OpenCV REQUIRED)

# target_include_directories(detect_test PUBLIC ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(detect_test detect_light LightDescriptor ${OpenCV_LIBS})

project(test)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

SET(CMAKE_C_COMPILER "g++")
SET(CMAKE_CXX_COMPILER "g++")

add_subdirectory(detect_light)
add_subdirectory(LightDescriptor)
add_subdirectory(match_light)
add_subdirectory(calculate)

# set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test test_light_descriptor.cpp)

target_link_libraries(test detect_light LightDescriptor match_light cal ${OpenCV_LIBS})
