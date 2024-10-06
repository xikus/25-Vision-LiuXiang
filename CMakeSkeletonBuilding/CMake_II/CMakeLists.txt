cmake_minimum_required(VERSION 3.10)

project(Test)

add_subdirectory(modules)
add_subdirectory(common)


add_executable(server server.cpp)
add_executable(client client.cpp)

find_package(OpenCV REQUIRED)

#server的链接库
target_link_libraries(server rmath module1 module2)
target_include_directories(server PUBLIC ${singleton})

#client的链接库
target_link_libraries(client module2)
# # # # 以下内容禁止修改 # # # #
option(BUILD_A "build modules a" ON)
option(BUILD_B "build modules b" OFF)
option(BUILD_TESTS "build tests" OFF)

if(BUILD_TESTS)
enable_testing()
endif()


# 小提示: module2 需要用到 pthread，即 target_link_libraries(xxx pthread)
# # # # 以上内容禁止修改 # # # #

if(BUILD_A)
    add_definitions(-DWITH_A)
endif()

if(BUILD_B)
    add_definitions(-DWITH_B)
endif()

# message(STATUS "ASSEMBLY1_TEST variable: ${ASSEMBLY1_TEST}")
add_subdirectory(${ASSEMBLY1_TEST})



