# 查找当前目录下的src文件夹中的所有源文件
aux_source_directory(./src processor_DIR)
# 生成共享库
add_library(processor ${processor_DIR})  

# set(CUDA_LINK_DIRECTORIES /usr/local/cuda/lib64)

# 设置库的包含目录
target_include_directories(processor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(processor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/stb)
# target_include_directories(processor PUBLIC ${CUDA_INCLUDE_DIRS})
# target_link_libraries(processor PUBLIC ${CUDA_LIBRARIES})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
# target_link_directories(processor PUBLIC ${CUDA_LINK_DIRECTORIES})


message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_TOOLKIT_ROOT_DIR}/include")
message(STATUS "CUDA_LIBRARIES: ${CUDA_TOOLKIT_ROOT_DIR}/lib64")
