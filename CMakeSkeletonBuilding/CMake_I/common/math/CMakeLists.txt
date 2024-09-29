# 查找当前目录下的src文件夹中的所有源文件
aux_source_directory(./src MATH_DIR)

# 生成共享库
add_library(math ${MATH_DIR})  # 确保指定 SHARED 以生成共享库

# 设置库的包含目录
target_include_directories(math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)