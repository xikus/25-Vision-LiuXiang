# 查找当前目录下的src文件夹中的所有源文件
aux_source_directory(./src A2_DIR)

# 生成共享库
add_library(A2 ${A2_DIR})  

# 设置库的包含目录
target_include_directories(A2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
