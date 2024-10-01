# 查找当前目录下的src文件夹中的所有源文件
aux_source_directory(./src core_DIR)

# 生成共享库
add_library(core ${core_DIR})

# 设置库的包含目录
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)