# 查找当前目录下的src文件夹中的所有源文件
aux_source_directory(./src/module1a module1a_DIR)
aux_source_directory(./src/module1b module1b_DIR)

# 生成共享库
add_library(module1 ${module1a_DIR} ${module1b_DIR})

# 设置库的包含目录
target_include_directories(module1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(module1 assembly2)