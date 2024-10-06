# 查找当前目录下的src文件夹中的所有源文件
aux_source_directory(./src M2_DIR)

# 生成共享库
add_library(M2 ${M2_DIR}) 

# 设置库的包含目录
target_include_directories(M2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_include_directories(M2 PUBLIC ${kalman})

target_link_libraries(M2 A1 A2)