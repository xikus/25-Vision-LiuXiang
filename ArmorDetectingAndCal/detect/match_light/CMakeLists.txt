aux_source_directory(./src match_light_DIR)

# 生成共享库
add_library(match_light ${match_light_DIR})

# 设置库的包含目录#
target_include_directories(match_light PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(match_light LightDescriptor ${OpenCV_LIBS})