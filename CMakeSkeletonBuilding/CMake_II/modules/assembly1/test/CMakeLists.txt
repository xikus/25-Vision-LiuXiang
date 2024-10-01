add_executable(assembly1_test assembly1_test.cpp)

target_link_libraries(assembly1_test assembly1)

add_test(NAME assembly1_test COMMAND assembly1_test)

