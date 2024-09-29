# Unit tests
option(BUILD_TESTS "Build accuracy & regression tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    # Using gtests command
    include(GoogleTest)
    # Find GoogleTest
    find_package(GTest)
    find_package(Threads)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://gitee.com/mirrors/googletest/repository/archive/main.zip
        )
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
endif()

