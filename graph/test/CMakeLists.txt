include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY    https://github.com/google/googletest.git
)

FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE TOP_TST "*.cpp")
add_executable(runTests
    ${TOP_TST}
)

target_include_directories(runTests SYSTEM PRIVATE ${googletest_SRC_DIR}/include)

target_link_libraries(runTests PRIVATE gtest_main)

add_test(NAME SerialTests COMMAND runTests)

target_link_libraries(runTests PRIVATE relearn-full_graph_static)
