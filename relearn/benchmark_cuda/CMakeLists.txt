# benchmark# dont clutter the compile_commands file with libraries
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

set(BENCHMARK_ENABLE_TESTING
        OFF
        CACHE INTERNAL "")
FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.6.1)

FetchContent_MakeAvailable(googlebenchmark)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

get_target_property(googlebenchmark_includes benchmark
        INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(benchmark PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
        "${googlebenchmark_includes}")

add_executable(
        relearn_benchmarks_cuda
        main.cpp
        benchmark_octree_gpu.cpp
        benchmark_barnes_hut.cpp

        # for visual studio

        # fehlen noch libaries
        ${relearn_benchmark_additional_files})

target_link_libraries(relearn_benchmarks_cuda PUBLIC relearn_gpu)


target_link_libraries(relearn_benchmarks_cuda PRIVATE project_options
        project_libraries)
target_link_libraries(relearn_benchmarks_cuda PRIVATE relearn_lib relearn_harness)
target_link_libraries(relearn_benchmarks_cuda PRIVATE benchmark::benchmark)
set_target_properties(relearn_benchmarks_cuda PROPERTIES CXX_STANDARD 20)

get_target_property(relearn_benchmark_sources relearn_benchmarks_cuda SOURCES)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${relearn_benchmark_sources})
