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
        relearn_benchmarks
        main.cpp
        benchmark_background.cpp
        benchmark_calcium.cpp
        benchmark_input_calculator.cpp
        benchmark_kernel.cpp
        benchmark_network_graph.cpp
#        benchmark_neuron_model.cpp

        # for visual studio

        # fehlen noch libaries
        ${relearn_benchmark_additional_files})

if (CUDA_FOUND)
    target_sources(relearn_benchmarks PRIVATE benchmark_octree_gpu.cpp)
    target_link_libraries(relearn_benchmarks PUBLIC relearn_gpu)
endif ()

target_link_libraries(relearn_benchmarks PRIVATE project_options
        project_libraries)
target_link_libraries(relearn_benchmarks PRIVATE relearn_lib relearn_harness)
target_link_libraries(relearn_benchmarks PRIVATE benchmark::benchmark)
set_target_properties(relearn_benchmarks PROPERTIES CXX_STANDARD 20)

get_target_property(relearn_benchmark_sources relearn_benchmarks SOURCES)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${relearn_benchmark_sources})
