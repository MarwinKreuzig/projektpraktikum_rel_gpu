# benchmark
set(BENCHMARK_ENABLE_TESTING
    OFF
    CACHE INTERNAL "")
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.6.1)

FetchContent_MakeAvailable(googlebenchmark)

add_executable(
  relearn_benchmarks
  main.cpp
  benchmark_background.cpp
  benchmark_calcium.cpp
  benchmark_input_calculator.cpp
  benchmark_kernel.cpp
  benchmark_network_graph.cpp
  benchmark_neuron_model.cpp
  # for visual studio
  ${relearn_benchmark_additional_files})

target_link_libraries(relearn_benchmarks PRIVATE project_options
                                                 project_libraries)
target_link_libraries(relearn_benchmarks PRIVATE relearn_lib relearn_adapter)
target_link_libraries(relearn_benchmarks PRIVATE benchmark::benchmark)

get_target_property(relearn_benchmark_sources relearn_benchmarks SOURCES)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${relearn_benchmark_sources})
