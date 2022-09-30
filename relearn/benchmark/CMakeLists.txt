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
  benchmark_kernel.cpp
  benchmark_network_graph.cpp
  benchmark_neuron_model.cpp)

target_link_libraries(relearn_benchmarks PRIVATE project_options
                                                 project_libraries)
target_link_libraries(relearn_benchmarks PRIVATE relearn_lib)
target_link_libraries(relearn_benchmarks PRIVATE benchmark::benchmark)