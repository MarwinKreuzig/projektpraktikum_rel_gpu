# benchmark
set(BENCHMARK_ENABLE_TESTING
    OFF
    CACHE INTERNAL "")
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark
  GIT_TAG v1.7.0)

# benchmark
FetchContent_MakeAvailable(benchmark)

add_executable(relearn_benchmarks)

target_sources(
  relearn_benchmarks
  PRIVATE main.cpp
          benchmark_kernel.cpp
          benchmark_network_graph.cpp
          benchmark_neuron_model.cpp)

target_link_libraries(relearn_benchmarks PRIVATE project_options
                                                 project_libraries)
target_link_libraries(relearn_benchmarks PRIVATE relearn_lib)
target_link_libraries(relearn_benchmarks PRIVATE benchmark::benchmark)
