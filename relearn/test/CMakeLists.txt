enable_testing()

# add custom check target, builds and tests all tests

# processor count detection and option
include(ProcessorCount)
ProcessorCount(proc_count)
set(CTEST_JOB_COUNT
    "${proc_count}"
    CACHE STRING "number of jobs that run the tests")

set(check_args -j${CTEST_JOB_COUNT} --output-on-failure ${CTEST_EXTRA_ARGS})

add_custom_target(
  check
  COMMAND
    ${CMAKE_COMMAND} -E env
    $<$<BOOL:${ENABLE_SANITIZER_ADDRESS}>:ASAN_OPTIONS=fast_unwind_on_malloc=0,symbolize=1>
    $<$<BOOL:${ENABLE_SANITIZER_LEAK}>:LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/cmake/sanitizers/lsan.supp>
    ${CMAKE_CTEST_COMMAND} ${check_args}
  USES_TERMINAL)

if(${COVERAGE_REPORT_AVAILABLE})
  add_custom_target(
    check-coverage
    DEPENDS check
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target
            coverage-report)
endif()

# # googletest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG release-1.12.1)

# google test
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(
  WRITE ${CMAKE_BINARY_DIR}/test_get_relearn_path.cpp
  "#include <filesystem>\nstd::filesystem::path get_relearn_path() { return \"${CMAKE_SOURCE_DIR}\"; }"
)

add_executable(relearn_tests ${SRC_TST})
target_sources(
  relearn_tests
  PRIVATE test_background_activity.cpp
          test_barnes_hut.cpp
          test_calcium_calculator.cpp
          test_cell.cpp
          test_connector.cpp
          test_fast_gauss.cpp
          test_kernel.cpp
          test_kernel_gamma.cpp
          test_kernel_gaussian.cpp
          test_kernel_linear.cpp
          test_kernel_weibull.cpp
          test_monitor_parser.cpp
          test_mpi_rank.cpp
          test_network_graph.cpp
          test_neuron_assignment.cpp
          test_neuron_extra_info.cpp
          test_neuron_io.cpp
          test_neuron_models.cpp
          test_neurons.cpp
          test_octree.cpp
          test_partition.cpp
          test_rank_neuron_id.cpp
          test_space_filling_curve.cpp
          test_step_parser.cpp
          test_synaptic_elements.cpp
          test_synaptic_input.cpp
          test_tagged_id.cpp
          test_vector.cpp
          ${CMAKE_BINARY_DIR}/test_get_relearn_path.cpp
          RelearnTest.cpp)
target_link_libraries(relearn_tests PRIVATE project_options project_libraries)
target_link_libraries(relearn_tests PRIVATE relearn_lib)
target_link_libraries(relearn_tests PRIVATE gtest_main)
add_test(NAME SerialTests COMMAND relearn_tests)
add_dependencies(check relearn_tests)
