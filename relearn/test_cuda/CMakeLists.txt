if (CUDA_FOUND)
    enable_testing()

    # add custom check target, builds and tests all tests

    # # googletest# benchmark# dont clutter the compile_commands file with libraries
    set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest
            GIT_TAG release-1.12.1)

    # google test
    set(gtest_force_shared_crt
            ON
            CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    add_library(cuda_tests STATIC)
    target_include_directories(cuda_tests PRIVATE ../source/shared)
    target_include_directories(cuda_tests PRIVATE ../source)
    target_link_libraries(cuda_tests PRIVATE relearn_harness)
    target_sources(cuda_tests PRIVATE cuda_vector/test_cuda_vector.cu
            cuda_double3/test_cuda_double3.cu
            cuda_barnes_hut/test_cuda_barnes_hut.cu
            cuda_kernel/test_cuda_bh_kernel.cu
            cuda_stack/test_cuda_stack.cu
            cuda_stack/test_cuda_coalesced_stack.cu
            cuda_random/test_cuda_random.cu
            cuda_synaptic_elements/test_cuda_synaptic_elements.cu
            cuda_neurons_extra_infos/test_cuda_neurons_extra_infos.cu
            # for visual studio
            ${relearn_tests_cuda_additional_files})

    target_include_directories(cuda_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(cuda_tests PRIVATE project_options project_libraries_gpu project_libraries)
    target_link_libraries(cuda_tests PRIVATE relearn_gpu)
    target_link_libraries(cuda_tests PRIVATE gtest)
    set_target_properties(cuda_tests PROPERTIES CXX_STANDARD 17)
    set_target_properties(cuda_tests PROPERTIES ENABLE_EXPORTS 1)
    target_compile_features(cuda_tests PRIVATE cxx_std_17 cuda_std_17)
    set_target_properties(cuda_tests PROPERTIES CMAKE_CUDA_STANDARD 17 CMAKE_CXX_STANDARD 17 CMAKE_CUDA_STANDARD_REQUIRED ON CMAKE_CXX_STANDARD_REQUIRED ON)


    add_executable(relearn_tests_cuda ${SRC_TST} RelearnTest.cpp)
    target_include_directories(relearn_tests_cuda PRIVATE ../source/shared)
    target_include_directories(relearn_tests_cuda PRIVATE ../source)
    target_link_libraries(relearn_tests_cuda PRIVATE relearn_harness)
    target_link_libraries(relearn_tests_cuda PRIVATE cuda_tests)
    target_sources(
            relearn_tests_cuda
            PRIVATE ${CMAKE_BINARY_DIR}/test_get_relearn_path.cpp
            RelearnTest.cpp
            # for visual studio
            ${relearn_tests_cuda_additional_files})

    #set_target_properties(relearn_tests_cuda PROPERTIES CUDA_ARCHITECTURES 61)

    target_include_directories(relearn_tests_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(relearn_tests_cuda PRIVATE project_options project_libraries)
    #target_link_libraries(relearn_tests_cuda PRIVATE relearn_gpu)
    #target_link_libraries(relearn_tests_cuda PRIVATE relearn_lib)
    #target_link_libraries(relearn_tests_cuda PRIVATE relearn_harness)
    target_link_libraries(relearn_tests_cuda PRIVATE gtest)
    set_target_properties(relearn_tests_cuda PROPERTIES CXX_STANDARD 20)
    set_target_properties(relearn_tests_cuda PROPERTIES ENABLE_EXPORTS 1)
    target_compile_features(relearn_tests_cuda PRIVATE cxx_std_20)
    set_target_properties(relearn_tests_cuda PROPERTIES CMAKE_CXX_STANDARD 20 CMAKE_CXX_STANDARD_REQUIRED ON)

    add_test(NAME SerialTests COMMAND relearn_tests_cuda)

    get_target_property(relearn_tests_cuda_sources relearn_tests_cuda SOURCES)
    source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${relearn_tests_cuda_sources})
endif ()
