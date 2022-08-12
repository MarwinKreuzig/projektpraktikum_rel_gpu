cmake_minimum_required(VERSION 3.10)
project(relearn)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BOOST_ENABLE_CMAKE ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(
    STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE
      RelWithDebInfo
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui, ccmake
  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS
             "Debug"
             "Release"
             "MinSizeRel"
             "RelWithDebInfo")
endif()

# project options
add_library(project_options INTERFACE)

option(ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
if(ENABLE_CLANG_TIDY)
  set(CMAKE_CXX_CLANG_TIDY clang-tidy)
endif()

include(FetchContent)
find_package(OpenMP)

option(ENABLE_MPI "Enable mpi" ON)
if(ENABLE_MPI)
  find_package(MPI)
endif()

if(WIN32)
  FetchContent_Declare(
    boostorg
    GIT_REPOSITORY https://github.com/boostorg/boost.git
    GIT_TAG master)

  FetchContent_GetProperties(boostorg)
  if(NOT boostorg_POPULATED)
    FetchContent_Populate(boostorg)
    add_subdirectory(${boostorg_SOURCE_DIR} ${boostorg_BINARY_DIR})
  endif()

  include_directories(${boostorg_SOURCE_DIR})
else()
  find_package(Boost REQUIRED COMPONENTS RANDOM)
endif()

option(ENABLE_SCOREP "Enable scorep" OFF)
if(ENABLE_SCOREP)
  set(SCOREP_ARGS
      ""
      CACHE STRING "Arguments for scorep")
  string(
    REPLACE ";"
            " "
            SCOREP_ARGS
            "${SCOREP_ARGS}")
  message(STATUS "scorep command: scorep ${SCOREP_ARGS}")
  message(
    NOTICE
    "scorep instrumentation enabled; targets test and benchmarks are disabled")
  message(
    NOTICE
    "Using compiler/linker launcher to run scorep triggers rebuilds, to avoid rebuilds reconfigure with:\n\tSCOREP_WRAPPER=off CC=scorep-gcc CXX=scorep-g++ cmake .. -DENABLE_SCOREP=OFF \n\tSCOREP_WRAPPER_INSTRUMENTER_FLAGS=\"--thread=omp...\" make"
  )
  file(WRITE ${CMAKE_BINARY_DIR}/scorep_launcher.sh
       "#!/bin/sh\nscorep ${SCOREP_ARGS} $@\n")
  file(
    CHMOD
    ${CMAKE_BINARY_DIR}/scorep_launcher.sh
    FILE_PERMISSIONS
    OWNER_EXECUTE
    OWNER_READ
    OWNER_WRITE)
  set(SCOREP_CMD ${CMAKE_BINARY_DIR}/scorep_launcher.sh)
  add_custom_target(scorep_launcher
                    DEPENDS ${CMAKE_BINARY_DIR}/scorep_launcher.sh)
  add_dependencies(project_options scorep_launcher)
  set(CMAKE_C_COMPILER_LAUNCHER ${SCOREP_CMD})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${SCOREP_CMD})
  set(CMAKE_C_LINKER_LAUNCHER ${SCOREP_CMD})
  set(CMAKE_CXX_LINKER_LAUNCHER ${SCOREP_CMD})
endif()

include(FetchContent)

find_package(Threads REQUIRED)

file(GLOB_RECURSE SRC_ALG "source/algorithm/*")
file(GLOB_RECURSE SRC_ALG_NAIVE "source/algorithm/NaiveInternal/*")
file(GLOB_RECURSE SRC_ALG_BH "source/algorithm/BarnesHutInternal/*")
file(GLOB_RECURSE SRC_ALG_FMM "source/algorithm/FMMInternal/*")
file(GLOB_RECURSE SRC_ALG_KERNEL "source/algorithm/Kernel/*")
file(GLOB_RECURSE SRC_IO "source/io/*")
file(GLOB_RECURSE SRC_MPI "source/mpi/*")
file(GLOB_RECURSE SRC_SIM "source/sim/*")
file(GLOB_RECURSE SRC_SIM_FILE "source/sim/file/*")
file(GLOB_RECURSE SRC_SIM_RANDOM "source/sim/random/*")
file(GLOB_RECURSE SRC_STR "source/structure/*")
file(GLOB_RECURSE SRC_UTIL "source/util/*")
file(
  GLOB_RECURSE
  SRC_NEU
  "source/neurons/*.h"
  "source/neurons/*.cpp")
file(GLOB_RECURSE SRC_NEU_MOD "source/neurons/models/*")
file(GLOB_RECURSE SRC_NEU_HEL "source/neurons/helper/*")

file(GLOB_RECURSE SRC_MAIN "source/main.cpp")
file(GLOB_RECURSE SRC_RELEARN "source/main_relearn.cpp")
file(GLOB_RECURSE SRC_ANALYSIS "source/main_analysis.cpp")
file(GLOB_RECURSE SRC_CONF "source/Config.h")
file(GLOB_RECURSE SRC_TYPES "source/Types.h")

file(GLOB_RECURSE SRC_TST "test/*")
file(GLOB_RECURSE SRC_BENCH "benchmark/*")

source_group(
  TREE ${CMAKE_CURRENT_SOURCE_DIR}
  FILES ${SRC_ALG}
        ${SRC_ALG_NAIVE}
        ${SRC_ALG_BH}
        ${SRC_ALG_FMM}
        ${SRC_ALG_KERNEL}
        ${SRC_IO}
        ${SRC_MPI}
        ${SRC_SIM}
        ${SRC_STR}
        ${SRC_UTIL}
        ${SRC_NEU}
        ${SRC_NEU_MOD}
        ${SRC_NEU_HEL}
        ${SRC_MAIN}
        ${SRC_CONF}
        ${SRC_TYPES}
        ${SRC_SIM_FILE}
        ${SRC_SIM_RANDOM})

source_group("tests\\" FILES ${SRC_TST})
source_group("benchmark\\" FILES ${SRC_BENCH})

target_include_directories(project_options SYSTEM
                           INTERFACE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(project_options INTERFACE Threads::Threads)

target_include_directories(
  project_options
  PUBLIC
  INTERFACE "source/")

target_compile_options(project_options INTERFACE ${MPI_CXX_COMPILE_FLAGS})

target_compile_definitions(project_options
                           INTERFACE -DMPI_FOUND=$<BOOL:${MPI_FOUND}>)

target_link_libraries(project_options INTERFACE ${MPI_CXX_LIBRARIES}
                                                ${MPI_CXX_LINK_FLAGS})
if(UNIX)
  target_link_libraries(project_options INTERFACE stdc++fs)
endif()
target_link_libraries(project_options INTERFACE Boost::random)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.x)
FetchContent_MakeAvailable(spdlog)
target_link_libraries(project_options INTERFACE spdlog)

if(OpenMP_CXX_FOUND)
  target_link_libraries(project_options INTERFACE OpenMP::OpenMP_CXX)
endif()

get_target_property(spdlog_includes spdlog INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(spdlog PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                        "${spdlog_includes}")

add_library(
  relearn_lib STATIC
  ${SRC_ALG}
  ${SRC_ALG_NAIVE}
  ${SRC_ALG_BH}
  ${SRC_ALG_FMM}
  ${SRC_ALG_KERNEL}
  ${SRC_IO}
  ${SRC_MPI}
  ${SRC_SIM}
  ${SRC_STR}
  ${SRC_UTIL}
  ${SRC_NEU}
  ${SRC_NEU_MOD}
  ${SRC_NEU_HEL}
  ${SRC_CONF}
  ${SRC_TYPES}
  ${SRC_SIM_FILE}
  ${SRC_SIM_RANDOM})

target_include_directories(relearn_lib PUBLIC ${CMAKE_SOURCE_DIR}/source)

set_target_properties(relearn_lib PROPERTIES ENABLE_EXPORTS 1)

target_link_libraries(relearn_lib PUBLIC project_options)

# relearn executable
add_executable(relearn)
target_sources(relearn PRIVATE source/relearn.cpp)
target_link_libraries(relearn PUBLIC relearn_lib)
set_target_properties(relearn PROPERTIES ENABLE_EXPORTS 1)

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG 815553211b644b2a64b5dc671c602233c49c065f)
FetchContent_MakeAvailable(CLI11)
target_include_directories(relearn SYSTEM PRIVATE ${CLI11_SOURCE_DIR}/include)

if(NOT ENABLE_SCOREP)
  # # Download and unpack googletest at configure time
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0) # For Windows: Prevent overriding the parent
                            # project's compiler/linker settings
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.6.1)

  FetchContent_MakeAvailable(googlebenchmark)

add_executable(relearn_tests ${SRC_TST})

add_executable(relearn_benchmarks ${SRC_BENCH})

target_link_libraries(relearn_tests PRIVATE project_options)
target_link_libraries(relearn_tests PRIVATE relearn_lib)
target_link_libraries(relearn_tests PRIVATE gtest_main)

add_test(NAME SerialTests COMMAND relearn_tests)

target_link_libraries(relearn_benchmarks PRIVATE project_options)
target_link_libraries(relearn_benchmarks PRIVATE relearn_lib)
target_link_libraries(relearn_benchmarks PRIVATE benchmark::benchmark)
