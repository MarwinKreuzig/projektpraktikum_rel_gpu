cmake_minimum_required(VERSION 3.10)
project(relearn)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(
    STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE
      RelWithDebInfo
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui, ccmake
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

option(ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
if(ENABLE_CLANG_TIDY)
	set(CMAKE_CXX_CLANG_TIDY clang-tidy)
endif()

include(FetchContent)

option(ENABLE_MPI "Enable mpi" ON)
if(ENABLE_MPI)
	find_package(MPI)
endif()

find_package(OpenMP)

file(GLOB_RECURSE SRC_ALG "source/algorithm/*")
file(GLOB_RECURSE SRC_IO "source/io/*")
file(GLOB_RECURSE SRC_MPI "source/mpi/*")
file(GLOB_RECURSE SRC_SIM "source/sim/*")
file(GLOB_RECURSE SRC_SIM_FILE "source/sim/file/*")
file(GLOB_RECURSE SRC_SIM_RANDOM "source/sim/random/*")
file(GLOB_RECURSE SRC_STR "source/structure/*")
file(GLOB_RECURSE SRC_UTIL "source/util/*")
file(GLOB_RECURSE SRC_NEU "source/neurons/*.h" "source/neurons/*.cpp")
file(GLOB_RECURSE SRC_NEU_MOD "source/neurons/models/*")
file(GLOB_RECURSE SRC_NEU_HEL "source/neurons/helper/*")

file(GLOB_RECURSE SRC_MAIN "source/main.cpp")
file(GLOB_RECURSE SRC_CONF "source/Config.h")
file(GLOB_RECURSE SRC_TYPES "source/Types.h")

file(GLOB_RECURSE SRC_TST "test/*")
file(GLOB_RECURSE SRC_BENCH "benchmark/*")

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES 
${SRC_ALG} ${SRC_IO} ${SRC_MPI} ${SRC_SIM} 
${SRC_STR} ${SRC_UTIL} ${SRC_NEU} ${SRC_NEU_MOD} 
${SRC_NEU_HEL} ${SRC_MAIN} ${SRC_CONF} ${SRC_TYPES} ${SRC_SIM_FILE} ${SRC_SIM_RANDOM})

source_group("tests\\" FILES ${SRC_TST})
source_group("benchmark\\" FILES ${SRC_BENCH})

# project options
add_library(project_options INTERFACE)

target_include_directories (project_options SYSTEM INTERFACE ${MPI_CXX_INCLUDE_PATH})
target_include_directories(project_options SYSTEM INTERFACE "source/")

target_compile_options (project_options INTERFACE ${MPI_CXX_COMPILE_FLAGS})

target_compile_definitions(project_options INTERFACE -DMPI_FOUND=$<BOOL:${MPI_FOUND}>)

target_link_libraries (project_options INTERFACE ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
if (UNIX)
	target_link_libraries (project_options INTERFACE stdc++fs)
endif ()

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.x
)
FetchContent_MakeAvailable(spdlog)
target_link_libraries(project_options INTERFACE spdlog)

if(OpenMP_CXX_FOUND)
    target_link_libraries(project_options INTERFACE OpenMP::OpenMP_CXX)
endif()

get_target_property(spdlog_includes spdlog INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(spdlog PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${spdlog_includes}")

add_library(relearn_lib STATIC
${SRC_ALG}
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
${SRC_SIM_RANDOM}
)

target_include_directories(relearn_lib PUBLIC "source/")

set_target_properties(relearn_lib PROPERTIES ENABLE_EXPORTS 1)

target_link_libraries(relearn_lib PUBLIC project_options)

# relearn executable
add_executable(relearn
	${SRC_MAIN}
)

target_link_libraries(relearn PRIVATE relearn_lib)

set_target_properties(relearn PROPERTIES ENABLE_EXPORTS 1)

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        815553211b644b2a64b5dc671c602233c49c065f
)
FetchContent_MakeAvailable(CLI11)
target_include_directories(relearn SYSTEM PRIVATE ${CLI11_SOURCE_DIR}/include)

#Download and unpack googletest at configure time
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.6.1)
        
FetchContent_MakeAvailable(googlebenchmark)

add_executable(runTests
	${SRC_TST}
)

add_executable(benchmarks
	${SRC_BENCH}
)

target_link_libraries(runTests PRIVATE project_options)
target_link_libraries(runTests PRIVATE relearn_lib)
target_link_libraries(runTests PRIVATE gtest_main)

add_test(NAME SerialTests COMMAND runTests)

target_link_libraries(benchmarks PRIVATE project_options)
target_link_libraries(benchmarks PRIVATE relearn_lib)
target_link_libraries(benchmarks PRIVATE benchmark::benchmark)
