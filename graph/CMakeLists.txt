cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0091 NEW)
project(relearn-full_graph C CXX)

file(GLOB_RECURSE TOP_SRC "source/*.cpp" "source/*.c")
file(GLOB_RECURSE TOP_INCS "source/*.hpp" "source/*.h")
file(GLOB_RECURSE TOP_CUD  "source/*.cu")

list(REMOVE_ITEM TOP_SRC "${PROJECT_SOURCE_DIR}/source/main.cpp")

set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
	option(ENABLE_CUDA "Use CUDA" ON)
	set(CUDA_FOUND $<AND:$<BOOL:${ENABLE_CUDA}>,$<BOOL:${CMAKE_CUDA_COMPILER}>>)

	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	set(CMAKE_CUDA_ARCHITECTURES 52 61)
else()
	set(CUDA_FOUND 0)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_CXX_CLANG_TIDY clang-tidy "-header-filter=^((?!(mpi|gtest)).)*$" "-checks=*,-fuchsia*")

find_package(Threads)
find_package(OpenMP)

set(Boost_ADDITIONAL_VERSIONS 1.74.0)

find_package (Boost) #find_package ( Boost 1.40 COMPONENTS program_options REQUIRED )
include_directories ( ${Boost_INCLUDE_DIR} )

add_library(relearn-full_graph_static
	${TOP_INCS}
	${TOP_SRC}
	$<${CUDA_FOUND}:${TOP_CUD}>
)

if (UNIX)
	target_compile_options(relearn-full_graph_static PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wpedantic -Wextra>)
else()
	target_compile_options(relearn-full_graph_static PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall>)
endif ()

target_link_libraries (relearn-full_graph_static PRIVATE ${CMAKE_THREAD_LIBS_INIT})

if(OpenMP_CXX_FOUND)
    target_link_libraries(relearn-full_graph_static PUBLIC OpenMP::OpenMP_CXX)
endif()

if (UNIX)
	target_link_libraries (relearn-full_graph_static PRIVATE stdc++fs)
endif ()

target_compile_definitions(relearn-full_graph_static PUBLIC -DCUDA_FOUND=${CUDA_FOUND})
target_include_directories(relearn-full_graph_static SYSTEM PUBLIC $<${CUDA_FOUND}:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>)

include(FetchContent)
FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        5ce8958c7e3d2c871d1ba3180a4e4f1543eece4a
)
FetchContent_MakeAvailable(CLI11)
target_include_directories(relearn-full_graph_static SYSTEM PUBLIC ${CLI11_SOURCE_DIR}/include)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.x
)
FetchContent_MakeAvailable(spdlog)
target_link_libraries(relearn-full_graph_static PUBLIC spdlog)

get_target_property(spdlog_includes spdlog INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(spdlog PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${spdlog_includes}")

add_executable(relearn-full_graph "source/main.cpp")


target_link_libraries(relearn-full_graph PRIVATE relearn-full_graph_static)

enable_testing()
add_subdirectory(test)
