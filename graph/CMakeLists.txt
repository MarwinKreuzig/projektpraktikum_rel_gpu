cmake_minimum_required(VERSION 3.10)
project(relearn-full_graph C CXX CUDA)

file(GLOB_RECURSE TOP_SRC "source/*.cpp" "source/*.c")
file(GLOB_RECURSE TOP_INCS "source/*.hpp" "source/*.h")
file(GLOB_RECURSE TOP_CUD  "source/*.cu")

set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
	option(USE_CUDA "Use cuda" ON)
	set(USE_CUDA $<AND:$<BOOL:${USE_CUDA}>,$<BOOL:${CMAKE_CUDA_COMPILER}>>)

	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	set(CMAKE_CUDA_ARCHITECTURES 52 61)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_CXX_CLANG_TIDY clang-tidy "-header-filter=^((?!(mpi|gtest)).)*$" "-checks=*,-fuchsia*")

find_package (Threads)
find_package(OpenMP)

set(Boost_ADDITIONAL_VERSIONS 1.74.0)

find_package (Boost) #find_package ( Boost 1.40 COMPONENTS program_options REQUIRED )
include_directories ( ${Boost_INCLUDE_DIR} )

add_executable(relearn-full_graph
	${TOP_INCS}
	${TOP_SRC}
	$<${USE_CUDA}:${TOP_CUD}>
)

target_compile_options(relearn-full_graph PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wpedantic -Wextra>)

target_link_libraries (relearn-full_graph PRIVATE ${CMAKE_THREAD_LIBS_INIT})

if(OpenMP_CXX_FOUND)
    target_link_libraries(relearn-full_graph PUBLIC OpenMP::OpenMP_CXX)
endif()

if (UNIX)
	target_link_libraries (relearn-full_graph PRIVATE stdc++fs)
endif ()

target_compile_definitions(relearn-full_graph PRIVATE -DUSE_CUDA=${USE_CUDA})
target_include_directories(relearn-full_graph SYSTEM PRIVATE $<${USE_CUDA}:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>)
