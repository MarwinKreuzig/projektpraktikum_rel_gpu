cmake_minimum_required(VERSION 3.10)
project(relearn)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

option(ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
if(ENABLE_CLANG_TIDY)
	set(CMAKE_CXX_CLANG_TIDY clang-tidy)
endif()

include(FetchContent)

find_package(MPI)
find_package(OpenMP)

file(GLOB_RECURSE SRC_ALG "source/algorithm/*")
file(GLOB_RECURSE SRC_IO "source/io/*")
file(GLOB_RECURSE SRC_MPI "source/mpi/*")
file(GLOB_RECURSE SRC_SIM "source/sim/*")
file(GLOB_RECURSE SRC_STR "source/structure/*")
file(GLOB_RECURSE SRC_UTIL "source/util/*")
file(GLOB_RECURSE SRC_NEU "source/neurons/*.h" "source/neurons/*.cpp")
file(GLOB_RECURSE SRC_NEU_MOD "source/neurons/models/*")
file(GLOB_RECURSE SRC_NEU_HEL "source/neurons/helper/*")

file(GLOB_RECURSE SRC_MAIN "source/main.cpp")
file(GLOB_RECURSE SRC_CONF "source/Config.h")

file(GLOB_RECURSE SRC_TST "test/*")

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC_ALG} ${SRC_IO} ${SRC_MPI} ${SRC_SIM} ${SRC_STR} ${SRC_UTIL} ${SRC_NEU} ${SRC_NEU_MOD} ${SRC_NEU_HEL} ${SRC_MAIN} ${SRC_CONF})

source_group("tests\\" FILES ${SRC_TST})

# project options
add_library(project_options INTERFACE)

target_include_directories (project_options SYSTEM INTERFACE ${MPI_CXX_INCLUDE_PATH})

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

# relearn executable
add_executable(relearn
	${SRC_ALG}
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
)

set_target_properties(relearn PROPERTIES ENABLE_EXPORTS 1)

target_link_libraries(relearn PRIVATE project_options)

if(OpenMP_CXX_FOUND)
    target_link_libraries(relearn PUBLIC OpenMP::OpenMP_CXX)
endif()

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(CLI11)
target_include_directories(relearn SYSTEM PRIVATE ${CLI11_SOURCE_DIR}/include)

#Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
		RESULT_VARIABLE result
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
	message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
		RESULT_VARIABLE result
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
	message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
		${CMAKE_BINARY_DIR}/googletest-build
		EXCLUDE_FROM_ALL)

add_executable(runTests
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
	${SRC_TST}
)

target_link_libraries(runTests PRIVATE project_options)
target_link_libraries(runTests PRIVATE gtest_main)

add_test(NAME SerialTests COMMAND runTests)
