cmake_minimum_required(VERSION 3.10)
project(relearn)

file(GLOB_RECURSE TOP_SRC "source/*.cpp" "source/*.c")
file(GLOB_RECURSE TOP_INCS "source/*.hpp" "source/*.h")

set(TOP_SRC_OM ${TOP_SRC})
list(REMOVE_ITEM TOP_SRC_OM "${PROJECT_SOURCE_DIR}/source/main.cpp")

file(GLOB_RECURSE TOP_TST "test/*")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

if(noclangtidy)
else()
	set(CMAKE_CXX_CLANG_TIDY clang-tidy "-header-filter=^((?!mpi).)*$" "--checks=*,-fuchsia*,-modernize-use-trailing-return-type,-llvm-header-guard,-misc-non-private-member-variables-in-classes,-cppcoreguidelines-non-private-member-variables-in-classes")
endif()

find_package (MPI REQUIRED)

source_group("tests" FILES ${TOP_TST})

add_executable(relearn
	${TOP_INCS}
	${TOP_SRC}
)

set_property(TARGET relearn PROPERTY ENABLE_EXPORTS 1)


target_include_directories (relearn PRIVATE ${MPI_CXX_INCLUDE_PATH} )
target_compile_options (relearn PRIVATE ${MPI_CXX_COMPILE_FLAGS})

target_link_libraries (relearn ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})

if (UNIX)
	target_link_libraries (relearn stdc++fs)
endif ()

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
	${TOP_INCS}
	${TOP_SRC_OM}
	${TOP_TST}
)
#target_include_directories(runTests PRIVATE)
target_link_libraries(runTests gtest_main)
add_test(NAME SerialTests COMMAND runTests)

target_include_directories (runTests PRIVATE ${MPI_CXX_INCLUDE_PATH} )

target_compile_options (runTests PRIVATE ${MPI_CXX_COMPILE_FLAGS})

target_link_libraries (runTests ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})

if (UNIX)
	target_link_libraries (runTests stdc++fs)
endif ()
