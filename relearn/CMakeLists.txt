cmake_minimum_required(VERSION 3.0)
project(relearn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

find_package (Threads)
find_package (MPI REQUIRED)

file(GLOB_RECURSE TOP_INCS "source/*.hpp" "source/*.h" "source/*.cpp" "source/*.c")

#source_group("group_name" FILES ${TOP_INCS})

add_executable(relearn
	${TOP_INCS}
)

target_include_directories (relearn PRIVATE ${MPI_CXX_INCLUDE_PATH} )

if (WIN32)
	#string (REPLACE "/D_WINDOWS" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}) 
	#string (REPLACE "/DWIN32" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}) 
endif ()

target_compile_options (relearn PRIVATE ${MPI_CXX_COMPILE_FLAGS})

target_link_libraries (relearn ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
target_link_libraries (relearn ${CMAKE_THREAD_LIBS_INIT})

if (UNIX)
	target_link_libraries (relearn stdc++fs)
endif ()