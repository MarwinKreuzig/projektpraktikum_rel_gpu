file(GLOB_RECURSE SRC_ALG "${CMAKE_SOURCE_DIR}/source/algorithm/*")
file(GLOB_RECURSE SRC_ALG_NAIVE
     "${CMAKE_SOURCE_DIR}/source/algorithm/NaiveInternal/*")
file(GLOB_RECURSE SRC_ALG_BH
     "${CMAKE_SOURCE_DIR}/source/algorithm/BarnesHutInternal/*")
file(GLOB_RECURSE SRC_ALG_FMM
     "${CMAKE_SOURCE_DIR}/source/algorithm/FMMInternal/*")
file(GLOB_RECURSE SRC_ALG_KERNEL
     "${CMAKE_SOURCE_DIR}/source/algorithm/Kernel/*")
file(GLOB_RECURSE SRC_IO "${CMAKE_SOURCE_DIR}/source/io/*")
file(GLOB_RECURSE SRC_MPI "${CMAKE_SOURCE_DIR}/source/mpi/*")
file(GLOB_RECURSE SRC_SIM "${CMAKE_SOURCE_DIR}/source/sim/*")
file(GLOB_RECURSE SRC_SIM_FILE "${CMAKE_SOURCE_DIR}/source/sim/file/*")
file(GLOB_RECURSE SRC_SIM_RANDOM "${CMAKE_SOURCE_DIR}/source/sim/random/*")
file(GLOB_RECURSE SRC_STR "${CMAKE_SOURCE_DIR}/source/structure/*")
file(GLOB_RECURSE SRC_UTIL "${CMAKE_SOURCE_DIR}/source/util/*")
file(
  GLOB_RECURSE
  SRC_NEU
  "${CMAKE_SOURCE_DIR}/source/neurons/*.h"
  "${CMAKE_SOURCE_DIR}/source/neurons/*.cpp")
file(GLOB_RECURSE SRC_NEU_MOD "${CMAKE_SOURCE_DIR}/source/neurons/models/*")
file(GLOB_RECURSE SRC_NEU_HEL "${CMAKE_SOURCE_DIR}/source/neurons/helper/*")

file(GLOB_RECURSE SRC_MAIN "${CMAKE_SOURCE_DIR}/source/main.cpp")
file(GLOB_RECURSE SRC_RELEARN "${CMAKE_SOURCE_DIR}/source/main_relearn.cpp")
file(GLOB_RECURSE SRC_ANALYSIS "${CMAKE_SOURCE_DIR}/source/main_analysis.cpp")
file(GLOB_RECURSE SRC_CONF "${CMAKE_SOURCE_DIR}/source/Config.h")
file(GLOB_RECURSE SRC_TYPES "${CMAKE_SOURCE_DIR}/source/Types.h")

file(GLOB_RECURSE SRC_TST "${CMAKE_SOURCE_DIR}/test/*")
file(GLOB_RECURSE SRC_BENCH " ${CMAKE_SOURCE_DIR}/benchmark/*")

message("The path is:")
message(${CMAKE_SOURCE_DIR})
message(${SRC_UTIL})

source_group(
  TREE ${CMAKE_SOURCE_DIR}
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

add_library(
  relearn_lib_2 STATIC
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

