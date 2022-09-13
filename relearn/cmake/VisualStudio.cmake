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
