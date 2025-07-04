add_library(relearn_lib STATIC)

#target_include_directories(relearn_lib PUBLIC shared)

target_sources(
        relearn_lib
        PRIVATE # algorithm
        algorithm/BarnesHutInternal/BarnesHut.cpp
        algorithm/BarnesHutInternal/BarnesHutInverted.cpp
        algorithm/BarnesHutInternal/BarnesHutLocationAware.cpp
        algorithm/FMMInternal/FastMultipoleMethod.cpp
        algorithm/FMMInternal/FastMultipoleMethodInverted.cpp
        algorithm/NaiveInternal/Naive.cpp

        # gpu
        gpu/neurons/models/NeuronModelGPU.cpp
        gpu/neurons/models/PoissonModelGPU.cpp
        gpu/neurons/models/FitzHughNagumoModelGPU.cpp
        gpu/neurons/models/AEIFModelGPU.cpp
        gpu/neurons/models/IzhikevichModelGPU.cpp
        gpu/utils/Interface.cpp
        gpu/algorithm/BarnesHutGPU.cpp
        gpu/utils/CudaHelper.cpp

        # io
        io/CalciumIO.cpp
        io/Event.cpp
        io/InteractiveNeuronIO.cpp
        io/LogFiles.cpp
        io/NeuronIO.cpp

        # mpi
        mpi/MPINoWrapper.cpp
        mpi/MPIWrapper.cpp

        # neurons
        neurons/helper/AreaMonitor.cpp
        neurons/helper/NeuronMonitor.cpp
        neurons/helper/SynapseDeletionFinder.cpp
        neurons/input/FiredStatusApproximator.cpp
        neurons/input/FiredStatusCommunicationMap.cpp
        neurons/input/SynapticInputCalculator.cpp
        neurons/input/SynapticInputCalculators.cpp
        neurons/models/AEIFModel.cpp
        neurons/models/FitzHughNagumoModel.cpp
        neurons/models/IzhikevichModel.cpp
        neurons/models/NeuronModel.cpp
        neurons/models/PoissonModel.cpp
        neurons/models/SynapticElements.cpp
        neurons/CalciumCalculator.cpp
        neurons/Neurons.cpp
        neurons/NeuronsExtraInfo.cpp

        # sim
        sim/file/MultipleFilesSynapseLoader.cpp
        sim/file/MultipleSubdomainsFromFile.cpp
        sim/random/BoxBasedRandomSubdomainAssignment.cpp
        sim/random/RandomSynapseLoader.cpp
        sim/random/SubdomainFromNeuronDensity.cpp
        sim/random/SubdomainFromNeuronPerRank.cpp
        sim/NeuronToSubdomainAssignment.cpp
        sim/Simulation.cpp
        sim/SynapseLoader.cpp

        # structure
        structure/Partition.cpp

        # util
        util/StringUtil.cpp
        util/RelearnException.cpp
        util/Timers.cpp
        util/Utility.cpp
        # for visual studio
        ${relearn_lib_additional_files})

target_include_directories(relearn_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} shared)
set_target_properties(relearn_lib PROPERTIES ENABLE_EXPORTS 1)
target_link_libraries(relearn_lib PUBLIC project_options project_libraries)
set_target_properties(relearn_lib PROPERTIES CXX_STANDARD 20)
target_compile_definitions(relearn_lib PRIVATE RELEARN_LIB)

if (CUDA_FOUND)
    target_link_libraries(relearn_lib PUBLIC project_options relearn_gpu)
endif ()

# benchmark# dont clutter the compile_commands file with libraries
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

FetchContent_Declare(
        cli11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG 815553211b644b2a64b5dc671c602233c49c065f)
FetchContent_MakeAvailable(cli11)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

get_target_property(cli11_includes CLI11 INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(CLI11 PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
        "${cli11_includes}")

# relearn executable
add_executable(relearn)
target_sources(relearn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/relearn.cpp)
target_link_libraries(relearn PUBLIC relearn_lib)
target_link_libraries(relearn PRIVATE CLI11::CLI11)
set_target_properties(relearn PROPERTIES ENABLE_EXPORTS 1)
set_target_properties(relearn PROPERTIES CXX_STANDARD 20)

# relearn_analysis executable
add_executable(relearn_analysis ${SRC_ANALYSIS})
target_sources(relearn_analysis
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/relearn_analysis.cpp)
target_link_libraries(relearn_analysis PRIVATE relearn_lib)
target_link_libraries(relearn_analysis PRIVATE CLI11::CLI11)
target_link_libraries(relearn_lib PUBLIC project_options project_libraries)
set_target_properties(relearn_analysis PROPERTIES CXX_STANDARD 20)
set_target_properties(relearn_analysis PROPERTIES ENABLE_EXPORTS 1)

get_target_property(relearn_lib_sources relearn_lib SOURCES)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${relearn_lib_sources})
