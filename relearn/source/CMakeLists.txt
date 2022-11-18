add_library(relearn_lib STATIC)
target_sources(
  relearn_lib
  PRIVATE # algorithm
          algorithm/BarnesHutInternal/BarnesHut.cpp
          algorithm/BarnesHutInternal/BarnesHutInverted.cpp
          algorithm/BarnesHutInternal/BarnesHutLocationAware.cpp
          algorithm/FMMInternal/FastMultipoleMethods.cpp
          algorithm/FMMInternal/FastMultipoleMethodsInverted.cpp
          algorithm/NaiveInternal/Naive.cpp
          # io
          io/CalciumIO.cpp
          io/InteractiveNeuronIO.cpp
          io/LogFiles.cpp
          io/NeuronIO.cpp
          # mpi
          mpi/MPINoWrapper.cpp
          mpi/MPIWrapper.cpp
          # neurons
		  neurons/helper/AreaMonitor.cpp
          neurons/helper/NeuronMonitor.cpp
          neurons/models/AEIFModel.cpp
          neurons/models/FiredStatusCommunicationMap.cpp
          neurons/models/FitzHughNagumoModel.cpp
          neurons/models/IzhikevichModel.cpp
          neurons/models/NeuronModels.cpp
          neurons/models/PoissonModel.cpp
          neurons/models/SynapticElements.cpp
          neurons/models/SynapticInputCalculator.cpp
          neurons/models/SynapticInputCalculators.cpp
          neurons/CalciumCalculator.cpp
          neurons/Neurons.cpp
          neurons/NeuronsExtraInfo.cpp
          # sim
          sim/file/FileSynapseLoader.cpp
          sim/file/MultipleFilesSynapseLoader.cpp
          sim/file/MultipleSubdomainsFromFile.cpp
          sim/file/SubdomainFromFile.cpp
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
		  util/FileLoader.cpp
          util/RelearnException.cpp
          util/Timers.cpp
		  
		  ${relearn_lib_additional_files})

target_include_directories(relearn_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(relearn_lib PROPERTIES ENABLE_EXPORTS 1)
target_link_libraries(relearn_lib PUBLIC project_options project_libraries)

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG 815553211b644b2a64b5dc671c602233c49c065f)
FetchContent_MakeAvailable(cli11)

# relearn executable
add_executable(relearn)
target_sources(relearn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/relearn.cpp)
target_link_libraries(relearn PRIVATE relearn_lib)
target_link_libraries(relearn PRIVATE CLI11::CLI11)
set_target_properties(relearn PROPERTIES ENABLE_EXPORTS 1)

# relearn_analysis executable
add_executable(relearn_analysis ${SRC_ANALYSIS})
target_sources(relearn_analysis
               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/relearn_analysis.cpp)
target_link_libraries(relearn_analysis PRIVATE relearn_lib)
target_link_libraries(relearn_analysis PRIVATE CLI11::CLI11)
set_target_properties(relearn_analysis PROPERTIES ENABLE_EXPORTS 1)

get_target_property(relearn_lib_sources relearn_lib SOURCES)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${relearn_lib_sources})
