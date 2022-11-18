set(relearn_lib_additional_files "" CACHE INTERNAL "")
	
if(WIN32)
	# root files
	list(APPEND relearn_lib_additional_files "Config.h")
	list(APPEND relearn_lib_additional_files "Types.h")
	
    # algorithm
	list(APPEND relearn_lib_additional_files "algorithm/Algorithm.h")
	list(APPEND relearn_lib_additional_files "algorithm/AlgorithmEnum.h")
	list(APPEND relearn_lib_additional_files "algorithm/Algorithms.h")
	list(APPEND relearn_lib_additional_files "algorithm/Cells.h")
	list(APPEND relearn_lib_additional_files "algorithm/Connector.h")
	list(APPEND relearn_lib_additional_files "algorithm/VirtualPlasticityElement.h")
	
	# BarnesHutInternal
	list(APPEND relearn_lib_additional_files "algorithm/BarnesHutInternal/BarnesHut.h")
	list(APPEND relearn_lib_additional_files "algorithm/BarnesHutInternal/BarnesHutBase.h")
	list(APPEND relearn_lib_additional_files "algorithm/BarnesHutInternal/BarnesHutCell.h")
	list(APPEND relearn_lib_additional_files "algorithm/BarnesHutInternal/BarnesHutInverted.h")
	list(APPEND relearn_lib_additional_files "algorithm/BarnesHutInternal/BarnesHutInvertedCell.h")
	list(APPEND relearn_lib_additional_files "algorithm/BarnesHutInternal/BarnesHutLocationAware.h")
	
	# FMMInternal
	list(APPEND relearn_lib_additional_files "algorithm/FMMInternal/FastMultipoleMethods.h")
	list(APPEND relearn_lib_additional_files "algorithm/FMMInternal/FastMultipoleMethodsBase.h")
	list(APPEND relearn_lib_additional_files "algorithm/FMMInternal/FastMultipoleMethodsCell.h")
	list(APPEND relearn_lib_additional_files "algorithm/FMMInternal/FastMultipoleMethodsInverted.h")
	
	# Internal
	list(APPEND relearn_lib_additional_files "algorithm/Internal/AlgorithmImpl.h")
	list(APPEND relearn_lib_additional_files "algorithm/Internal/ExchangingAlgorithm.h")
	
	# Kernel
	list(APPEND relearn_lib_additional_files "algorithm/Kernel/Gamma.h")
	list(APPEND relearn_lib_additional_files "algorithm/Kernel/Gaussian.h")
	list(APPEND relearn_lib_additional_files "algorithm/Kernel/Kernel.h")
	list(APPEND relearn_lib_additional_files "algorithm/Kernel/Linear.h")
	list(APPEND relearn_lib_additional_files "algorithm/Kernel/Weibull.h")
	
	# Naive
	list(APPEND relearn_lib_additional_files "algorithm/NaiveInternal/Naive.h")
	list(APPEND relearn_lib_additional_files "algorithm/NaiveInternal/NaiveCell.h")
	
	# io
	list(APPEND relearn_lib_additional_files "io/CalciumIO.h")
	list(APPEND relearn_lib_additional_files "io/FileValidator.h")
	list(APPEND relearn_lib_additional_files "io/InteractiveNeuronIO.h")
	list(APPEND relearn_lib_additional_files "io/LogFiles.h")
	list(APPEND relearn_lib_additional_files "io/NeuronIO.h")
	
	# mpi
	list(APPEND relearn_lib_additional_files "mpi/CommunicationMap.h")
	list(APPEND relearn_lib_additional_files "mpi/MPINoWrapper.h")
	list(APPEND relearn_lib_additional_files "mpi/MPIWrapper.h")
	
	# neurons
	list(APPEND relearn_lib_additional_files "neurons/CalciumCalculator.h")
	list(APPEND relearn_lib_additional_files "neurons/ElementType.h")
	list(APPEND relearn_lib_additional_files "neurons/FiredStatus.h")
	list(APPEND relearn_lib_additional_files "neurons/NetworkGraph.h")
	list(APPEND relearn_lib_additional_files "neurons/Neurons.h")
	list(APPEND relearn_lib_additional_files "neurons/NeuronsExtraInfo.h")
	list(APPEND relearn_lib_additional_files "neurons/SignalType.h")
	list(APPEND relearn_lib_additional_files "neurons/TargetCalciumDecay.h")
	list(APPEND relearn_lib_additional_files "neurons/UpdateStatus.h")
	
	# neurons
	list(APPEND relearn_lib_additional_files "neurons/helper/DistantNeuronRequests.h")
	list(APPEND relearn_lib_additional_files "neurons/helper/NeuronMonitor.h")
	list(APPEND relearn_lib_additional_files "neurons/helper/RankNeuronId.h")
	list(APPEND relearn_lib_additional_files "neurons/helper/Synapse.h")
	list(APPEND relearn_lib_additional_files "neurons/helper/SynapseCreationRequests.h")
	list(APPEND relearn_lib_additional_files "neurons/helper/SynapseDeletionRequests.h")
	
	# neurons
	list(APPEND relearn_lib_additional_files "neurons/models/BackgroundActivityCalculator.h")
	list(APPEND relearn_lib_additional_files "neurons/models/BackgroundActivityCalculators.h")
	list(APPEND relearn_lib_additional_files "neurons/models/FiredStatusCommunicationMap.h")
	list(APPEND relearn_lib_additional_files "neurons/models/FiredStatusCommunicator.h")
	list(APPEND relearn_lib_additional_files "neurons/models/ModelParameter.h")
	list(APPEND relearn_lib_additional_files "neurons/models/NeuronModels.h")
	list(APPEND relearn_lib_additional_files "neurons/models/SynapticElements.h")
	list(APPEND relearn_lib_additional_files "neurons/models/SynapticInputCalculator.h")
	list(APPEND relearn_lib_additional_files "neurons/models/SynapticInputCalculators.h")
	
	# sim
	list(APPEND relearn_lib_additional_files "sim/LoadedNeuron.h")
	list(APPEND relearn_lib_additional_files "sim/NeuronToSubdomainAssignment.h")
	list(APPEND relearn_lib_additional_files "sim/Simulation.h")
	list(APPEND relearn_lib_additional_files "sim/SynapseLoader.h")
	
	# file
	list(APPEND relearn_lib_additional_files "sim/file/FileSynapseLoader.h")
	list(APPEND relearn_lib_additional_files "sim/file/MultipleFilesSynapseLoader.h")
	list(APPEND relearn_lib_additional_files "sim/file/MultipleSubdomainsFromFile.h")
	list(APPEND relearn_lib_additional_files "sim/file/SubdomainFromFile.h")
	
	# random
	list(APPEND relearn_lib_additional_files "sim/random/BoxBasedRandomSubdomainAssignment.h")
	list(APPEND relearn_lib_additional_files "sim/random/RandomSynapseLoader.h")
	list(APPEND relearn_lib_additional_files "sim/random/SubdomainFromNeuronDensity.h")
	list(APPEND relearn_lib_additional_files "sim/random/SubdomainFromNeuronPerRank.h")
	
	# structure
	list(APPEND relearn_lib_additional_files "structure/BaseCell.h")
	list(APPEND relearn_lib_additional_files "structure/Cell.h")
	list(APPEND relearn_lib_additional_files "structure/NodeCache.h")
	list(APPEND relearn_lib_additional_files "structure/Octree.h")
	list(APPEND relearn_lib_additional_files "structure/OctreeNode.h")
	list(APPEND relearn_lib_additional_files "structure/Partition.h")
	list(APPEND relearn_lib_additional_files "structure/SpaceFillingCurve.h")
	
	# util
	list(APPEND relearn_lib_additional_files "util/MemoryHolder.h")
	list(APPEND relearn_lib_additional_files "util/MonitorParser.h")
	list(APPEND relearn_lib_additional_files "util/MPIRank.h")
	list(APPEND relearn_lib_additional_files "util/Random.h")
	list(APPEND relearn_lib_additional_files "util/RelearnException.h")
	list(APPEND relearn_lib_additional_files "util/SemiStableVector.h")
	list(APPEND relearn_lib_additional_files "util/Stack.h")
	list(APPEND relearn_lib_additional_files "util/StatisticalMeasures.h")
	list(APPEND relearn_lib_additional_files "util/StepParser.h")
	list(APPEND relearn_lib_additional_files "util/TaggedID.h")
	list(APPEND relearn_lib_additional_files "util/Timers.h")
	list(APPEND relearn_lib_additional_files "util/Utility.h")
	list(APPEND relearn_lib_additional_files "util/Vec3.h")
	
	# shuffle
	list(APPEND relearn_lib_additional_files "util/shuffle/shuffle.h")
endif()