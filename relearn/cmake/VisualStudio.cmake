set(relearn_lib_additional_files "" CACHE INTERNAL "")
set(relearn_tests_additional_files "" CACHE INTERNAL "")
	
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
	list(APPEND relearn_lib_additional_files "neurons/LocalAreaTranslator.h")
	list(APPEND relearn_lib_additional_files "neurons/NetworkGraph.h")
	list(APPEND relearn_lib_additional_files "neurons/Neurons.h")
	list(APPEND relearn_lib_additional_files "neurons/NeuronsExtraInfo.h")
	list(APPEND relearn_lib_additional_files "neurons/SignalType.h")
	list(APPEND relearn_lib_additional_files "neurons/TargetCalciumDecay.h")
	list(APPEND relearn_lib_additional_files "neurons/UpdateStatus.h")
	
	# helper
	list(APPEND relearn_lib_additional_files "neurons/helper/AreaMonitor.h")
	list(APPEND relearn_lib_additional_files "neurons/helper/DistantNeuronRequests.h")
	list(APPEND relearn_lib_additional_files "neurons/helper/NeuronMonitor.h")
	list(APPEND relearn_lib_additional_files "neurons/helper/RankNeuronId.h")
	list(APPEND relearn_lib_additional_files "neurons/helper/Synapse.h")
	list(APPEND relearn_lib_additional_files "neurons/helper/SynapseCreationRequests.h")
	list(APPEND relearn_lib_additional_files "neurons/helper/SynapseDeletionRequests.h")
	
	# models
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
	list(APPEND relearn_lib_additional_files "util/Interval.h")
	list(APPEND relearn_lib_additional_files "util/MemoryHolder.h")
	list(APPEND relearn_lib_additional_files "util/MonitorParser.h")
	list(APPEND relearn_lib_additional_files "util/MPIRank.h")
	list(APPEND relearn_lib_additional_files "util/Random.h")
	list(APPEND relearn_lib_additional_files "util/RelearnException.h")
	list(APPEND relearn_lib_additional_files "util/SemiStableVector.h")
	list(APPEND relearn_lib_additional_files "util/Stack.h")
	list(APPEND relearn_lib_additional_files "util/StatisticalMeasures.h")
	list(APPEND relearn_lib_additional_files "util/StepParser.h")
	list(APPEND relearn_lib_additional_files "util/StimulusParser.h")
	list(APPEND relearn_lib_additional_files "util/StringUtil.h")
	list(APPEND relearn_lib_additional_files "util/TaggedID.h")
	list(APPEND relearn_lib_additional_files "util/Timers.h")
	list(APPEND relearn_lib_additional_files "util/Utility.h")
	list(APPEND relearn_lib_additional_files "util/Vec3.h")
	
	# shuffle
	list(APPEND relearn_lib_additional_files "util/shuffle/shuffle.h")
	
	#tests
	list(APPEND relearn_tests_additional_files "RandomAdapter.h")
	list(APPEND relearn_tests_additional_files "RelearnTest.hpp")
	
	list(APPEND relearn_tests_additional_files "background_activity/test_background_activity.h")
	
	list(APPEND relearn_tests_additional_files "barnes_hut/test_barnes_hut.h")
	
	list(APPEND relearn_tests_additional_files "calcium_calculator/test_calcium_calculator.h")
	
	list(APPEND relearn_tests_additional_files "cell/test_cell.h")
	
	list(APPEND relearn_tests_additional_files "connector/test_connector.h")
	
	list(APPEND relearn_tests_additional_files "fast_multipole_method/test_fast_multipole_method.h")
	
	list(APPEND relearn_tests_additional_files "helper/rank_neuron_id_adapter.h")
	list(APPEND relearn_tests_additional_files "helper/test_distant_neuron_request.h")
	list(APPEND relearn_tests_additional_files "helper/test_rank_neuron_id.h")
	list(APPEND relearn_tests_additional_files "helper/test_synapse_creation_request.h")
	list(APPEND relearn_tests_additional_files "helper/test_synapse_deletion_request.h")
	
	list(APPEND relearn_tests_additional_files "interval/interval_adapter.h")
	list(APPEND relearn_tests_additional_files "interval/test_interval.h")
	
	list(APPEND relearn_tests_additional_files "kernel/test_kernel.h")
	
	list(APPEND relearn_tests_additional_files "local_area_translator/test_local_area_translator.h")
	
	list(APPEND relearn_tests_additional_files "misc/test_misc.h")
	
	list(APPEND relearn_tests_additional_files "mpi/mpi_rank_adapter.h")
	list(APPEND relearn_tests_additional_files "mpi/test_mpi_rank.h")
	
	list(APPEND relearn_tests_additional_files "network_graph/test_network_graph.h")
	
	list(APPEND relearn_tests_additional_files "neuron_assignment/neuron_assignment_adapter.h")
	list(APPEND relearn_tests_additional_files "neuron_assignment/test_neuron_assignment.h")
	
	list(APPEND relearn_tests_additional_files "neuron_extra_info/test_neuron_extra_info.h")
	
	list(APPEND relearn_tests_additional_files "neuron_io/test_neuron_io.h")
	
	list(APPEND relearn_tests_additional_files "neuron_models/test_neuron_models.h")
	
	list(APPEND relearn_tests_additional_files "neurons/test_neurons.h")
	
	list(APPEND relearn_tests_additional_files "octree/test_octree.h")
	
	list(APPEND relearn_tests_additional_files "parser/test_monitor_parser.h")	
	list(APPEND relearn_tests_additional_files "parser/test_step_parser.h")
	
	list(APPEND relearn_tests_additional_files "partition/test_partition.h")
	
	list(APPEND relearn_tests_additional_files "space_filling_curve/test_space_filling_curve.h")
	
	list(APPEND relearn_tests_additional_files "synaptic_elements/test_synaptic_elements.h")
	
	list(APPEND relearn_tests_additional_files "synaptic_input/test_synaptic_input.h")
	
	list(APPEND relearn_tests_additional_files "tagged_id/tagged_id_adapter.h")
	list(APPEND relearn_tests_additional_files "tagged_id/test_tagged_id.h")
	
	list(APPEND relearn_tests_additional_files "vector/test_vector.h")
endif()
