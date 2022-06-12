/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"
#include "Types.h"
#include "algorithm/Algorithms.h"
#include "io/InteractiveNeuronIO.h"
#include "io/LogFiles.h"
#include "mpi/CommunicationMap.h"
#include "mpi/MPIWrapper.h"
#include "neurons/ElementType.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/models/NeuronModels.h"
#include "neurons/models/SynapticElements.h"
#include "sim/NeuronToSubdomainAssignment.h"
#include "sim/Simulation.h"
#include "sim/file/SubdomainFromFile.h"
#include "sim/random/SubdomainFromNeuronDensity.h"
#include "sim/random/SubdomainFromNeuronPerRank.h"
#include "structure/Octree.h"
#include "structure/Partition.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include "spdlog/spdlog.h"

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>

#ifdef _OPENMP
#include <omp.h>
#else
void omp_set_num_threads(int num) { }
#endif

#include <array>
#include <bitset>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>

void print_sizes() {
    struct empty_t {
        using position_type = VirtualPlasticityElement::position_type;
        using counter_type = VirtualPlasticityElement::counter_type;
    };

    const auto sizeof_vec3_double = sizeof(Vec3d);
    const auto sizeof_vec3_size_t = sizeof(Vec3s);

    const auto sizeof_virtual_plasticity_element = sizeof(VirtualPlasticityElement);

    const auto sizeof_empty_t = sizeof(empty_t);
    const auto sizeof_fmm_cell_attributes = sizeof(FastMultipoleMethodsCell);
    const auto sizeof_bh_cell_attributes = sizeof(BarnesHutCell);
    const auto sizeof_bh_naive_attributes = sizeof(NaiveCell);

    const auto sizeof_empty_cell = sizeof(Cell<empty_t>);
    const auto sizeof_fmm_cell = sizeof(Cell<FastMultipoleMethodsCell>);
    const auto sizeof_bh_cell = sizeof(Cell<BarnesHutCell>);
    const auto sizeof_naive_cell = sizeof(Cell<NaiveCell>);

    const auto sizeof_octreenode = sizeof(OctreeNode<empty_t>);
    const auto sizeof_fmm_octreenode = sizeof(OctreeNode<FastMultipoleMethodsCell>);
    const auto sizeof_bh_octreenode = sizeof(OctreeNode<BarnesHutCell>);
    const auto sizeof_naive_octreenode = sizeof(OctreeNode<NaiveCell>);

    const auto sizeof_neuron_id = sizeof(NeuronID);
    const auto sizeof_rank_neuron_id = sizeof(RankNeuronId);

    const auto sizeof_local_synapse = sizeof(LocalSynapse);
    const auto sizeof_distant_in_synapse = sizeof(DistantInSynapse);
    const auto sizeof_distant_out_synapse = sizeof(DistantOutSynapse);

    std::stringstream ss{};

    ss << '\n';

    ss << "Size of Vec3d: " << sizeof_vec3_double << '\n';
    ss << "Size of Vec3s: " << sizeof_vec3_size_t << '\n';

    ss << "Size of VirtualPlasticityElement: " << sizeof_virtual_plasticity_element << "\n";
    ss << "Size of FastMultipoleMethodsCell: " << sizeof_fmm_cell_attributes << '\n';

    ss << "Size of empty_t: " << sizeof_empty_t << '\n';
    ss << "Size of BarnesHutCell: " << sizeof_bh_cell_attributes << '\n';
    ss << "Size of NaiveCell: " << sizeof_bh_naive_attributes << "\n";

    ss << "Size of Cell<empty_t>: " << sizeof_empty_cell << '\n';
    ss << "Size of Cell<FastMultipoleMethodsCell>: " << sizeof_fmm_cell << '\n';
    ss << "Size of Cell<BarnesHutCell>: " << sizeof_bh_cell << '\n';
    ss << "Size of Cell<NaiveCell>: " << sizeof_naive_cell << "\n";

    ss << "Size of OctreeNode<empty_t>: " << sizeof_octreenode << '\n';
    ss << "Size of OctreeNode<FastMultipoleMethodsCell>: " << sizeof_fmm_octreenode << '\n';
    ss << "Size of OctreeNode<BarnesHutCell>: " << sizeof_bh_octreenode << '\n';
    ss << "Size of OctreeNode<NaiveCell>: " << sizeof_naive_octreenode << '\n';

    ss << "Size of NeuronID: " << sizeof_neuron_id << '\n';
    ss << "Size of RankNeuronID: " << sizeof_rank_neuron_id << '\n';

    ss << "Size of LocalSynapse: " << sizeof_local_synapse << '\n';
    ss << "Size of DistantInSynapse: " << sizeof_distant_in_synapse << '\n';
    ss << "Size of DistantOutSynapse: " << sizeof_distant_out_synapse << "\n";

    LogFiles::print_message_rank(0, ss.str());
}

void print_arguments(int argc, char** argv) {
    std::stringstream ss{};

    for (auto i = 0; i < argc; i++) {
        ss << argv[i] << ' ';
    }

    LogFiles::print_message_rank(0, ss.str());
}

int main(int argc, char** argv) {
    /**
     * Init MPI and store some MPI infos
     */
    MPIWrapper::init(argc, argv);

    print_arguments(argc, argv);
    print_sizes();

    if constexpr (Config::do_debug_checks) {
        std::cout << "I'm performing Debug Checks\n";
    } else {
        std::cout << "I'm skipping Debug Checks\n";
    }

    const int my_rank = MPIWrapper::get_my_rank();
    const int num_ranks = MPIWrapper::get_num_ranks();

    // Command line arguments
    CLI::App app{ "" };

    AlgorithmEnum algorithm = AlgorithmEnum::BarnesHut;
    std::map<std::string, AlgorithmEnum> cli_parse_algorithm{
        { "naive", AlgorithmEnum::Naive },
        { "barnes-hut", AlgorithmEnum::BarnesHut },
        { "barnes-hut-inverted", AlgorithmEnum::BarnesHutInverted },
        { "barnes-hut-location-aware", AlgorithmEnum::BarnesHutLocationAware },
        { "fast-multipole-methods", AlgorithmEnum::FastMultipoleMethods }
    };

    NeuronModelEnum neuron_model = NeuronModelEnum::Poisson;
    std::map<std::string, NeuronModelEnum> cli_parse_neuron_model{
        { "poisson", NeuronModelEnum::Poisson },
        { "izhikevich", NeuronModelEnum::Izhikevich },
        { "aeif", NeuronModelEnum::AEIF },
        { "fitzhughnagumo", NeuronModelEnum::FitzHughNagumo }
    };

    auto* opt_neuron_model = app.add_option("--neuron-model", neuron_model, "The neuron model");
    opt_neuron_model->transform(CLI::CheckedTransformer(cli_parse_neuron_model, CLI::ignore_case));

    auto* opt_algorithm = app.add_option("-a,--algorithm", algorithm, "The algorithm that is used for finding the targets");
    opt_algorithm->required()->transform(CLI::CheckedTransformer(cli_parse_algorithm, CLI::ignore_case));

    double accept_criterion{ BarnesHut::default_theta };
    auto* opt_accept_criterion = app.add_option("-t,--theta", accept_criterion, "Theta, the acceptance criterion for Barnes-Hut. Default: 0.3. Required Barnes-Hut.");

    double scaling_constant{ Constants::default_sigma };
    app.add_option("--sigma", scaling_constant, "Scaling parameter for the probabilty kernel. Default: 750");

    size_t number_neurons{};
    auto* opt_num_neurons = app.add_option("-n,--num-neurons", number_neurons, "Number of neurons. This option is only advised when using one MPI rank!");

    size_t number_neurons_per_rank{};
    auto* opt_num_neurons_per_rank = app.add_option("--num-neurons-per-rank", number_neurons_per_rank, "Number neurons per MPI rank.");

    std::string file_positions{};
    auto* opt_file_positions = app.add_option("-f,--file", file_positions, "File with neuron positions.");

    std::string file_network{};
    auto* opt_file_network = app.add_option("-g,--graph", file_network, "File with neuron connections.");

    std::string file_enable_interrupts{};
    auto* opt_file_enable_interrupts = app.add_option("--enable-interrupts", file_enable_interrupts, "File with the enable interrupts.");

    std::string file_disable_interrupts{};
    auto* opt_file_disable_interrupts = app.add_option("--disable-interrupts", file_disable_interrupts, "File with the disable interrupts.");

    std::string file_creation_interrupts{};
    auto* opt_file_creation_interrups = app.add_option("--creation-interrupts", file_creation_interrupts, "File with the creation interrupts.");

    double base_background_activity{ NeuronModel::default_base_background_activity };
    app.add_option("--base-background-activity", base_background_activity, "The base background activity by which all neurons are excited. The background activity is calculated as <base> + N(mean, stddev)");

    double background_activity_mean{ NeuronModel::default_background_activity_mean };
    app.add_option("--background-activity-mean", background_activity_mean, "The mean background activity by which all neurons are excited. The background activity is calculated as <base> + N(mean, stddev)");

    double background_activity_stddev{ NeuronModel::default_background_activity_stddev };
    app.add_option("--background-activity-stddev", background_activity_stddev, "The standard deviation of the background activity by which all neurons are excited. The background activity is calculated as <base> + N(mean, stddev)");

    double synapse_conductance{ NeuronModel::default_k };
    app.add_option("--synapse-conductance", synapse_conductance, "The activity that is transfered to its neighbors when a neuron spikes. Default is 0.03");

    double calcium_decay{ NeuronModel::default_tau_C };
    app.add_option("--calcium-decay", calcium_decay, "The decay constant for the intercellular calcium");

    double retract_ratio{ SynapticElements::default_vacant_retract_ratio };
    app.add_option("--retract-ratio", retract_ratio, "The ratio by which vacant synapses retract.");

    std::string log_prefix{};
    auto* opt_log_prefix = app.add_option("-p,--log-prefix", log_prefix, "Prefix for log files.");

    std::string log_path{};
    auto* opt_log_path = app.add_option("-l,--log-path", log_path, "Path for log files.");

    size_t simulation_steps{};
    app.add_option("-s,--steps", simulation_steps, "Simulation steps in ms.")->required();

    unsigned int random_seed{ 0 };
    app.add_option("-r,--random-seed", random_seed, "Random seed. Default: 0.");

    int openmp_threads{ 1 };
    app.add_option("--openmp", openmp_threads, "Number of OpenMP Threads.");

    auto* flag_interactive = app.add_flag("-i,--interactive", "Run interactively.");

    size_t first_plasticity_step{ Config::first_plasticity_update };
    app.add_option("--first-plasticity-step", first_plasticity_step, "The first step in which the plasticity is updated.");

    opt_num_neurons->excludes(opt_file_positions);
    opt_num_neurons->excludes(opt_file_network);
    opt_file_positions->excludes(opt_num_neurons);
    opt_file_network->excludes(opt_num_neurons);

    opt_num_neurons_per_rank->excludes(opt_num_neurons);
    opt_num_neurons->excludes(opt_num_neurons_per_rank);

    opt_num_neurons_per_rank->excludes(opt_file_positions);
    opt_num_neurons_per_rank->excludes(opt_file_network);
    opt_file_positions->excludes(opt_num_neurons_per_rank);
    opt_file_network->excludes(opt_num_neurons_per_rank);

    opt_file_network->needs(opt_file_positions);

    opt_file_positions->check(CLI::ExistingFile);
    opt_file_network->check(CLI::ExistingFile);

    opt_file_enable_interrupts->check(CLI::ExistingFile);
    opt_file_disable_interrupts->check(CLI::ExistingFile);
    opt_file_creation_interrups->check(CLI::ExistingFile);

    opt_log_path->check(CLI::ExistingDirectory);

    double synaptic_elements_init_lb{ 0.0 };
    double synaptic_elements_init_ub{ 0.0 };
    app.add_option("--synaptic-elements-lower-bound", synaptic_elements_init_lb, "The minimum number of vacant synaptic elements per neuron. Must be smaller of equal to synaptic-elements-upper-bound.");
    app.add_option("--synaptic-elements-upper-bound", synaptic_elements_init_ub, "The maximum number of vacant synaptic elements per neuron. Must be larger or equal to synaptic-elements-lower-bound.");

    double target_calcium{ SynapticElements::default_C_target };
    app.add_option("--target-ca", target_calcium, "The target Ca2+ ions in each neuron. Default is 0.7.");

    double initial_calcium{ 0.0 };
    app.add_option("--initial-ca", initial_calcium, "The initial Ca2+ ions in each neuron. Default is 0.0.");

    double nu{ SynapticElements::default_nu };
    app.add_option("--growth-rate", nu, "The growth rate for the synaptic elements. Default is 1e-5");

    double beta{ NeuronModel::default_beta };
    app.add_option("--beta", beta, "The amount of calcium ions gathered when a neuron fires. Default is 0.001");

    double min_calcium_axons{ SynapticElements::default_eta_Axons };
    app.add_option("--min-calcium-axons", min_calcium_axons, "The minimum intercellular calcium for axons to grow. Default is 0.4");

    double min_calcium_excitatory_dendrites{ SynapticElements::default_eta_Dendrites_exc };
    app.add_option("--min-calcium-excitatory-dendrites", min_calcium_excitatory_dendrites, "The minimum intercellular calcium for excitatory dendrites to grow. Default is 0.1");

    double min_calcium_inhibitory_dendrites{ SynapticElements::default_eta_Dendrites_inh };
    app.add_option("--min-calcium-inhibitory-dendrites", min_calcium_inhibitory_dendrites, "The minimum intercellular calcium for inhibitory dendrites to grow. Default is 0.0");

    CLI11_PARSE(app, argc, argv);

    RelearnException::check(synaptic_elements_init_lb >= 0.0, "The minimum number of vacant synaptic elements must not be negative");
    RelearnException::check(synaptic_elements_init_ub >= synaptic_elements_init_lb, "The minimum number of vacant synaptic elements must not be larger than the maximum number");
    RelearnException::check(static_cast<bool>(*opt_num_neurons) || static_cast<bool>(*opt_file_positions) || static_cast<bool>(*opt_num_neurons_per_rank),
        "Missing command line option, need a total number of neurons (-n,--num-neurons), a number of neurons per rank (--num-neurons-per-rank), or file_positions (-f,--file).");
    RelearnException::check(openmp_threads > 0, "Number of OpenMP Threads must be greater than 0 (or not set).");
    RelearnException::check(calcium_decay > 0.0, "The calcium decay constant must be greater than 0.");

    if (algorithm == AlgorithmEnum::BarnesHut || algorithm == AlgorithmEnum::BarnesHutInverted || algorithm == AlgorithmEnum::BarnesHutLocationAware) {
        RelearnException::check(accept_criterion <= BarnesHut::max_theta, "Acceptance criterion must be smaller or equal to {}", BarnesHut::max_theta);
        RelearnException::check(accept_criterion > 0.0, "Acceptance criterion must be larger than 0.0");
    } else if (algorithm == AlgorithmEnum::FastMultipoleMethods) {
        const auto accept_criterion_set = opt_accept_criterion->count() > 0;
        RelearnException::check(!accept_criterion_set, "Acceptance criterion can only be set if Barnes-Hut is used");
    } else if (algorithm == AlgorithmEnum::Naive) {
        const auto accept_criterion_set = opt_accept_criterion->count() > 0;
        RelearnException::check(!accept_criterion_set, "Acceptance criterion can only be set if Barnes-Hut is used");
    } else {
        RelearnException::fail("Wrong algorithm chosen");
    }

    RelearnException::check(target_calcium >= SynapticElements::min_C_target, "Target calcium is smaller than {}", SynapticElements::min_C_target);
    RelearnException::check(target_calcium <= SynapticElements::max_C_target, "Target calcium is larger than {}", SynapticElements::max_C_target);

    RelearnException::check(nu >= SynapticElements::min_nu, "Growth rate is smaller than {}", SynapticElements::min_nu);
    RelearnException::check(nu <= SynapticElements::max_nu, "Growth rate is larger than {}", SynapticElements::max_nu);

    Config::first_plasticity_update = first_plasticity_step;

    omp_set_num_threads(openmp_threads);

    /**
     * Initialize the simuliation log files
     */
    if (static_cast<bool>(*opt_log_path)) {
        LogFiles::set_output_path(log_path);
    }
    if (static_cast<bool>(*opt_log_prefix)) {
        LogFiles::set_general_prefix(log_prefix);
    }

    LogFiles::init();

    // Init random number seeds
    RandomHolder::seed(RandomHolderKey::Partition, static_cast<unsigned int>(my_rank));
    RandomHolder::seed(RandomHolderKey::Algorithm, random_seed);

    // Rank 0 prints start time of simulation
    MPIWrapper::barrier();
    if (0 == my_rank) {
        LogFiles::print_message_rank(0,
            "START: {}\n"
            "Chosen lower bound for vacant synaptic elements: {}\n"
            "Chosen upper bound for vacant synaptic elements: {}\n"
            "Chosen target calcium value: {}\n"
            "Chosen beta value: {}\n"
            "Chosen nu value: {}\n"
            "Chosen retract ratio: {}\n"
            "Chosen synapse conductance: {}\n"
            "Chosen background activity base: {}\n"
            "Chosen background activity mean: {}\n"
            "Chosen background activity stddev: {}",
            Timers::wall_clock_time(),
            synaptic_elements_init_lb,
            synaptic_elements_init_ub,
            target_calcium,
            beta,
            nu,
            retract_ratio,
            synapse_conductance,
            base_background_activity,
            background_activity_mean,
            background_activity_stddev);

        LogFiles::write_to_file(LogFiles::EventType::Essentials, false,
            "Number of steps: {}\n"
            "Chosen lower bound for vacant synaptic elements: {}\n"
            "Chosen upper bound for vacant synaptic elements: {}\n"
            "Chosen target calcium value: {}\n"
            "Chosen beta value: {}\n"
            "Chosen nu value: {}\n"
            "Chosen retract ratio: {}\n"
            "Chosen synapse conductance: {}\n"
            "Chosen background activity base: {}\n"
            "Chosen background activity mean: {}\n"
            "Chosen background activity stddev: {}",
            simulation_steps,
            synaptic_elements_init_lb,
            synaptic_elements_init_ub,
            target_calcium,
            beta,
            nu,
            retract_ratio,
            synapse_conductance,
            base_background_activity,
            background_activity_mean,
            background_activity_stddev);
    }

    LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdate, false, "#step: creations deletions netto");
    LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdateCSV, false, "#step;creations;deletions;netto");
    LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdateLocal, false, "#step: creations deletions netto");

    Timers::start(TimerRegion::INITIALIZATION);

    /**
     * Calculate what my partition of the domain consist of
     */
    auto partition = std::make_shared<Partition>(num_ranks, my_rank);
    const size_t number_local_subdomains = partition->get_number_local_subdomains();

    if (algorithm == AlgorithmEnum::BarnesHut || algorithm == AlgorithmEnum::BarnesHutLocationAware) {
        // Check if int type can contain total size of branch nodes to receive in bytes
        // Every rank sends the same number of branch nodes, which is Partition::get_number_local_subdomains()
        if (std::numeric_limits<int>::max() < (number_local_subdomains * sizeof(OctreeNode<BarnesHutCell>))) {
            RelearnException::fail("int type is too small to hold the size in bytes of the branch nodes that are received from every rank in MPI_Allgather()");
        }

        // Create MPI RMA memory allocator
        MPIWrapper::init_buffer_octree<BarnesHutCell>();
    } else if (algorithm == AlgorithmEnum::BarnesHutInverted) {
        // Check if int type can contain total size of branch nodes to receive in bytes
        // Every rank sends the same number of branch nodes, which is Partition::get_number_local_subdomains()
        if (std::numeric_limits<int>::max() < (number_local_subdomains * sizeof(OctreeNode<BarnesHutInvertedCell>))) {
            RelearnException::fail("int type is too small to hold the size in bytes of the branch nodes that are received from every rank in MPI_Allgather()");
        }

        // Create MPI RMA memory allocator
        MPIWrapper::init_buffer_octree<BarnesHutInvertedCell>();
    } else if (algorithm == AlgorithmEnum::FastMultipoleMethods) {
        // Check if int type can contain total size of branch nodes to receive in bytes
        // Every rank sends the same number of branch nodes, which is Partition::get_number_local_subdomains()
        if (std::numeric_limits<int>::max() < (number_local_subdomains * sizeof(OctreeNode<FastMultipoleMethodsCell>))) {
            RelearnException::fail("int type is too small to hold the size in bytes of the branch nodes that are received from every rank in MPI_Allgather()");
        }

        // Create MPI RMA memory allocator
        MPIWrapper::init_buffer_octree<FastMultipoleMethodsCell>();
    } else if (algorithm == AlgorithmEnum::Naive) {
        // Check if int type can contain total size of branch nodes to receive in bytes
        // Every rank sends the same number of branch nodes, which is Partition::get_number_local_subdomains()
        if (std::numeric_limits<int>::max() < (number_local_subdomains * sizeof(OctreeNode<NaiveCell>))) {
            RelearnException::fail("int type is too small to hold the size in bytes of the branch nodes that are received from every rank in MPI_Allgather()");
        }

        // Create MPI RMA memory allocator
        MPIWrapper::init_buffer_octree<NaiveCell>();
    }

    std::unique_ptr<NeuronModel> neuron_models;
    if (neuron_model == NeuronModelEnum::Poisson) {
        neuron_models = std::make_unique<models::PoissonModel>(synapse_conductance, NeuronModel::default_tau_C, beta, NeuronModel::default_h,
            base_background_activity, background_activity_mean, background_activity_stddev,
            models::PoissonModel::default_x_0, models::PoissonModel::default_tau_x, models::PoissonModel::default_refrac_time);
    } else if (neuron_model == NeuronModelEnum::Izhikevich) {
        neuron_models = std::make_unique<models::IzhikevichModel>(synapse_conductance, NeuronModel::default_tau_C, beta, NeuronModel::default_h,
            base_background_activity, background_activity_mean, background_activity_stddev,
            models::IzhikevichModel::default_a, models::IzhikevichModel::default_b, models::IzhikevichModel::default_c,
            models::IzhikevichModel::default_d, models::IzhikevichModel::default_V_spike, models::IzhikevichModel::default_k1,
            models::IzhikevichModel::default_k2, models::IzhikevichModel::default_k3);
    } else if (neuron_model == NeuronModelEnum::FitzHughNagumo) {
        neuron_models = std::make_unique<models::FitzHughNagumoModel>(synapse_conductance, NeuronModel::default_tau_C, NeuronModel::default_beta, NeuronModel::default_h,
            base_background_activity, background_activity_mean, background_activity_stddev,
            models::FitzHughNagumoModel::default_a, models::FitzHughNagumoModel::default_b, models::FitzHughNagumoModel::default_phi);
    } else if (neuron_model == NeuronModelEnum::AEIF) {
        neuron_models = std::make_unique<models::AEIFModel>(synapse_conductance, NeuronModel::default_tau_C, NeuronModel::default_beta, NeuronModel::default_h,
            base_background_activity, background_activity_mean, background_activity_stddev,
            models::AEIFModel::default_C, models::AEIFModel::default_g_L, models::AEIFModel::default_E_L, models::AEIFModel::default_V_T,
            models::AEIFModel::default_d_T, models::AEIFModel::default_tau_w, models::AEIFModel::default_a, models::AEIFModel::default_b,
            models::AEIFModel::default_V_spike);
    }

    auto axon_models = std::make_shared<SynapticElements>(ElementType::Axon, min_calcium_axons,
        nu, retract_ratio, synaptic_elements_init_lb, synaptic_elements_init_ub);

    auto dend_ex_models = std::make_shared<SynapticElements>(ElementType::Dendrite, min_calcium_excitatory_dendrites,
        nu, retract_ratio, synaptic_elements_init_lb, synaptic_elements_init_ub);

    auto dend_in_models = std::make_shared<SynapticElements>(ElementType::Dendrite, min_calcium_inhibitory_dendrites,
        nu, retract_ratio, synaptic_elements_init_lb, synaptic_elements_init_ub);

    // Lock local RMA memory for local stores
    MPIWrapper::lock_window(my_rank, MPI_Locktype::Exclusive);

    Simulation sim(partition);
    sim.set_neuron_model(std::move(neuron_models));
    sim.set_axons(std::move(axon_models));
    sim.set_dendrites_ex(std::move(dend_ex_models));
    sim.set_dendrites_in(std::move(dend_in_models));
    sim.set_probabilty_scaling_parameter(scaling_constant);

    if (algorithm == AlgorithmEnum::BarnesHut || algorithm == AlgorithmEnum::BarnesHutInverted || algorithm == AlgorithmEnum::BarnesHutLocationAware) {
        sim.set_acceptance_criterion_for_barnes_hut(accept_criterion);
    }

    sim.set_algorithm(algorithm);

    if (static_cast<bool>(*opt_num_neurons)) {
        auto sfnd = std::make_unique<SubdomainFromNeuronDensity>(number_neurons, 0.8, SubdomainFromNeuronDensity::default_um_per_neuron, partition);
        sim.set_subdomain_assignment(std::move(sfnd));
    } else if (static_cast<bool>(*opt_num_neurons_per_rank)) {
        auto sfdpr = std::make_unique<SubdomainFromNeuronPerRank>(number_neurons_per_rank, 0.8, SubdomainFromNeuronPerRank::default_um_per_neuron, partition);
        sim.set_subdomain_assignment(std::move(sfdpr));
    } else {
        std::optional<std::filesystem::path> path_to_network{};
        if (static_cast<bool>(*opt_file_network)) {
            path_to_network = file_network;
        }

        auto sff = std::make_unique<SubdomainFromFile>(file_positions, std::move(path_to_network), partition);
        sim.set_subdomain_assignment(std::move(sff));
    }

    if (*opt_file_enable_interrupts) {
        auto enable_interrupts = InteractiveNeuronIO::load_enable_interrups(file_enable_interrupts);
        sim.set_enable_interrupts(std::move(enable_interrupts));
    }

    if (*opt_file_disable_interrupts) {
        auto disable_interrupts = InteractiveNeuronIO::load_disable_interrups(file_disable_interrupts);
        sim.set_disable_interrupts(std::move(disable_interrupts));
    }

    if (*opt_file_creation_interrups) {
        auto creation_interrups = InteractiveNeuronIO::load_creation_interrups(file_creation_interrupts);
        sim.set_creation_interrupts(std::move(creation_interrups));
    }

    auto target_calcium_calculator = [target = target_calcium](NeuronID::value_type /*neuron_id*/) { return target; };
    sim.set_target_calcium_calculator(std::move(target_calcium_calculator));

    auto initial_calcium_calculator = [inital = initial_calcium](NeuronID::value_type /*neuron_id*/) { return inital; };
    sim.set_initial_calcium_calculator(std::move(initial_calcium_calculator));

    /**********************************************************************************/

    // The barrier ensures that every rank finished its local stores.
    // Otherwise, a "fast" rank might try to read from the RMA window of another
    // rank which has not finished (or even begun) its local stores
    MPIWrapper::barrier(); // TODO(future) Really needed?

    const auto steps_per_simulation = simulation_steps / Config::monitor_step;
    sim.increase_monitoring_capacity(steps_per_simulation);

    sim.initialize();

    // Unlock local RMA memory and make local stores visible in public window copy
    MPIWrapper::unlock_window(my_rank);

    Timers::stop_and_add(TimerRegion::INITIALIZATION);

    sim.register_neuron_monitor(NeuronID{ 6 });
    sim.register_neuron_monitor(NeuronID{ 1164 });
    sim.register_neuron_monitor(NeuronID{ 28001 });

    auto simulate = [&]() {
        sim.simulate(simulation_steps);

        Timers::print();

        MPIWrapper::barrier();

        sim.finalize();
    };

    simulate();

    if (static_cast<bool>(*flag_interactive)) {
        while (true) {
            spdlog::info("Interactive run. Run another {} simulation steps? [y/n]\n", simulation_steps);
            char yn{ 'n' };
            std::cin >> std::ws >> yn;

            if (yn == 'n' || yn == 'N') {
                break;
            }

            if (yn == 'y' || yn == 'Y') {
                sim.increase_monitoring_capacity(steps_per_simulation);
                simulate();
            } else {
                RelearnException::fail("Input for question to run another {} simulation steps was not valid.", simulation_steps);
            }
        }
    }

    LogFiles::write_to_file(LogFiles::EventType::Cout, true, "number of bytes send: {}, number of bytes received: {}, number of bytes accessed remotely: {}", MPIWrapper::get_number_bytes_sent(), MPIWrapper::get_number_bytes_received(), MPIWrapper::get_number_bytes_remote_accessed());

    MPIWrapper::finalize();

    return 0;
}
