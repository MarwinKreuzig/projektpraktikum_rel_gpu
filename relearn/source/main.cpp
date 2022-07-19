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
#include "io/CalciumIO.h"
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
#include "util/StepParser.h"
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
#include <climits>
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

    constexpr auto number_bits_in_byte = CHAR_BIT;

    constexpr auto sizeof_vec3_double = sizeof(Vec3d);
    constexpr auto sizeof_vec3_size_t = sizeof(Vec3s);

    constexpr auto sizeof_virtual_plasticity_element = sizeof(VirtualPlasticityElement);

    constexpr auto sizeof_empty_t = sizeof(empty_t);
    constexpr auto sizeof_fmm_cell_attributes = sizeof(FastMultipoleMethodsCell);
    constexpr auto sizeof_bh_cell_attributes = sizeof(BarnesHutCell);
    constexpr auto sizeof_bh_naive_attributes = sizeof(NaiveCell);

    constexpr auto sizeof_empty_cell = sizeof(Cell<empty_t>);
    constexpr auto sizeof_fmm_cell = sizeof(Cell<FastMultipoleMethodsCell>);
    constexpr auto sizeof_bh_cell = sizeof(Cell<BarnesHutCell>);
    constexpr auto sizeof_naive_cell = sizeof(Cell<NaiveCell>);

    constexpr auto sizeof_octreenode = sizeof(OctreeNode<empty_t>);
    constexpr auto sizeof_fmm_octreenode = sizeof(OctreeNode<FastMultipoleMethodsCell>);
    constexpr auto sizeof_bh_octreenode = sizeof(OctreeNode<BarnesHutCell>);
    constexpr auto sizeof_naive_octreenode = sizeof(OctreeNode<NaiveCell>);

    constexpr auto sizeof_neuron_id = sizeof(NeuronID);
    constexpr auto sizeof_rank_neuron_id = sizeof(RankNeuronId);

    constexpr auto sizeof_local_synapse = sizeof(LocalSynapse);
    constexpr auto sizeof_distant_in_synapse = sizeof(DistantInSynapse);
    constexpr auto sizeof_distant_out_synapse = sizeof(DistantOutSynapse);

    std::stringstream ss{};

    ss << '\n';

    ss << "Number of bits in a byte: " << number_bits_in_byte << '\n';

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
        LogFiles::print_message_rank(0, "I'm performing Debug Checks");
    } else {
        LogFiles::print_message_rank(0, "I'm skipping Debug Checks");
    }

    const auto my_rank = MPIWrapper::get_my_rank();
    const auto num_ranks = MPIWrapper::get_num_ranks();

    // Command line arguments
    CLI::App app{ "" };

    AlgorithmEnum chosen_algorithm = AlgorithmEnum::BarnesHut;
    std::map<std::string, AlgorithmEnum> cli_parse_algorithm{
        { "naive", AlgorithmEnum::Naive },
        { "barnes-hut", AlgorithmEnum::BarnesHut },
        { "barnes-hut-inverted", AlgorithmEnum::BarnesHutInverted },
        { "barnes-hut-location-aware", AlgorithmEnum::BarnesHutLocationAware },
        { "fast-multipole-methods", AlgorithmEnum::FastMultipoleMethods }
    };

    NeuronModelEnum chosen_neuron_model = NeuronModelEnum::Poisson;
    std::map<std::string, NeuronModelEnum> cli_parse_neuron_model{
        { "poisson", NeuronModelEnum::Poisson },
        { "izhikevich", NeuronModelEnum::Izhikevich },
        { "aeif", NeuronModelEnum::AEIF },
        { "fitzhughnagumo", NeuronModelEnum::FitzHughNagumo }
    };

    KernelType chosen_kernel_type = KernelType::Gaussian;
    std::map<std::string, KernelType> cli_parse_kernel_type{
        { "gamma", KernelType::Gamma },
        { "gaussian", KernelType::Gaussian },
        { "linear", KernelType::Linear },
        { "weibull", KernelType::Weibull }
    };

    size_t simulation_steps{};
    app.add_option("-s,--steps", simulation_steps, "Simulation steps in ms.")->required();

    size_t first_plasticity_step{ Config::first_plasticity_update };
    app.add_option("--first-plasticity-step", first_plasticity_step, "The first step in which the plasticity is updated.");

    size_t calcium_log_step{ Config::calcium_log_step };
    app.add_option("--calcium-log-step", calcium_log_step, "Sets the interval for logging all calcium values.");

    const auto* flag_interactive = app.add_flag("-i,--interactive", "Run interactively.");

    unsigned int random_seed{ 0 };
    app.add_option("-r,--random-seed", random_seed, "Random seed. Default: 0.");

    int openmp_threads{ 1 };
    app.add_option("--openmp", openmp_threads, "Number of OpenMP Threads.");

    std::filesystem::path log_path{};
    auto* const opt_log_path = app.add_option("-l,--log-path", log_path, "Path for log files.");

    std::string log_prefix{};
    const auto* opt_log_prefix = app.add_option("-p,--log-prefix", log_prefix, "Prefix for log files.");

    const auto* flag_disable_positions = app.add_flag("--no-print-positions", "Disables printing the positions to a file.");
    const auto* flag_disable_network = app.add_flag("--no-print-network", "Disables printing the network to a file.");
    const auto* flag_disable_plasticity = app.add_flag("--no-print-plasticity", "Disables printing the plasticity changes to a file.");

    size_t number_neurons{};
    auto* const opt_num_neurons = app.add_option("-n,--num-neurons", number_neurons, "Number of neurons. This option only works with one MPI rank!");

    size_t number_neurons_per_rank{};
    auto* const opt_num_neurons_per_rank = app.add_option("--num-neurons-per-rank", number_neurons_per_rank, "Number neurons per MPI rank.");

    double fraction_excitatory_neurons{ 1.0 };
    app.add_option("--fraction-excitatory-neurons", fraction_excitatory_neurons, "The fraction of excitatory neurons, must be from [0.0, 1.0]. Required --num-neurons or --num-neurons-per-rank to take effect.");

    double um_per_neuron{ 1.0 };
    app.add_option("--um-per-neuron", um_per_neuron, "The micrometer per neuron in one dimension, must be from (0.0, \\inf). Required --num-neurons or --num-neurons-per-rank to take effect.");

    std::filesystem::path file_positions{};
    auto* const opt_file_positions = app.add_option("-f,--file", file_positions, "File with neuron positions. This option only works with one MPI rank!");

    std::filesystem::path file_network{};
    auto* const opt_file_network = app.add_option("-g,--graph", file_network, "File with neuron connections. This option only works with one MPI rank!");

    std::filesystem::path file_enable_interrupts{};
    auto* const opt_file_enable_interrupts = app.add_option("--enable-interrupts", file_enable_interrupts, "File with the enable interrupts.");

    std::filesystem::path file_disable_interrupts{};
    auto* const opt_file_disable_interrupts = app.add_option("--disable-interrupts", file_disable_interrupts, "File with the disable interrupts.");

    std::filesystem::path file_creation_interrupts{};
    auto* const opt_file_creation_interrups = app.add_option("--creation-interrupts", file_creation_interrupts, "File with the creation interrupts.");

    auto* const opt_algorithm = app.add_option("-a,--algorithm", chosen_algorithm, "The algorithm that is used for finding the targets");
    opt_algorithm->required()->transform(CLI::CheckedTransformer(cli_parse_algorithm, CLI::ignore_case));

    double accept_criterion{ BarnesHut::default_theta };
    const auto* const opt_accept_criterion = app.add_option("-t,--theta", accept_criterion, "Theta, the acceptance criterion for Barnes-Hut. Default: 0.3. Requires Barnes-Hut or inverted Barnes-Hut.");

    auto* const opt_kernel_type = app.add_option("--kernel-type", chosen_kernel_type, "The probability kernel type, cannot be set for the fast multipole methods.");
    opt_kernel_type->transform(CLI::CheckedTransformer(cli_parse_kernel_type, CLI::ignore_case));

    double gamma_k{ GammaDistributionKernel::default_k };
    app.add_option("--gamma-k", gamma_k, "Shape parameter for the gamma probability kernel.");

    double gamma_theta{ GammaDistributionKernel::default_theta };
    app.add_option("--gamma-theta", gamma_theta, "Scale parameter for the gamma probability kernel.");

    double gaussian_sigma{ GaussianDistributionKernel::default_sigma };
    app.add_option("--gaussian-sigma", gaussian_sigma, "Scaling parameter for the gaussian probability kernel. Default: 750");

    double gaussian_mu{ GaussianDistributionKernel::default_mu };
    app.add_option("--gaussian-mu", gaussian_mu, "Translation parameter for the gaussian probability kernel. Default: 0");

    double linear_cutoff{ LinearDistributionKernel::default_cutoff };
    app.add_option("--linear-cutoff", linear_cutoff, "Cut-off parameter for the linear probability kernel. Default: +inf");

    double weibull_k{ WeibullDistributionKernel::default_k };
    app.add_option("--weibull-k", weibull_k, "Shape parameter for the weibull probability kernel.");

    double weibull_b{ WeibullDistributionKernel::default_b };
    app.add_option("--weibull-b", weibull_b, "Scale parameter for the weibull probability kernel.");

    auto* const opt_neuron_model = app.add_option("--neuron-model", chosen_neuron_model, "The neuron model");
    opt_neuron_model->transform(CLI::CheckedTransformer(cli_parse_neuron_model, CLI::ignore_case));

    double base_background_activity{ NeuronModel::default_base_background_activity };
    app.add_option("--base-background-activity", base_background_activity, "The base background activity by which all neurons are excited. The background activity is calculated as <base> + N(mean, stddev)");

    double background_activity_mean{ NeuronModel::default_background_activity_mean };
    app.add_option("--background-activity-mean", background_activity_mean, "The mean background activity by which all neurons are excited. The background activity is calculated as <base> + N(mean, stddev)");

    double background_activity_stddev{ NeuronModel::default_background_activity_stddev };
    app.add_option("--background-activity-stddev", background_activity_stddev, "The standard deviation of the background activity by which all neurons are excited. The background activity is calculated as <base> + N(mean, stddev)");

    double synapse_conductance{ NeuronModel::default_k };
    app.add_option("--synapse-conductance", synapse_conductance, "The activity that is transfered to its neighbors when a neuron spikes. Default is 0.03");

    double calcium_decay{ NeuronModel::default_tau_C };
    app.add_option("--calcium-decay", calcium_decay, "The decay constant for the intercellular calcium. Must be greater than 0.0");

    double target_calcium{ SynapticElements::default_C_target };
    auto* const opt_target_calcium = app.add_option("--target-ca", target_calcium, "The target Ca2+ ions in each neuron. Default is 0.7.");

    double initial_calcium{ 0.0 };
    auto* const opt_initial_calcium = app.add_option("--initial-ca", initial_calcium, "The initial Ca2+ ions in each neuron. Default is 0.0.");

    std::string file_calcium{};
    auto* const opt_file_calcium = app.add_option("--file_calcium", file_calcium, "File with calcium values.");

    double beta{ NeuronModel::default_beta };
    app.add_option("--beta", beta, "The amount of calcium ions gathered when a neuron fires. Default is 0.001.");

    size_t h{ NeuronModel::default_h };
    app.add_option("--integration-step-size", h, "The step size for the numerical integration of the electrical acticity. Default is 10.");

    double retract_ratio{ SynapticElements::default_vacant_retract_ratio };
    app.add_option("--retract-ratio", retract_ratio, "The ratio by which vacant synapses retract.");

    double synaptic_elements_init_lb{ 0.0 };
    app.add_option("--synaptic-elements-lower-bound", synaptic_elements_init_lb, "The minimum number of vacant synaptic elements per neuron. Must be smaller of equal to synaptic-elements-upper-bound.");

    double synaptic_elements_init_ub{ 0.0 };
    app.add_option("--synaptic-elements-upper-bound", synaptic_elements_init_ub, "The maximum number of vacant synaptic elements per neuron. Must be larger or equal to synaptic-elements-lower-bound.");

    double growth_rate{ SynapticElements::default_nu };
    app.add_option("--growth-rate", growth_rate, "The growth rate for the synaptic elements. Default is 1e-5");

    double min_calcium_axons{ SynapticElements::default_eta_Axons };
    app.add_option("--min-calcium-axons", min_calcium_axons, "The minimum intercellular calcium for axons to grow. Default is 0.4");

    double min_calcium_excitatory_dendrites{ SynapticElements::default_eta_Dendrites_exc };
    app.add_option("--min-calcium-excitatory-dendrites", min_calcium_excitatory_dendrites, "The minimum intercellular calcium for excitatory dendrites to grow. Default is 0.1");

    double min_calcium_inhibitory_dendrites{ SynapticElements::default_eta_Dendrites_inh };
    app.add_option("--min-calcium-inhibitory-dendrites", min_calcium_inhibitory_dendrites, "The minimum intercellular calcium for inhibitory dendrites to grow. Default is 0.0");

    opt_num_neurons->excludes(opt_file_positions);
    opt_num_neurons->excludes(opt_file_network);
    opt_num_neurons->excludes(opt_num_neurons_per_rank);

    opt_num_neurons_per_rank->excludes(opt_num_neurons);
    opt_num_neurons_per_rank->excludes(opt_file_positions);
    opt_num_neurons_per_rank->excludes(opt_file_network);

    opt_file_positions->excludes(opt_num_neurons);
    opt_file_network->excludes(opt_num_neurons);
    opt_file_positions->excludes(opt_num_neurons_per_rank);
    opt_file_network->excludes(opt_num_neurons_per_rank);

    opt_file_network->needs(opt_file_positions);

    opt_file_positions->check(CLI::ExistingFile);
    opt_file_network->check(CLI::ExistingFile);

    opt_file_calcium->excludes(opt_initial_calcium);
    opt_file_calcium->excludes(opt_target_calcium);
    opt_initial_calcium->excludes(opt_file_calcium);
    opt_target_calcium->excludes(opt_file_calcium);

    opt_file_calcium->check(CLI::ExistingFile);

    opt_file_enable_interrupts->check(CLI::ExistingFile);
    opt_file_disable_interrupts->check(CLI::ExistingFile);
    opt_file_creation_interrups->check(CLI::ExistingFile);

    opt_log_path->check(CLI::ExistingDirectory);

    CLI11_PARSE(app, argc, argv);

    if (static_cast<bool>(*opt_accept_criterion)) {
        RelearnException::check(is_barnes_hut(chosen_algorithm), "Acceptance criterion can only be set if Barnes-Hut is used");
        RelearnException::check(accept_criterion <= BarnesHut::max_theta, "Acceptance criterion must be smaller or equal to {}", BarnesHut::max_theta);
        RelearnException::check(accept_criterion > 0.0, "Acceptance criterion must be larger than 0.0");
    }

    if (static_cast<bool>(*opt_num_neurons)) {
        RelearnException::check(num_ranks == 1, "The option --num-neurons can only be used for one MPI rank. There are {} ranks.", num_ranks);
    }

    if (static_cast<bool>(*opt_file_positions)) {
        RelearnException::check(num_ranks == 1, "The option --file can only be used for one MPI rank. There are {} ranks.", num_ranks);
    }

    RelearnException::check(fraction_excitatory_neurons >= 0.0 && fraction_excitatory_neurons <= 1.0, "The fraction of excitatory neurons must be from [0.0, 1.0]");
    RelearnException::check(um_per_neuron > 0.0, "The micrometer per neuron must be greater than 0.0.");

    RelearnException::check(synaptic_elements_init_lb >= 0.0, "The minimum number of vacant synaptic elements must not be negative");
    RelearnException::check(synaptic_elements_init_ub >= synaptic_elements_init_lb, "The minimum number of vacant synaptic elements must not be larger than the maximum number");
    RelearnException::check(static_cast<bool>(*opt_num_neurons) || static_cast<bool>(*opt_file_positions) || static_cast<bool>(*opt_num_neurons_per_rank),
        "Missing command line option, need a total number of neurons (-n,--num-neurons), a number of neurons per rank (--num-neurons-per-rank), or file_positions (-f,--file).");
    RelearnException::check(openmp_threads > 0, "Number of OpenMP Threads must be greater than 0 (or not set).");
    RelearnException::check(calcium_decay > 0.0, "The calcium decay constant must be greater than 0.");

    if (static_cast<bool>(*opt_target_calcium)) {
        RelearnException::check(target_calcium >= SynapticElements::min_C_target, "Target calcium is smaller than {}", SynapticElements::min_C_target);
        RelearnException::check(target_calcium <= SynapticElements::max_C_target, "Target calcium is larger than {}", SynapticElements::max_C_target);
    }

    RelearnException::check(growth_rate >= SynapticElements::min_nu, "Growth rate is smaller than {}", SynapticElements::min_nu);
    RelearnException::check(growth_rate <= SynapticElements::max_nu, "Growth rate is larger than {}", SynapticElements::max_nu);

    Config::first_plasticity_update = first_plasticity_step;
    Config::calcium_log_step = calcium_log_step;

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

    if (static_cast<bool>(*flag_disable_positions)) {
        LogFiles::set_log_status(LogFiles::EventType::Positions, true);
    }
    if (static_cast<bool>(*flag_disable_network)) {
        LogFiles::set_log_status(LogFiles::EventType::InNetwork, true);
        LogFiles::set_log_status(LogFiles::EventType::OutNetwork, true);
        LogFiles::set_log_status(LogFiles::EventType::Network, true);
        LogFiles::set_log_status(LogFiles::EventType::NetworkInExcitatoryHistogramLocal, true);
        LogFiles::set_log_status(LogFiles::EventType::NetworkInInhibitoryHistogramLocal, true);
        LogFiles::set_log_status(LogFiles::EventType::NetworkOutHistogramLocal, true);
    }
    if (static_cast<bool>(*flag_disable_plasticity)) {
        LogFiles::set_log_status(LogFiles::EventType::PlasticityUpdate, true);
        LogFiles::set_log_status(LogFiles::EventType::PlasticityUpdateCSV, true);
        LogFiles::set_log_status(LogFiles::EventType::PlasticityUpdateLocal, true);
    }

    LogFiles::init();

    // Init random number seeds
    RandomHolder::seed(RandomHolderKey::Partition, static_cast<unsigned int>(my_rank));
    RandomHolder::seed(RandomHolderKey::Algorithm, random_seed);

    // Rank 0 prints start time of simulation
    MPIWrapper::barrier();
    if (0 == my_rank) {
        LogFiles::write_to_file(LogFiles::EventType::Essentials, true,
            "START OF SIMULATION: {}\n"
            "Number of steps: {}\n"
            "Chosen lower bound for vacant synaptic elements: {}\n"
            "Chosen upper bound for vacant synaptic elements: {}\n"
            "Chosen target calcium value: {}\n"
            "Chosen beta value: {}\n"
            "Chosen calcium decay: {}\n"
            "Chosen growth_rate value: {}\n"
            "Chosen retract ratio: {}\n"
            "Chosen synapse conductance: {}\n"
            "Chosen background activity base: {}\n"
            "Chosen background activity mean: {}\n"
            "Chosen background activity stddev: {}\n"
            "Chosen kernel type: {}",
            Timers::wall_clock_time(),
            simulation_steps,
            synaptic_elements_init_lb,
            synaptic_elements_init_ub,
            target_calcium,
            beta,
            calcium_decay,
            growth_rate,
            retract_ratio,
            synapse_conductance,
            base_background_activity,
            background_activity_mean,
            background_activity_stddev,
            chosen_kernel_type);

        if (chosen_kernel_type == KernelType::Gamma) {
            LogFiles::write_to_file(LogFiles::EventType::Essentials, true,
                "Chosen shape parameter: {}\n"
                "Chosen scale parameter: {}",
                gamma_k,
                gamma_theta);
        } else if (chosen_kernel_type == KernelType::Gaussian) {
            LogFiles::write_to_file(LogFiles::EventType::Essentials, true,
                "Chosen translation parameter: {}\n"
                "Chosen scale parameter: {}",
                gaussian_mu,
                gaussian_sigma);
        } else if (chosen_kernel_type == KernelType::Linear) {
            LogFiles::write_to_file(LogFiles::EventType::Essentials, true,
                "Chosen cut-off parameter: {}",
                linear_cutoff);
        } else if (chosen_kernel_type == KernelType::Weibull) {
            LogFiles::write_to_file(LogFiles::EventType::Essentials, true,
                "Chosen shape parameter: {}\n"
                "Chosen scale parameter: {}",
                weibull_k,
                weibull_b);
        }
    }

    LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdate, false, "#step: creations deletions netto");
    LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdateCSV, false, "#step;creations;deletions;netto");
    LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdateLocal, false, "#step: creations deletions netto");

    Timers::start(TimerRegion::INITIALIZATION);

    // Set the correct kernel and initalize the MPIWrapper to return the correct type
    if (chosen_algorithm == AlgorithmEnum::BarnesHut || chosen_algorithm == AlgorithmEnum::BarnesHutLocationAware) {
        Kernel<BarnesHutCell>::set_kernel_type(chosen_kernel_type);
        MPIWrapper::init_buffer_octree<BarnesHutCell>();
    } else if (chosen_algorithm == AlgorithmEnum::BarnesHutInverted) {
        Kernel<BarnesHutInvertedCell>::set_kernel_type(chosen_kernel_type);
        MPIWrapper::init_buffer_octree<BarnesHutInvertedCell>();
    } else if (chosen_algorithm == AlgorithmEnum::FastMultipoleMethods) {
        Kernel<FastMultipoleMethodsCell>::set_kernel_type(chosen_kernel_type);
        MPIWrapper::init_buffer_octree<FastMultipoleMethodsCell>();
    } else {
        RelearnException::check(chosen_algorithm == AlgorithmEnum::Naive, "An algorithm was chosen that is not supported");
        Kernel<NaiveCell>::set_kernel_type(chosen_kernel_type);
        MPIWrapper::init_buffer_octree<NaiveCell>();
    }

    if (is_fast_multipole_method(chosen_algorithm)) {
        RelearnException::check(chosen_kernel_type == KernelType::Gaussian, "Setting the probability kernel type is not supported for the fast multipole methods!");
    }

    std::unique_ptr<NeuronModel> neuron_model{};
    if (chosen_neuron_model == NeuronModelEnum::Poisson) {
        neuron_model = std::make_unique<models::PoissonModel>(synapse_conductance, calcium_decay, beta, h,
            base_background_activity, background_activity_mean, background_activity_stddev,
            models::PoissonModel::default_x_0, models::PoissonModel::default_tau_x, models::PoissonModel::default_refrac_time);
    } else if (chosen_neuron_model == NeuronModelEnum::Izhikevich) {
        neuron_model = std::make_unique<models::IzhikevichModel>(synapse_conductance, calcium_decay, beta, h,
            base_background_activity, background_activity_mean, background_activity_stddev,
            models::IzhikevichModel::default_a, models::IzhikevichModel::default_b, models::IzhikevichModel::default_c,
            models::IzhikevichModel::default_d, models::IzhikevichModel::default_V_spike, models::IzhikevichModel::default_k1,
            models::IzhikevichModel::default_k2, models::IzhikevichModel::default_k3);
    } else if (chosen_neuron_model == NeuronModelEnum::FitzHughNagumo) {
        neuron_model = std::make_unique<models::FitzHughNagumoModel>(synapse_conductance, calcium_decay, beta, h,
            base_background_activity, background_activity_mean, background_activity_stddev,
            models::FitzHughNagumoModel::default_a, models::FitzHughNagumoModel::default_b, models::FitzHughNagumoModel::default_phi);
    } else if (chosen_neuron_model == NeuronModelEnum::AEIF) {
        neuron_model = std::make_unique<models::AEIFModel>(synapse_conductance, calcium_decay, beta, h,
            base_background_activity, background_activity_mean, background_activity_stddev,
            models::AEIFModel::default_C, models::AEIFModel::default_g_L, models::AEIFModel::default_E_L, models::AEIFModel::default_V_T,
            models::AEIFModel::default_d_T, models::AEIFModel::default_tau_w, models::AEIFModel::default_a, models::AEIFModel::default_b,
            models::AEIFModel::default_V_spike);
    }

    auto axons_model = std::make_shared<SynapticElements>(ElementType::Axon, min_calcium_axons,
        growth_rate, retract_ratio, synaptic_elements_init_lb, synaptic_elements_init_ub);

    auto excitatory_dendrites_model = std::make_shared<SynapticElements>(ElementType::Dendrite, min_calcium_excitatory_dendrites,
        growth_rate, retract_ratio, synaptic_elements_init_lb, synaptic_elements_init_ub);

    auto inhibitory_dendrites_model = std::make_shared<SynapticElements>(ElementType::Dendrite, min_calcium_inhibitory_dendrites,
        growth_rate, retract_ratio, synaptic_elements_init_lb, synaptic_elements_init_ub);

    /**
     * Calculate what my partition of the domain consist of
     */
    auto partition = std::make_shared<Partition>(num_ranks, my_rank);

    Simulation sim(partition);
    sim.set_neuron_model(std::move(neuron_model));
    sim.set_axons(std::move(axons_model));
    sim.set_dendrites_ex(std::move(excitatory_dendrites_model));
    sim.set_dendrites_in(std::move(inhibitory_dendrites_model));

    // Set the parameters for all kernel types, even though only one is used later one
    GammaDistributionKernel::set_k(gamma_k);
    GammaDistributionKernel::set_theta(gamma_theta);

    GaussianDistributionKernel::set_sigma(gaussian_sigma);
    GaussianDistributionKernel::set_mu(gaussian_mu);

    LinearDistributionKernel::set_cutoff(linear_cutoff);

    WeibullDistributionKernel::set_b(weibull_b);
    WeibullDistributionKernel::set_k(weibull_k);

    if (is_barnes_hut(chosen_algorithm)) {
        sim.set_acceptance_criterion_for_barnes_hut(accept_criterion);
    }

    sim.set_algorithm(chosen_algorithm);

    if (static_cast<bool>(*opt_num_neurons)) {
        auto sfnd = std::make_unique<SubdomainFromNeuronDensity>(number_neurons, fraction_excitatory_neurons, um_per_neuron, partition);
        sim.set_subdomain_assignment(std::move(sfnd));
    } else if (static_cast<bool>(*opt_num_neurons_per_rank)) {
        auto sfdpr = std::make_unique<SubdomainFromNeuronPerRank>(number_neurons_per_rank, fraction_excitatory_neurons, um_per_neuron, partition);
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

    if (*opt_file_calcium) {
        auto [initial_calcium_calculator, target_calcium_calculator] = CalciumIO::load_initial_and_target_function(file_calcium);

        sim.set_initial_calcium_calculator(std::move(initial_calcium_calculator));
        sim.set_target_calcium_calculator(std::move(target_calcium_calculator));
    } else {
        auto initial_calcium_calculator = [inital = initial_calcium](int /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return inital; };
        sim.set_initial_calcium_calculator(std::move(initial_calcium_calculator));

        auto target_calcium_calculator = [target = target_calcium](int /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return target; };
        sim.set_target_calcium_calculator(std::move(target_calcium_calculator));
    }

    const auto steps_per_simulation = simulation_steps / Config::monitor_step;
    sim.increase_monitoring_capacity(steps_per_simulation);

    /**********************************************************************************/

    // The barrier ensures that every rank finished its local stores.
    // Otherwise, a "fast" rank might try to read from the RMA window of another
    // rank which has not finished (or even begun) its local stores
    MPIWrapper::barrier(); // TODO(future) Really needed?

    // Lock local RMA memory for local stores
    MPIWrapper::lock_window(my_rank, MPI_Locktype::Exclusive);

    sim.initialize();

    // Unlock local RMA memory and make local stores visible in public window copy
    MPIWrapper::unlock_window(my_rank);

    Timers::stop_and_add(TimerRegion::INITIALIZATION);

    //sim.register_neuron_monitor(NeuronID{ 6 });
    //sim.register_neuron_monitor(NeuronID{ 1164 });
    //sim.register_neuron_monitor(NeuronID{ 28001 });

    auto simulate = [&sim, &simulation_steps]() {
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
