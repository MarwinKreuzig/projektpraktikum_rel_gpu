/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Simulation.h"

#include "algorithm/Algorithms.h"
#include "io/LogFiles.h"
#include "mpi/MPIWrapper.h"
#include "neurons/NetworkGraph.h"
#include "neurons/Neurons.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/models/NeuronModels.h"
#include "sim/NeuronToSubdomainAssignment.h"
#include "sim/SynapseLoader.h"
#include "structure/Octree.h"
#include "structure/Partition.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <fstream>
#include <iomanip>
#include <sstream>

Simulation::Simulation(std::shared_ptr<Partition> partition)
    : partition(std::move(partition)) {

    monitors = std::make_shared<std::vector<NeuronMonitor>>();
    area_monitors = std::make_shared<std::vector<AreaMonitor>>();
}

void Simulation::register_neuron_monitor(const NeuronID& neuron_id) {
    monitors->emplace_back(neuron_id);
}

void Simulation::set_acceptance_criterion_for_barnes_hut(const double value) {
    // Needed to avoid creating autapses
    RelearnException::check(value <= BarnesHut::max_theta,
        "Simulation::set_acceptance_criterion_for_barnes_hut: Acceptance criterion must be smaller or equal to {} but was {}", BarnesHut::max_theta, value);
    RelearnException::check(value > 0.0, "Simulation::set_acceptance_criterion_for_barnes_hut: Acceptance criterion must larger than 0.0, but it was {}", value);

    accept_criterion = value;
}

void Simulation::set_neuron_model(std::unique_ptr<NeuronModel>&& nm) noexcept {
    neuron_models = std::move(nm);
}

void Simulation::set_calcium_calculator(std::unique_ptr<CalciumCalculator>&& calculator) noexcept {
    calcium_calculator = std::move(calculator);
}

void Simulation::set_axons(std::shared_ptr<SynapticElements>&& se) noexcept {
    axons = std::move(se);
}

void Simulation::set_dendrites_ex(std::shared_ptr<SynapticElements>&& se) noexcept {
    dendrites_ex = std::move(se);
}

void Simulation::set_dendrites_in(std::shared_ptr<SynapticElements>&& se) noexcept {
    dendrites_in = std::move(se);
}

void Simulation::set_enable_interrupts(std::vector<std::pair<size_t, std::vector<NeuronID>>> interrupts) {
    enable_interrupts = std::move(interrupts);

    for (auto& [step, ids] : enable_interrupts) {
        std::sort(ids.begin(), ids.end());
    }
}

void Simulation::set_disable_interrupts(std::vector<std::pair<size_t, std::vector<NeuronID>>> interrupts) {
    disable_interrupts = std::move(interrupts);

    for (auto& [step, ids] : disable_interrupts) {
        std::sort(ids.begin(), ids.end());
    }
}

void Simulation::set_creation_interrupts(std::vector<std::pair<size_t, size_t>> interrupts) noexcept {
    creation_interrupts = std::move(interrupts);
}

void Simulation::set_algorithm(const AlgorithmEnum algorithm) noexcept {
    algorithm_enum = algorithm;
}

void Simulation::set_subdomain_assignment(std::unique_ptr<NeuronToSubdomainAssignment>&& subdomain_assignment) noexcept {
    neuron_to_subdomain_assignment = std::move(subdomain_assignment);
}

void Simulation::initialize() {
    RelearnException::check(neuron_models != nullptr, "Simulation::initialize: neuron_models is nullptr");
    RelearnException::check(calcium_calculator != nullptr, "Simulation::initialize: calcium_calculator is nullptr");
    RelearnException::check(axons != nullptr, "Simulation::initialize: axons is nullptr");
    RelearnException::check(dendrites_ex != nullptr, "Simulation::initialize: dendrites_ex is nullptr");
    RelearnException::check(dendrites_in != nullptr, "Simulation::initialize: dendrites_in is nullptr");
    RelearnException::check(neuron_to_subdomain_assignment != nullptr, "Simulation::initialize: neuron_to_subdomain_assignment is nullptr");

    neuron_to_subdomain_assignment->initialize();
    const auto number_total_neurons = neuron_to_subdomain_assignment->get_total_number_placed_neurons();

    partition->set_total_number_neurons(number_total_neurons);
    const auto number_local_neurons = partition->get_number_local_neurons();

    const auto my_rank = MPIWrapper::get_my_rank();
    RelearnException::check(number_local_neurons > 0, "I have 0 neurons at rank {}", my_rank);

    neurons = std::make_shared<Neurons>(partition, std::move(neuron_models), std::move(calcium_calculator), axons, dendrites_ex, dendrites_in);
    neurons->init(number_local_neurons);
    NeuronMonitor::neurons_to_monitor = neurons;

    auto number_local_neurons_ntsa = neuron_to_subdomain_assignment->get_number_neurons_in_subdomains();

    RelearnException::check(number_local_neurons_ntsa == number_local_neurons,
        "Simulation::initialize: The partition and the NTSA had a disagreement about the number of local neurons");

    auto neuron_positions = neuron_to_subdomain_assignment->get_neuron_positions_in_subdomains();
    auto area_names = neuron_to_subdomain_assignment->get_neuron_area_names_in_subdomains();
    auto signal_types = neuron_to_subdomain_assignment->get_neuron_types_in_subdomains();

    RelearnException::check(neuron_positions.size() == number_local_neurons, "Simulation::initialize: neuron_positions had the wrong size");
    RelearnException::check(area_names.size() == number_local_neurons, "Simulation::initialize: area_names had the wrong size");
    RelearnException::check(signal_types.size() == number_local_neurons, "Simulation::initialize: signal_types had the wrong size");

    partition->print_my_subdomains_info_rank(-1);

    LogFiles::print_message_rank(0, "Neurons created");

    const auto& [simulation_box_min, simulation_box_max] = partition->get_simulation_box_size();
    const auto level_of_branch_nodes = partition->get_level_of_subdomain_trees();

    if (algorithm_enum == AlgorithmEnum::BarnesHut) {
        global_tree = std::make_shared<OctreeImplementation<BarnesHutCell>>(simulation_box_min, simulation_box_max, level_of_branch_nodes);
    } else if (algorithm_enum == AlgorithmEnum::BarnesHutInverted) {
        global_tree = std::make_shared<OctreeImplementation<BarnesHutInvertedCell>>(simulation_box_min, simulation_box_max, level_of_branch_nodes);
    } else if (algorithm_enum == AlgorithmEnum::BarnesHutLocationAware) {
        global_tree = std::make_shared<OctreeImplementation<BarnesHutCell>>(simulation_box_min, simulation_box_max, level_of_branch_nodes);
    } else if (algorithm_enum == AlgorithmEnum::FastMultipoleMethods) {
        global_tree = std::make_shared<OctreeImplementation<FastMultipoleMethodsCell>>(simulation_box_min, simulation_box_max, level_of_branch_nodes);
    } else if (algorithm_enum == AlgorithmEnum::Naive) {
        global_tree = std::make_shared<OctreeImplementation<NaiveCell>>(simulation_box_min, simulation_box_max, level_of_branch_nodes);
    } else {
        RelearnException::fail("Simulation::initialize: Cannot construct the octree for an unknown algorithm.");
    }

    LogFiles::print_message_rank(0, "Level of branch nodes is: {}", global_tree->get_level_of_branch_nodes());

    for (const auto& neuron_id : NeuronID::range(number_local_neurons)) {
        const auto& position = neuron_positions[neuron_id.get_neuron_id()];
        global_tree->insert(position, neuron_id);
    }

    global_tree->initializes_leaf_nodes(number_local_neurons);

    LogFiles::print_message_rank(0, "Inserted a total of {} neurons", number_total_neurons);

    if (algorithm_enum == AlgorithmEnum::BarnesHut) {
        auto cast = std::static_pointer_cast<OctreeImplementation<BarnesHutCell>>(global_tree);
        auto algorithm_barnes_hut = std::make_shared<BarnesHut>(std::move(cast));
        algorithm_barnes_hut->set_acceptance_criterion(accept_criterion);
        algorithm = std::move(algorithm_barnes_hut);
    } else if (algorithm_enum == AlgorithmEnum::BarnesHutInverted) {
        auto cast = std::static_pointer_cast<OctreeImplementation<BarnesHutInvertedCell>>(global_tree);
        auto algorithm_barnes_hut_inverted = std::make_shared<BarnesHutInverted>(std::move(cast));
        algorithm_barnes_hut_inverted->set_acceptance_criterion(accept_criterion);
        algorithm = std::move(algorithm_barnes_hut_inverted);
    } else if (algorithm_enum == AlgorithmEnum::BarnesHutLocationAware) {
        auto cast = std::static_pointer_cast<OctreeImplementation<BarnesHutCell>>(global_tree);
        auto algorithm_barnes_hut_location_aware = std::make_shared<BarnesHutLocationAware>(std::move(cast));
        algorithm_barnes_hut_location_aware->set_acceptance_criterion(accept_criterion);
        algorithm = std::move(algorithm_barnes_hut_location_aware);
    } else if (algorithm_enum == AlgorithmEnum::FastMultipoleMethods) {
        auto cast = std::static_pointer_cast<OctreeImplementation<FastMultipoleMethodsCell>>(global_tree);
        auto algorithm_barnes_hut = std::make_shared<FastMultipoleMethods>(std::move(cast));
        algorithm = std::move(algorithm_barnes_hut);
    } else {
        RelearnException::fail("Simulation::initialize: AlgorithmEnum {} not yet implemented!", static_cast<int>(algorithm_enum));
    }

    network_graph = std::make_shared<NetworkGraph>(number_local_neurons, my_rank);

    algorithm->set_synaptic_elements(axons, dendrites_ex, dendrites_in);

    neurons->set_area_names(std::move(area_names));
    neurons->set_signal_types(std::move(signal_types));
    neurons->set_positions(std::move(neuron_positions));

    neurons->set_network_graph(network_graph);
    neurons->set_octree(global_tree);
    neurons->set_algorithm(algorithm);

    for(const auto& area_name : neurons->get_extra_info()->get_unique_area_names()) {
        area_monitors->emplace_back(this, area_name, neurons->get_extra_info()->get_nr_neurons_in_area(area_name));
    }

    auto synapse_loader = neuron_to_subdomain_assignment->get_synapse_loader();

    auto [local_synapses, in_synapses, out_synapses] = synapse_loader->load_synapses();

    Timers::start(TimerRegion::INITIALIZE_NETWORK_GRAPH);
    network_graph->add_edges(local_synapses, in_synapses, out_synapses);
    Timers::stop_and_add(TimerRegion::INITIALIZE_NETWORK_GRAPH);

    LogFiles::print_message_rank(0, "Network graph created");
    LogFiles::print_message_rank(0, "Synaptic elements initialized");

    neurons->init_synaptic_elements();
    neurons->debug_check_counts();
    neurons->print_neurons_overview_to_log_file_on_rank_0(0);
    neurons->print_sums_of_synapses_and_elements_to_log_file_on_rank_0(0, 0, 0, 0);
}

void Simulation::simulate(const size_t number_steps) {
    RelearnException::check(number_steps > 0, "Simulation::simulate: number_steps must be greater than 0");
    const auto my_rank = MPIWrapper::get_my_rank();

    Timers::start(TimerRegion::SIMULATION_LOOP);

    const auto previous_synapse_creations = total_synapse_creations;
    const auto previous_synapse_deletions = total_synapse_deletions;

    /**
     * Simulation loop
     */
    const auto final_step_count = step + number_steps;
    for (; step <= final_step_count; ++step) { // NOLINT(altera-id-dependent-backward-branch)
        if (step % Config::monitor_step == 0) {
            const auto number_neurons = neurons->get_number_neurons();

            for (auto& mn : *monitors) {
                if (mn.get_target_id().get_neuron_id() < number_neurons) {
                    mn.record_data();
                }
            }

            neurons->get_neuron_model()->reset_fired_recorder();
        }

        if ( step % Config::monitor_area_step == 0) {
            for(auto& area_monitor : *area_monitors) {
                area_monitor.prepare_recording();
            }
            for(RelearnTypes::neuron_id neuron_id = 0; neuron_id < neurons->get_number_neurons(); neuron_id++) {
                for(auto& area_monitor : *area_monitors) {
                    area_monitor.record_data(NeuronID(neuron_id));
                }
            }
            for(auto& area_monitor : *area_monitors) {
                area_monitor.finish_recording();
            }
        }

        for (const auto& [disable_step, disable_ids] : disable_interrupts) {
            if (disable_step == step) {
                LogFiles::write_to_file(LogFiles::EventType::Cout, true, "Disabling {} neurons in step {}", disable_ids.size(), disable_step);
                const auto num_deleted_synapses = neurons->disable_neurons(disable_ids);
                total_synapse_deletions += static_cast<int64_t>(num_deleted_synapses);
            }
        }

        for (const auto& [enable_step, enable_ids] : enable_interrupts) {
            if (enable_step == step) {
                LogFiles::write_to_file(LogFiles::EventType::Cout, true, "Enabling {} neurons in step {}", enable_ids.size(), enable_step);
                neurons->enable_neurons(enable_ids);
            }
        }

        for (const auto& [creation_step, creation_count] : creation_interrupts) {
            if (creation_step == step) {
                LogFiles::write_to_file(LogFiles::EventType::Cout, true, "Creating {} neurons in step {}", creation_count, creation_step);
                neurons->create_neurons(creation_count);
            }
        }

        // Provide neuronal network to neuron models for one iteration step
        Timers::start(TimerRegion::UPDATE_ELECTRICAL_ACTIVITY);
        neurons->update_electrical_activity(step);
        Timers::stop_and_add(TimerRegion::UPDATE_ELECTRICAL_ACTIVITY);

        // Calc how many synaptic elements grow/retract
        // Apply the change in number of elements during connectivity update

        if (step >= Config::first_plasticity_update) {
            Timers::start(TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA);
            neurons->update_number_synaptic_elements_delta();
            Timers::stop_and_add(TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA);
        }

        if (step % Config::plasticity_update_step == 0) {
            Timers::start(TimerRegion::UPDATE_CONNECTIVITY);

            const auto& [num_axons_deleted, num_dendrites_deleted, num_synapses_created] = neurons->update_connectivity();

            Timers::stop_and_add(TimerRegion::UPDATE_CONNECTIVITY);

            // Get total number of synapses deleted and created
            const std::array<int64_t, 3> local_cnts = { static_cast<int64_t>(num_axons_deleted), static_cast<int64_t>(num_dendrites_deleted), static_cast<int64_t>(num_synapses_created) };
            const std::array<int64_t, 3> global_cnts = MPIWrapper::reduce(local_cnts, MPIWrapper::ReduceFunction::Sum, 0);

            const auto local_deletions = local_cnts[0] + local_cnts[1];
            const auto local_creations = local_cnts[2];

            const auto global_deletions = global_cnts[0] + global_cnts[1];
            const auto global_creations = global_cnts[2];

            if (0 == my_rank) {
                total_synapse_deletions += global_deletions;
                total_synapse_creations += global_creations;
            }

            LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdate, false, "{}: {} {} {}", step, global_creations, global_deletions, global_creations - global_deletions);
            LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdateCSV, false, "{};{};{};{}", step, global_creations, global_deletions, global_creations - global_deletions);
            LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdateLocal, false, "{}: {} {} {}", step, local_creations, local_deletions, local_creations - local_deletions);

            neurons->print_sums_of_synapses_and_elements_to_log_file_on_rank_0(step, num_axons_deleted, num_dendrites_deleted, num_synapses_created);

            network_graph->debug_check();
        }

        if (Config::logfile_update_step > 0 && step % Config::logfile_update_step == 0) {
            neurons->print_local_network_histogram(step);
        }

        if (Config::calcium_log_step > 0 && step % Config::calcium_log_step == 0) {
            neurons->print_calcium_values_to_file(step);
        }

        if (Config::synaptic_input_log_step > 0 && step % Config::synaptic_input_log_step == 0) {
            neurons->print_synaptic_inputs_to_file(step);
        }

        if (step % Config::statistics_step == 0) {
            neurons->print_neurons_overview_to_log_file_on_rank_0(step);

            for (auto& [attribute, vector] : statistics) {
                vector.emplace_back(neurons->get_statistics(attribute));
            }
        }

        if (step % Config::console_update_step == 0) {
            if (my_rank != 0) {
                continue;
            }

            const auto netto_creations = total_synapse_creations - total_synapse_deletions;

            LogFiles::write_to_file(LogFiles::EventType::Cout, true,
                "[Step: {}\t] Total up to now     (creations, deletions, netto):\t{}\t\t{}\t\t{}",
                step, total_synapse_creations, total_synapse_deletions, netto_creations);
        }
    }

    delta_synapse_creations = total_synapse_creations - previous_synapse_creations;
    delta_synapse_deletions = total_synapse_deletions - previous_synapse_deletions;

    // Stop timing simulation loop
    Timers::stop_and_add(TimerRegion::SIMULATION_LOOP);

    print_neuron_monitors();

    for(auto& area_monitor : *area_monitors) {
        std::string path = LogFiles::get_output_path() / (MPIWrapper::get_my_rank_str() + "_area_" + area_monitor.get_area_name() + ".csv");
        area_monitor.write_data_to_file(path);
    }

    neurons->print_positions_to_log_file();
    neurons->print_network_graph_to_log_file();
}

void Simulation::finalize() const {
    const auto netto_creations = total_synapse_creations - total_synapse_deletions;
    const auto previous_netto_creations = delta_synapse_creations - delta_synapse_deletions;

    LogFiles::print_message_rank(0,
        "Total up to now     (creations, deletions, netto): {}\t{}\t{}\nDiff. from previous (creations, deletions, netto): {}\t{}\t{}\nEND: {}",
        total_synapse_creations, total_synapse_deletions, netto_creations,
        delta_synapse_creations, delta_synapse_deletions, previous_netto_creations,
        Timers::wall_clock_time());

    neurons->print_calcium_statistics_to_essentials();

    LogFiles::write_to_file(LogFiles::EventType::Essentials, false,
        "Created synapses: {}\n"
        "Deleted synapses: {}\n"
        "Netto synapses: {}",
        total_synapse_creations,
        total_synapse_deletions,
        netto_creations);
}

std::vector<std::unique_ptr<NeuronModel>> Simulation::get_models() {
    return NeuronModel::get_models();
}

void Simulation::print_neuron_monitors() {
    const auto& path = LogFiles::get_output_path();

    for (auto& monitor : *monitors) {
        const auto& file_path = path / (MPIWrapper::get_my_rank_str() + '_' + std::to_string(monitor.get_target_id().get_neuron_id()) + ".csv");
        std::ofstream outfile(file_path, std::ios::trunc);

        const auto file_is_good = outfile.good();
        const auto file_is_bad = outfile.bad();

        RelearnException::check(file_is_good && !file_is_bad, "Simulation::print_neuron_monitors: The file is bad: {}", file_path);

        constexpr auto description = "Step;Fired;Fired Fraction;x;Secondary Variable;Calcium;Target Calcium;Synaptic Input;Background Activity;Grown Axons;Connected Axons;Grown Excitatory Dendrites;Connected Excitatory Dendrites;Grown Inhibitory Dendrites;Connected Inhibitory Dendrites\n";

        constexpr auto filler = ";";
        constexpr auto width = 6;

        outfile << std::setprecision(Constants::print_precision);
        outfile.imbue(std::locale());

        outfile << description;

        const auto& infos = monitor.get_informations();
        auto current_step = static_cast<decltype(Config::monitor_step)>(0);
        for (const auto& info : infos) {
            outfile << current_step << filler;
            outfile << info.get_fired() << filler;
            outfile << info.get_fraction_fired() << filler;
            outfile << info.get_x() << filler;
            outfile << info.get_secondary() << filler;
            outfile << info.get_calcium() << filler;
            outfile << info.get_target_calcium() << filler;
            outfile << info.get_synaptic_input() << filler;
            outfile << info.get_background_activity() << filler;
            outfile << info.get_axons() << filler;
            outfile << info.get_axons_connected() << filler;
            outfile << info.get_excitatory_dendrites_grown() << filler;
            outfile << info.get_excitatory_dendrites_connected() << filler;
            outfile << info.get_inhibitory_dendrites_grown() << filler;
            outfile << info.get_inhibitory_dendrites_connected() << "\n";

            current_step += Config::monitor_step;
        }

        outfile.flush();
        outfile.close();
    }
}

void Simulation::increase_monitoring_capacity(const size_t size) {
    for (auto& mon : *monitors) {
        mon.increase_monitoring_capacity(size);
    }
}

void Simulation::snapshot_monitors() {
    if (!monitors->empty()) {
        // record data at step 0
        for (auto& m : *monitors) {
            m.record_data();
        }

        neurons->get_neuron_model()->reset_fired_recorder();
    }
}

void Simulation::save_network_graph(size_t current_steps) {
    // Check wether there are multiple runs or not
    if (current_steps == 0) {
        neurons->print_network_graph_to_log_file();
    } else {
        LogFiles::save_and_open_new(LogFiles::EventType::Network, "network_" + std::to_string(current_steps));
        neurons->print_network_graph_to_log_file();
    }
}
