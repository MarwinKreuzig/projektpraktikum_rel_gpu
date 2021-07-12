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

#include "../algorithm/BarnesHut.h"
#include "../Config.h"
#include "../io/LogFiles.h"
#include "../mpi/MPIWrapper.h"
#include "../neurons/NetworkGraph.h"
#include "../neurons/Neurons.h"
#include "../neurons/helper/NeuronMonitor.h"
#include "../neurons/models/NeuronModels.h"
#include "../structure/Octree.h"
#include "../structure/Partition.h"
#include "../util/RelearnException.h"
#include "../util/Timers.h"
#include "NeuronToSubdomainAssignment.h"
#include "SubdomainFromFile.h"
#include "SubdomainFromNeuronDensity.h"

#include <fstream>
#include <iomanip>
#include <sstream>

Simulation::Simulation(std::shared_ptr<Partition> partition)
    : partition(std::move(partition)) {
}

void Simulation::register_neuron_monitor(size_t neuron_id) {
    monitors.emplace_back(neuron_id);
}

void Simulation::set_acceptance_criterion_for_octree(double value) {
    // Needed to avoid creating autapses
    if (value > BarnesHut::max_theta) {
        RelearnException::fail("Acceptance criterion must be smaller or equal to 0.5");
    }

    accept_criterion = value;
}

void Simulation::set_neuron_models(std::unique_ptr<NeuronModels> nm) {
    neuron_models = std::move(nm);
}

void Simulation::set_axons(std::unique_ptr<SynapticElements> se) {
    axons = std::move(se);
}

void Simulation::set_dendrites_ex(std::unique_ptr<SynapticElements> se) {
    dendrites_ex = std::move(se);
}

void Simulation::set_dendrites_in(std::unique_ptr<SynapticElements> se) {
    dendrites_in = std::move(se);
}

void Simulation::set_enable_interrupts(std::vector<std::pair<size_t, std::vector<size_t>>> interrupts) {
    enable_interrupts = std::move(interrupts);

    for (auto& [step, ids] : enable_interrupts) {
        std::sort(ids.begin(), ids.end());
    }
}

void Simulation::set_disable_interrupts(std::vector<std::pair<size_t, std::vector<size_t>>> interrupts) {
    disable_interrupts = std::move(interrupts);

    for (auto& [step, ids] : disable_interrupts) {
        std::sort(ids.begin(), ids.end());
    }
}

void Simulation::set_creation_interrupts(std::vector<std::pair<size_t, size_t>> interrupts) {
    creation_interrupts = std::move(interrupts);
}

void Simulation::construct_neurons() {
    RelearnException::check(neuron_models != nullptr, "In simulation, neuron_models is nullptr");
    RelearnException::check(axons != nullptr, "In simulation, axons is nullptr");
    RelearnException::check(dendrites_ex != nullptr, "In simulation, dendrites_ex is nullptr");
    RelearnException::check(dendrites_in != nullptr, "In simulation, dendrites_in is nullptr");

    neurons = std::make_shared<Neurons>(partition, neuron_models->clone(), axons->clone(), dendrites_ex->clone(), dendrites_in->clone());
}

void Simulation::place_random_neurons(size_t num_neurons, double frac_exc) {
    neuron_to_subdomain_assignment = std::make_unique<SubdomainFromNeuronDensity>(num_neurons, frac_exc, SubdomainFromNeuronDensity::default_um_per_neuron);
    partition->set_total_num_neurons(num_neurons);
    initialize();
}

void Simulation::load_neurons_from_file(const std::string& path_to_positions, const std::optional<std::string>& optional_path_to_connections) {
    auto local_ptr = std::make_unique<SubdomainFromFile>(path_to_positions);
    partition->set_total_num_neurons(local_ptr->get_total_num_neurons_in_file());
    neuron_to_subdomain_assignment = std::move(local_ptr);
    initialize();

    if (optional_path_to_connections.has_value()) {
        const auto& path_to_connections = optional_path_to_connections.value();

        network_graph->add_edges_from_file(path_to_connections, path_to_positions, *partition);
        LogFiles::print_message_rank(0, "Network graph created");

        neurons->init_synaptic_elements();
        neurons->debug_check_counts();
        LogFiles::print_message_rank(0, "Synaptic elements initialized \n");
    }

    neurons->print_neurons_overview_to_log_file_on_rank_0(0);
    neurons->print_sums_of_synapses_and_elements_to_log_file_on_rank_0(0, 0, 0);
}

void Simulation::initialize() {
    construct_neurons();

    partition->load_data_from_subdomain_assignment(neurons, std::move(neuron_to_subdomain_assignment));

    NeuronMonitor::neurons_to_monitor = neurons;

    partition->print_my_subdomains_info_rank(0);
    partition->print_my_subdomains_info_rank(1);

    LogFiles::print_message_rank(0, "Neurons created");

    auto sim_box_min_max = partition->get_simulation_box_size();

    global_tree = std::make_shared<Octree>(std::move(std::get<0>(sim_box_min_max)), std::move(std::get<1>(sim_box_min_max)), partition->get_level_of_subdomain_trees());
    global_tree->set_no_free_in_destructor(); // This needs to be changed later, as it's cleaner to free the nodes at destruction

    // Insert my local (subdomain) trees into my global tree
    for (size_t i = 0; i < partition->get_my_num_subdomains(); i++) {
        size_t index_1d = partition->get_1d_index_for_local_subdomain(i);
        OctreeNode* local_tree = partition->get_subdomain_tree(i);
        global_tree->insert_local_tree(local_tree, index_1d);
        partition->delete_subdomain_tree(i);
    }

    LogFiles::print_message_rank(0, "Neurons inserted into subdomains");
    LogFiles::print_message_rank(0, "Subdomains inserted into global tree");

    network_graph = std::make_shared<NetworkGraph>(neurons->get_num_neurons());

    barnes_hut_algorithm = std::make_shared<BarnesHut>(global_tree);
    barnes_hut_algorithm->set_acceptance_criterion(accept_criterion);

    neurons->set_network_graph(network_graph);
    neurons->set_octree(global_tree);
    neurons->set_barnes_hut(barnes_hut_algorithm);
}

void Simulation::simulate(size_t number_steps, size_t step_monitor) {
    GlobalTimers::timers.start(TimerRegion::SIMULATION_LOOP);

    const auto previous_synapse_creations = total_synapse_creations;
    const auto previous_synapse_deletions = total_synapse_deletions;

    /**
	* Simulation loop
	*/
    for (size_t step = 1; step <= number_steps; step++) {
        if (step % step_monitor == 0) {
            for (auto& mn : monitors) {
                mn.record_data();
            }

            NeuronMonitor::current_step++;
        }

        for (const auto& [disable_step, disable_ids] : disable_interrupts) {
            if (disable_step == step) {
                LogFiles::write_to_file(LogFiles::EventType::Cout, true, "Disabling {} neurons in step {}", disable_ids.size(), disable_step);
                const auto num_deleted_synapses = neurons->disable_neurons(disable_ids);
                total_synapse_deletions += num_deleted_synapses;
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
        GlobalTimers::timers.start(TimerRegion::UPDATE_ELECTRICAL_ACTIVITY);
        neurons->update_electrical_activity();
        GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_ELECTRICAL_ACTIVITY);

        // Calc how many synaptic elements grow/retract
        // Apply the change in number of elements during connectivity update
        GlobalTimers::timers.start(TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA);
        neurons->update_number_synaptic_elements_delta();
        GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA);

        if (step % Constants::plasticity_update_step == 0) {
            size_t num_synapses_deleted = 0;
            size_t num_synapses_created = 0;

            //if (0 == MPIWrapper::get_my_rank()) {
            //    std::stringstream sstring; // For output generation
            //    sstring << "** UPDATE CONNECTIVITY AFTER: " << step << " of " << number_steps
            //            << " msec ** [" << Timers::wall_clock_time() << "]\n";
            //    sstring << std::flush;
            //    LogFiles::write_to_file(LogFiles::EventType::Cout, sstring.str(), true);
            //}

            GlobalTimers::timers.start(TimerRegion::UPDATE_CONNECTIVITY);

            std::tuple<size_t, size_t> deleted_created = neurons->update_connectivity();
            num_synapses_deleted = std::get<0>(deleted_created);
            num_synapses_created = std::get<1>(deleted_created);

            GlobalTimers::timers.stop_and_add(TimerRegion::UPDATE_CONNECTIVITY);

            // Get total number of synapses deleted and created
            std::array<int64_t, 2> local_cnts = { static_cast<int64_t>(num_synapses_deleted), static_cast<int64_t>(num_synapses_created) };
            std::array<int64_t, 2> global_cnts = MPIWrapper::reduce(local_cnts, MPIWrapper::ReduceFunction::sum, 0, MPIWrapper::Scope::global);

            if (0 == MPIWrapper::get_my_rank()) {
                total_synapse_deletions += global_cnts[0] / 2;
                total_synapse_creations += global_cnts[1] / 2;
            }

            LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdate, false, "{}: {} {} {}", step, global_cnts[1], global_cnts[0], global_cnts[1] + global_cnts[0]);
            LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdateLocal, false, "{}: {} {} {}", step, local_cnts[1], local_cnts[0], local_cnts[1] + local_cnts[0]);

            neurons->print_sums_of_synapses_and_elements_to_log_file_on_rank_0(step, num_synapses_deleted, num_synapses_created);

            network_graph->debug_check();
        }

        if (step % Constants::logfile_update_step == 0) {
            neurons->print_neurons_overview_to_log_file_on_rank_0(step);
        }

        if (step % Constants::console_update_step == 0) {
            if (MPIWrapper::get_my_rank() != 0) {
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
    GlobalTimers::timers.stop_and_add(TimerRegion::SIMULATION_LOOP);

    print_neuron_monitors();

    neurons->print_positions_to_log_file();
    neurons->print_network_graph_to_log_file();
}

void Simulation::finalize() const {
    if (0 == MPIWrapper::get_my_rank()) {
        const auto netto_creations = total_synapse_creations - total_synapse_deletions;
        const auto previous_netto_creations = delta_synapse_creations - delta_synapse_deletions;

        LogFiles::print_message_rank(0,
            "Total up to now     (creations, deletions, netto): {}\t{}\t{}\nDiff. from previous (creations, deletions, netto): {}\t{}\t{}\nEND: {}",
            total_synapse_creations, total_synapse_deletions, netto_creations,
            delta_synapse_creations, delta_synapse_deletions, previous_netto_creations,
            Timers::wall_clock_time());

        LogFiles::write_to_file(LogFiles::EventType::Essentials, false,
            "Created synapses: {}\n"
            "Deleted synapses: {}\n"
            "Netto synapses: {}",
            total_synapse_creations,
            total_synapse_deletions,
            netto_creations);

        neurons->print_statistics_to_essentials();
    }
}

std::vector<std::unique_ptr<NeuronModels>> Simulation::get_models() {
    return NeuronModels::get_models();
}

void Simulation::print_neuron_monitors() {
    for (auto& monitor : monitors) {
        std::ofstream outfile(std::to_string(monitor.get_target_id()) + ".csv", std::ios::trunc);
        outfile << std::setprecision(Constants::print_precision);

        outfile.imbue(std::locale());

        outfile << "Step;Fired;Refrac;x;Ca;I_sync;axons;axons_connected;dendrites_exc;dendrites_exc_connected;dendrites_inh;dendrites_inh_connected";
        outfile << "\n";

        const auto& infos = monitor.get_informations();

        const char* const filler = ";";
        const auto width = 6;

        auto ctr = 0;

        for (const auto& info : infos) {
            outfile << ctr << filler;
            outfile << info.get_fired() << filler;
            outfile << info.get_secondary() << filler;
            outfile << info.get_x() << filler;
            outfile << info.get_calcium() << filler;
            outfile << info.get_I_sync() << filler;
            outfile << info.get_axons() << filler;
            outfile << info.get_axons_connected() << filler;
            outfile << info.get_dendrites_exc() << filler;
            outfile << info.get_dendrites_exc_connected() << filler;
            outfile << info.get_dendrites_inh() << filler;
            outfile << info.get_dendrites_inh_connected() << "\n";

            ctr++;
        }

        outfile.flush();
        outfile.close();
    }
}

void Simulation::increase_monitoring_capacity(size_t size) {
    for (auto& mon : monitors) {
        mon.increase_monitoring_capacity(size);
    }
}
