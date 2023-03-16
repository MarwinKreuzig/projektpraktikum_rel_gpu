/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "AreaMonitor.h"

#include "mpi/MPIWrapper.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/LocalAreaTranslator.h"
#include "neurons/Neurons.h"
#include "neurons/NetworkGraph.h"
#include "sim/Simulation.h"

#include <fstream>
#include <set>
#include <tuple>
#include <utility>

AreaMonitor::AreaMonitor(Simulation *simulation, std::shared_ptr<GlobalAreaMapper> global_area_mapper, RelearnTypes::area_id area_id, RelearnTypes::area_name area_name,
                         int my_rank, std::filesystem::path &path)
        : sim(simulation), area_id(area_id), area_name(std::move(area_name)), my_rank(my_rank), path(std::move(path)), global_area_mapper(std::move(global_area_mapper)) {
    write_header();
}

 void AreaMonitor::monitor_connectivity() {
     Timers::start(TimerRegion::AREA_MONITORS_LOCAL_EDGES);
     for(const auto& synapse : sim->get_neurons()->last_created_local_synapses) {
         if(sim->get_neurons()->get_local_area_translator()->get_area_id_for_neuron_id(synapse.get_target().get_neuron_id()) != area_id) {
             continue;
         }
         const auto other_area_id = sim->get_neurons()->get_local_area_translator()->get_area_id_for_neuron_id(
                 synapse.get_source().get_neuron_id());

         const auto signal_type = synapse.get_weight() > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
         add_ingoing_connection({my_rank,other_area_id, synapse.get_target(), signal_type});
     }

     Timers::stop_and_add(TimerRegion::AREA_MONITORS_LOCAL_EDGES);
     Timers::start(TimerRegion::AREA_MONITORS_DISTANT_EDGES);
     for(const auto& synapse : sim->get_neurons()->last_created_in_synapses) {
         if(sim->get_neurons()->get_local_area_translator()->get_area_id_for_neuron_id(synapse.get_target().get_neuron_id()) != area_id) {
             continue;
         }
         // Other area is on different mpi rank. Save connection for communication over mpi
         const auto& rank_neuron_id = synapse.get_source();
         const auto& other_rank = rank_neuron_id.get_rank().get_rank();
         const auto other_area_id = global_area_mapper->get_area_id(rank_neuron_id);

         const auto signal_type = synapse.get_weight() > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
         add_ingoing_connection({other_rank,other_area_id, synapse.get_target(), signal_type});
     }
     Timers::stop_and_add(TimerRegion::AREA_MONITORS_DISTANT_EDGES);
 }

void AreaMonitor::record_data(NeuronID neuron_id) {
    Timers::start(TimerRegion::AREA_MONITORS_DELETIONS);

    //Deletions
    const auto& deletions_in_step = sim->get_neurons()->deletions_log[neuron_id.get_neuron_id()];
    for(const auto& [other_neuron_id, weight] : deletions_in_step) {
        const auto other_area_id = global_area_mapper->get_area_id(other_neuron_id);
        const auto& other_rank = other_neuron_id.get_rank().get_rank();

        auto pair = std::make_pair(other_rank, other_area_id);
        deletions[pair]++;
        const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
        remove_ingoing_connection(AreaConnection(other_rank, other_area_id, neuron_id, signal_type));
    }

    Timers::stop_and_add(TimerRegion::AREA_MONITORS_DELETIONS);
    Timers::start(TimerRegion::AREA_MONITORS_STATISTICS);

    // Store statistics
    axons_grown += sim->get_neurons()->get_axons().get_grown_elements(neuron_id);
    den_ex_grown += sim->get_neurons()->get_dendrites_exc().get_grown_elements(neuron_id);
    den_inh_grown += sim->get_neurons()->get_dendrites_inh().get_grown_elements(neuron_id);

    axons_conn += sim->get_neurons()->get_axons().get_connected_elements(neuron_id);
    den_ex_conn += sim->get_neurons()->get_dendrites_exc().get_connected_elements(neuron_id);
    den_inh_conn += sim->get_neurons()->get_dendrites_inh().get_connected_elements(neuron_id);

    calcium += sim->get_neurons()->get_calcium(neuron_id);
    fired_fraction +=
            static_cast<double>(sim->get_neurons()->get_neuron_model()->fired_recorder[NeuronModel::FireRecorderPeriod::AreaMonitor][neuron_id.get_neuron_id()]) /
            static_cast<double>(Config::plasticity_update_step);
    num_enabled_neurons++;

    Timers::stop_and_add(TimerRegion::AREA_MONITORS_STATISTICS);
}

void AreaMonitor::prepare_recording() {
    deletions = EnsembleDeletions{};
    axons_conn = 0;
    axons_grown = 0;
    den_ex_grown = 0;
    den_ex_conn = 0;
    den_inh_conn = 0;
    den_inh_grown = 0;
    calcium = 0;
    fired_fraction = 0;
    num_enabled_neurons = 0;
    const auto num_areas = sim->get_neurons()->get_local_area_translator()->get_number_of_areas();
}

void AreaMonitor::finish_recording() {
    data.emplace_back(connections, deletions, axons_grown, static_cast<double>(axons_conn), den_ex_grown,
                      static_cast<double>(den_ex_conn), den_inh_grown, static_cast<double>(den_inh_conn), calcium,
                      fired_fraction, num_enabled_neurons);
}

void AreaMonitor::write_header() {
    std::ofstream out(path, std::ios_base::app);

    // Header
    out << "# Connections from ensemble " << area_name << " (" << my_rank << ":" << area_id << ") to ..."
        << "\n";
    out << "# Rank: " << my_rank << "\n";
    out << "# Area id: " << area_id << "\n";
    out << "# Area name: " << area_name << "\n";
    out.close();
}

void AreaMonitor::write_data_to_file() {
    //Timers::start();
    std::ofstream out(path, std::ios_base::app);

    std::set<std::pair<int, RelearnTypes::area_id>> unique_area_ids;
    for (const auto &single_record: data) {
        auto connection_data = std::get<0>(single_record);
        auto deletion_data = std::get<1>(single_record);

        for (const auto &[rank_area_id, _]: connection_data) {
            unique_area_ids.insert(rank_area_id);
        }
        for (const auto &[rank_area_id, _]: deletion_data) {
            unique_area_ids.insert(rank_area_id);
        }
    }

    std::vector<std::pair<int, RelearnTypes::area_id>> unique_area_ids_list;
    std::copy(unique_area_ids.begin(), unique_area_ids.end(), std::back_inserter(unique_area_ids_list));
    std::sort(unique_area_ids_list.begin(), unique_area_ids_list.end());
    // Header
    out << "# Step;";
    for (const auto &[rank, area_id]: unique_area_ids_list) {
        out << rank << ":" << area_id << "ex;"
            << rank << ":" << area_id << "in;"
            << rank << ":" << area_id << "del;";
    }
    out << "Axons grown;Axons conn;Den ex grown;Den ex conn;Den inh grown;Den inh conn;Calcium;Fire rate;Enabled neurons;";
    out << "\n";

    // Data
    for (const auto &single_record: data) {
        out << step << ";";
        auto connection_data = std::get<0>(single_record);
        auto deletion_data = std::get<1>(single_record);
        for (const auto &rank_area_id: unique_area_ids_list) {
            const auto &connections = connection_data[rank_area_id];
            out << std::to_string(connections.den_ex) << ";";
            out << std::to_string(connections.den_inh) << ";";
            const auto &deletions_in_step = deletion_data[rank_area_id];
            out << std::to_string(deletions_in_step) << ";";
        }
        out << std::to_string(std::get<2>(single_record)) << ";";
        out << std::to_string(std::get<3>(single_record)) << ";";
        out << std::to_string(std::get<4>(single_record)) << ";";
        out << std::to_string(std::get<5>(single_record)) << ";";
        out << std::to_string(std::get<6>(single_record)) << ";";
        out << std::to_string(std::get<7>(single_record)) << ";";
        out << std::to_string(std::get<8>(single_record)) << ";";
        out << std::to_string(std::get<9>(single_record)) << ";";
        out << std::to_string(std::get<10>(single_record)) << ";";

        out << "\n";
        step += Config::plasticity_update_step;
    }
    out.close();
    data.clear();
    //Timers::stop_and_add();
}

void AreaMonitor::request_data(const NeuronID& neuron_id) const {
    //Deletions
    const auto& deletions = sim->get_neurons()->deletions_log[neuron_id.get_neuron_id()];
    for(const auto& [other_neuron_id, _] : deletions) {
        global_area_mapper->request_area_id(other_neuron_id);
    }
}

void AreaMonitor::request_data() const {
    Timers::start(TimerRegion::AREA_MONITORS_DISTANT_EDGES);
    for(const auto& synapse : sim->get_neurons()->last_created_in_synapses) {
        if(sim->get_neurons()->get_local_area_translator()->get_area_id_for_neuron_id(synapse.get_target().get_neuron_id()) != area_id) {
            continue;
        }
        // Other area is on different mpi rank. Save connection for communication over mpi
        const auto& rank_neuron_id = synapse.get_source();
        global_area_mapper->request_area_id(rank_neuron_id);
    }
    Timers::stop_and_add(TimerRegion::AREA_MONITORS_DISTANT_EDGES);
}

void AreaMonitor::add_ingoing_connection(const AreaMonitor::AreaConnection &connection) {
    auto pair = std::make_pair(connection.from_rank, connection.from_area);
    auto &conn = connections[pair];
    if (connection.signal_type == SignalType::Excitatory) {
        conn.den_ex += 1;
    } else {
        conn.den_inh += 1;
    }
}

void AreaMonitor::remove_ingoing_connection(const AreaMonitor::AreaConnection &connection) {
    auto pair = std::make_pair(connection.from_rank, connection.from_area);
    auto &conn = connections[pair];
    if (connection.signal_type == SignalType::Excitatory) {
        conn.den_ex -= 1;
    } else {
        conn.den_inh -= 1;
    }
}