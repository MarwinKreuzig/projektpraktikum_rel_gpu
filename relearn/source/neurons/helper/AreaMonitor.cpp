#include "AreaMonitor.h"

#include "neurons/helper/RankNeuronId.h"
#include "neurons/Neurons.h"
#include "neurons/NetworkGraph.h"
#include "sim/Simulation.h"

#include <fstream>
#include <set>
#include <tuple>

AreaMonitor::AreaMonitor(Simulation* simulation, std::string area_name, RelearnTypes::number_neurons_type nr_neurons_in_area)
    : sim(simulation)
    , area_name(std::move(area_name))
    , nr_neurons_in_area(nr_neurons_in_area) {
}

void AreaMonitor::record_data(NeuronID neuron_id) {
    const auto out_edges = sim->get_network_graph()->get_all_out_edges(neuron_id);
    const auto in_edges = sim->get_network_graph()->get_all_in_edges(neuron_id);

    for (const auto& edge : out_edges) {
        const RankNeuronId rank_neuron_id = std::get<0>(edge);
        const NeuronID other_neuron_id = rank_neuron_id.get_neuron_id();
        const std::string other_area_name = sim->get_neurons()->get_extra_info()->get_area_name(other_neuron_id);
        const RelearnTypes::synapse_weight weight = std::get<1>(edge);

        connections[other_area_name].axons += 1;
    }

    for (const auto& edge : in_edges) {
        const RankNeuronId rank_neuron_id = std::get<0>(edge);
        const NeuronID other_neuron_id = rank_neuron_id.get_neuron_id();
        const std::string other_area_name = sim->get_neurons()->get_extra_info()->get_area_name(other_neuron_id);
        const RelearnTypes::synapse_weight weight = std::get<1>(edge);

        if (weight > 0)
            connections[other_area_name].den_ex += 1;
        else
            connections[other_area_name].den_inh += 1;
    }

    axons_grown += sim->get_neurons()->get_axons().get_grown_elements(neuron_id);
    den_ex_grown += sim->get_neurons()->get_dendrites_exc().get_grown_elements(neuron_id);
    den_inh_grown += sim->get_neurons()->get_dendrites_inh().get_grown_elements(neuron_id);

    axons_conn += sim->get_neurons()->get_axons().get_connected_elements(neuron_id);
    den_ex_conn += sim->get_neurons()->get_dendrites_exc().get_connected_elements(neuron_id);
    den_inh_conn += sim->get_neurons()->get_dendrites_inh().get_connected_elements(neuron_id);

    calcium += sim->get_neurons()->get_calcium(neuron_id);
}

void AreaMonitor::prepare_recording() {
    connections = EnsembleConnections();
    axons_conn = 0;
    axons_grown = 0;
    den_ex_grown = 0;
    den_ex_conn = 0;
    den_inh_conn = 0;
    den_inh_grown = 0;
    calcium = 0;
}

void AreaMonitor::finish_recording() {
    data.emplace_back(connections, axons_grown / nr_neurons_in_area, static_cast<double>(axons_conn) / nr_neurons_in_area, den_ex_grown / nr_neurons_in_area, static_cast<double>(den_ex_conn) / nr_neurons_in_area, den_inh_grown / nr_neurons_in_area, static_cast<double>(den_inh_conn) / nr_neurons_in_area, calcium / nr_neurons_in_area);
}

void AreaMonitor::write_data_to_file(std::filesystem::path file_path) {
    std::ofstream out(file_path);

    const std::set<std::string> area_names = sim->get_neurons()->get_extra_info()->get_unique_area_names();

    // Header
    out << "#Connections from ensemble " << area_name << " to ..." << std::endl;
    for (const auto& area : area_names) {
        out << "Ensemble " << area << " - Axons;"
            << "Ensemble " << area << " - Den ex;"
            << "Ensemble " << area << " - Den inh;";
    }
    out << "Step;Axons grown;Axons conn;Den ex grown;Den ex conn;Den inh grown;Den inh conn;Calcium;";
    out << std::endl;

    // Data
    size_t step = 0;
    for (const auto& tup : data) {
        out << step << ";";
        auto monitor_data = std::get<0>(tup);
        for (const auto& area : area_names) {
            const auto& connections = monitor_data[area];
            out << std::to_string(connections.axons) << ";";
            out << std::to_string(connections.den_ex) << ";";
            out << std::to_string(connections.den_inh) << ";";
        }
        out << std::to_string(std::get<1>(tup)) << ";";
        out << std::to_string(std::get<2>(tup)) << ";";
        out << std::to_string(std::get<3>(tup)) << ";";
        out << std::to_string(std::get<4>(tup)) << ";";
        out << std::to_string(std::get<5>(tup)) << ";";
        out << std::to_string(std::get<6>(tup)) << ";";
        out << std::to_string(std::get<7>(tup)) << ";";

        out << std::endl;
        step += Config::monitor_area_step;
    }
    out.close();
}
