#include "AreaMonitor.h"

#include "mpi/MPIWrapper.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/Neurons.h"
#include "neurons/NetworkGraph.h"
#include "sim/Simulation.h"

#include <fstream>
#include <set>
#include <tuple>

AreaMonitor::AreaMonitor(Simulation* simulation, RelearnTypes::area_id area_id, RelearnTypes::area_name area_name, RelearnTypes::number_neurons_type nr_neurons_in_area, int my_rank)
    : sim(simulation)
    , area_id(area_id)
    , area_name(std::move(area_name))
    , nr_neurons_in_area(nr_neurons_in_area)
    , my_rank(my_rank) {
}

void AreaMonitor::record_data(NeuronID neuron_id) {
    const auto out_edges = sim->get_network_graph()->get_all_in_edges(neuron_id);

    for (const auto& [rank_neuron_id, weight] : out_edges) {
        if (rank_neuron_id.get_rank() == my_rank) {
            const auto& other_neuron_id = rank_neuron_id.get_neuron_id();
            const auto other_area_id = sim->get_neurons()->get_extra_info()->get_area_id_for_neuron_id(other_neuron_id);
            AreaMonitor& other_area_monitor = sim->get_area_monitors()->at(other_area_id);
            const auto signal_type = weight > 0 ? SignalType::Excitatory : SignalType::Inhibitory;
            other_area_monitor.add_connection({ my_rank, area_id, other_neuron_id, signal_type });
        } else {
            const NeuronID other_neuron_id = rank_neuron_id.get_neuron_id();
            const SignalType signal_type = (weight > 0) ? SignalType::Excitatory : SignalType::Inhibitory;
            RelearnException::check(mpi_data.size() == MPIWrapper::get_num_ranks(), "AreaMonitor::record_data: mpi_data has wrong size {} != {}", mpi_data.size(), MPIWrapper::get_num_ranks());
            mpi_data[rank_neuron_id.get_rank()].emplace_back(AreaConnection(my_rank, area_id, other_neuron_id, signal_type));
        }
    }

    axons_grown += sim->get_neurons()->get_axons().get_grown_elements(neuron_id);
    den_ex_grown += sim->get_neurons()->get_dendrites_exc().get_grown_elements(neuron_id);
    den_inh_grown += sim->get_neurons()->get_dendrites_inh().get_grown_elements(neuron_id);

    axons_conn += sim->get_neurons()->get_axons().get_connected_elements(neuron_id);
    den_ex_conn += sim->get_neurons()->get_dendrites_exc().get_connected_elements(neuron_id);
    den_inh_conn += sim->get_neurons()->get_dendrites_inh().get_connected_elements(neuron_id);

    calcium += sim->get_neurons()->get_calcium(neuron_id);
    fired_fraction += static_cast<double>(sim->get_neurons()->get_neuron_model()->fired_recorder[NeuronModel::FireRecorderPeriod::AREA_MONITOR][neuron_id.get_neuron_id()]) / static_cast<double>(Config::monitor_area_step);
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
    fired_fraction = 0;
    mpi_data.clear();
    mpi_data.resize(MPIWrapper::get_num_ranks(), {});
}

void AreaMonitor::finish_recording() {
    data.emplace_back(connections, axons_grown, static_cast<double>(axons_conn), den_ex_grown, static_cast<double>(den_ex_conn), den_inh_grown, static_cast<double>(den_inh_conn), calcium, fired_fraction);
    mpi_data.clear();
}

void AreaMonitor::write_data_to_file(const std::filesystem::path& file_path) {
    std::ofstream out(file_path);

    std::set<std::pair<int, RelearnTypes::area_id>> unique_area_ids;
    for (const auto& single_record : data) {
        auto connection_data = std::get<0>(single_record);
        for (const auto& [rank_area_id, _] : connection_data) {
            unique_area_ids.insert(rank_area_id);
        }
    }

    std::vector<std::pair<int, RelearnTypes::area_id>> unique_area_ids_list;
    std::copy(unique_area_ids.begin(), unique_area_ids.end(), std::back_inserter(unique_area_ids_list));
    std::sort(unique_area_ids_list.begin(), unique_area_ids_list.end()); /*, [](const auto& p1,const auto &p2) -> {
        p1.first < p2
    });*/
    // Header
    out << "# Connections from ensemble " << area_name << " (" << my_rank << ":" << area_id << ") to ..." << std::endl;
    out << "# Rank: " << my_rank << std::endl;
    out << "# Area id: " << area_id << std::endl;
    out << "# Area name: " << area_name << std::endl;
    out << "Step;";
    for (const auto& [rank, area_id] : unique_area_ids_list) {
        out << rank << ":" << area_id << "ex;"
            << rank << ":" << area_id << "in;";
    }
    out << "Axons grown;Axons conn;Den ex grown;Den ex conn;Den inh grown;Den inh conn;Calcium;Fire rate;";
    out << std::endl;

    // Data
    size_t step = 0;
    for (const auto& single_record : data) {
        out << step << ";";
        auto connection_data = std::get<0>(single_record);
        for (const auto& rank_area_id : unique_area_ids_list) {
            const auto& connections = connection_data[rank_area_id];
            out << std::to_string(connections.den_ex) << ";";
            out << std::to_string(connections.den_inh) << ";";
        }
        out << std::to_string(std::get<1>(single_record)) << ";";
        out << std::to_string(std::get<2>(single_record)) << ";";
        out << std::to_string(std::get<3>(single_record)) << ";";
        out << std::to_string(std::get<4>(single_record)) << ";";
        out << std::to_string(std::get<5>(single_record)) << ";";
        out << std::to_string(std::get<6>(single_record)) << ";";
        out << std::to_string(std::get<7>(single_record)) << ";";
        out << std::to_string(std::get<8>(single_record)) << ";";

        out << std::endl;
        step += Config::monitor_area_step;
    }
    out.close();
}
const std::vector<std::vector<AreaMonitor::AreaConnection>>& AreaMonitor::get_exchange_data() const {
    return mpi_data;
}

void AreaMonitor::add_connection(const AreaMonitor::AreaConnection& connection) {
    auto pair = std::make_pair(connection.from_rank, connection.from_area);
    if (connections.contains(pair)) {
        auto& conn = connections.at(pair);
        if (connection.signal_type == SignalType::Excitatory) {
            conn.den_ex += 1;
        } else {
            conn.den_inh += 1;
        }
    } else {
        auto conn = ConnectionCount();
        if (connection.signal_type == SignalType::Excitatory) {
            conn.den_ex += 1;
        } else {
            conn.den_inh += 1;
        }
        connections.insert(std::make_pair(pair, conn));
    }
}