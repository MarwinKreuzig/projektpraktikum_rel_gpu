#pragma once

#include "Types.h"
#include "util/TaggedID.h"
#include "neurons/SignalType.h"

#include <filesystem>
#include <map>
#include <string>
#include <tuple>
#include <vector>

class Simulation;

/**
 * Monitors the number of connections between ensembles
 */
class AreaMonitor {
public:
    struct AreaConnection {
    public:
        AreaConnection(int from_rank, RelearnTypes::area_id from_area, const NeuronID to_local_neuron_id, const SignalType signal_type)
            : from_rank(from_rank)
            , from_area(from_area)
            , to_local_neuron_id(to_local_neuron_id)
            , signal_type(signal_type) { }
        AreaConnection() { }
        int from_rank;
        RelearnTypes::area_id from_area;
        NeuronID to_local_neuron_id;
        SignalType signal_type;
    };

private:
    Simulation* sim;

    int my_rank;

    /**
     * Id of the ensemble which is monitored by this instance
     */
    RelearnTypes::area_name area_name;

    RelearnTypes::area_id area_id;

    unsigned int nr_neurons_in_area;

    /**
     * Number of connections to another ensemble in a single step
     */
    struct ConnectionCount {
        int den_ex = 0;
        int den_inh = 0;
    };

    double axons_grown = 0;
    double den_ex_grown = 0;
    double den_inh_grown = 0;
    int axons_conn = 0;
    int den_ex_conn = 0;
    int den_inh_conn = 0;
    double calcium = 0;
    using EnsembleConnections = std::map<std::pair<int, RelearnTypes::area_id>, ConnectionCount>;

    /**
     * For current logging step: Maps for each ensemble the number of connections
     */
    EnsembleConnections connections;

    /**
     * Complete data of all earlier logging steps
     */
    std::vector<std::tuple<EnsembleConnections, double, double, double, double, double, double, double>> data;

    std::vector<std::vector<AreaConnection>> mpi_data{};

public:
    [[nodiscard]] const std::vector<std::vector<AreaConnection>>& get_exchange_data() const;

    void add_connection(const AreaConnection& connection);

    /**
     *
     * @param simulation
     * @param ensembleId Id of the current ensemble
     */
    AreaMonitor(Simulation* simulation, RelearnTypes::area_id area_id, RelearnTypes::area_name area_name, RelearnTypes::number_neurons_type nr_neurons_in_area, int my_rank);

    /**
     * Prepares the monitor for a new logging step. Call this method before each logging step.
     */
    void prepare_recording();

    /**
     *
     * Add the data of a single neuron to the recording. The neuron must be part of the ensemble.
     * Call this method with each neuron of the ensemble in each logging step
     * @param neuron_id Neuron which is part of the ensemble
     */
    void record_data(NeuronID neuron_id);

    /**
     * Indicates end of a single logging step. Call this method after the data off each neuron was recorded.
     */
    void finish_recording();

    /**
     * Write all recorded data to a csv file
     * @param file_path Path to new csv file
     */
    void write_data_to_file(const std::filesystem::path& file_path);

    [[nodiscard]] const RelearnTypes::area_name& get_area_name() const noexcept {
        return area_name;
    }
    [[nodiscard]] const RelearnTypes::area_id& get_area_id() const noexcept {
        return area_id;
    }
};
