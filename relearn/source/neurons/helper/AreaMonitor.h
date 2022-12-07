#pragma once

#include "Types.h"
#include "util/TaggedID.h"
#include "neurons/LocalAreaTranslator.h"
#include "neurons/SignalType.h"
#include "util/HashPair.h"

#include <boost/functional/hash.hpp>
#include <filesystem>
#include <map>
#include <string>
#include <tuple>
#include <vector>

class Simulation;

/**
 * Monitors the number of connections between areas and more area statistics
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

    RelearnTypes::area_name area_name;

    RelearnTypes::area_id area_id;


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
    double fired_fraction = 0.0;
    using EnsembleConnections = std::unordered_map<std::pair<int, RelearnTypes::area_id>, ConnectionCount, HashPair>;

    /**
     * For current logging step: Maps for each ensemble the number of connections
     */
    EnsembleConnections connections;

    /**
     * Complete data of all earlier logging steps
     */
    std::vector<std::tuple<EnsembleConnections, double, double, double, double, double, double, double, double>> data;

    std::vector<std::vector<AreaConnection>> mpi_data{};

public:
    /**
     * If a connected neuron is managed by another mpi rank. This area monitor cannot notify the other area about the connection to this area.
     * Hence, it creates a vector with information for each mpi rank that shall be sent to the other mpi ranks
     * @return Index i of the returned vector contains the information which shall be sent to mpi rank i. Each rank receives a vector of AreaConnections
     */
    [[nodiscard]] const std::vector<std::vector<AreaConnection>>& get_exchange_data() const;

    /**
     * Add an ingoing connection to the area. This method shall be called by other area monitors with ingoing connections to this area
     * @param connection Connection whose source is this area
     */
    void add_outgoing_connection(const AreaConnection& connection);

    /**
     * Construct an object for monitoring a specific area on this mpi rank
     * @param simulation Pointer to the simulation
     * @param area_id Id of the area that will be monitored
     * @param area_name Name of the area that will be monitored
     * @param my_rank The mpi rank of this process
     */
    AreaMonitor(Simulation* simulation, RelearnTypes::area_id area_id, RelearnTypes::area_name area_name, int my_rank);

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

    /**
     * Returns the name of the area that is monitored
     * @return Area name
     */
    [[nodiscard]] const RelearnTypes::area_name& get_area_name() const noexcept {
        return area_name;
    }

    /**
     * Returns the id of the area that is monitored
     * @return Area id
     */
    [[nodiscard]] const RelearnTypes::area_id& get_area_id() const noexcept {
        return area_id;
    }
};
