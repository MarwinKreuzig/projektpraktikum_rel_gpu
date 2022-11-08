#pragma once

#include <vector>
#include "../../sim/Simulation.h"
#include "../NetworkGraph.h"
#include "../Neurons.h"



class Simulation;
class NetworkGraph;
class Neurons;

/**
 * Monitors the number of connections between ensembles
 */
class AreaMonitor {
private:
    Simulation * sim;

    /**
     * Id of the ensemble which is monitored by this instance
     */
    std::string area_name;

    /**
     * Number of connections to another ensemble in a single step
     */
    struct ConnectionCount {
        int axons = 0;
        int den_ex =0;
        int den_inh =0;
    };

    double axons_grown =0;
    double den_ex_grown =0;
    double den_inh_grown =0;
    int axons_conn = 0;
    int den_ex_conn = 0;
    int den_inh_conn = 0;
    double calcium = 0;
    using EnsembleConnections = std::map<std::string ,ConnectionCount>;

    unsigned int nr_neurons_in_area;

    /**
     * For current logging step: Maps for each ensemble the number of connections
     */
    EnsembleConnections connections;

    /**
     * Complete data of all earlier logging steps
     */
    std::vector<std::tuple<EnsembleConnections,double,double,double,double,double,double,double>> data;


public:
    /**
     *
     * @param simulation
     * @param ensembleId Id of the current ensemble
     */
    AreaMonitor(Simulation * simulation, std::string area_name, unsigned int nr_neurons_in_area) ;

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
    void write_data_to_file(std::string file_path);

    [[nodiscard]] const std::string & get_area_name() const noexcept {
        return area_name;
    }



};
