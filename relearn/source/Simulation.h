#ifndef SIMULATION_H
#define SIMULATION_H

#include "Commons.h"
#include "LogFiles.h"
#include "LogMessages.h"
#include "MPIWrapper.h"
#include "MPI_RMA_MemAllocator.h"
#include "NetworkGraph.h"
#include "NeuronIdMap.h"
#include "NeuronModels.h"
#include "NeuronMonitor.h"
#include "NeuronToSubdomainAssignment.h"
#include "Neurons.h"
#include "Octree.h"
#include "Parameters.h"
#include "Partition.h"
#include "RelearnException.h"
#include "SubdomainFromFile.h"
#include "SubdomainFromNeuronDensity.h"
#include "SynapticElements.h"
#include "Timers.h"
#include "Utility.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <array>
#include <bitset>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>

#include <map>
#include <iterator>
#include <variant>

class Simulation
{
public:
    explicit Simulation();
    void startSimulation(std::unique_ptr<NeuronModels> model, std::vector<size_t> neuron_ids, int argc, char** argv);

    std::vector<double> x_dims;
    std::vector<double> y_dims;
    std::vector<double> z_dims;
    NetworkGraph* network_graph;
    std::map<size_t, std::vector<NeuronInformation>> information;

private:
    struct MapStringToModelparameter {
        //Only covering up the standard parameters that are set in the model class
        Parameters params;
        std::map < std::string, std::variant<unsigned int, double, size_t>> lookUp;
        
        MapStringToModelparameter()
        {
            lookUp = {
                {"num_neurons", params.num_neurons},
                {"simulation_time", params.simulation_time},
                {"x_0", params.x_0},
                {"tau_x", params.tau_x},
                {"k", params.k},
                {"tau_C", params.tau_C},
                {"beta", params.beta},
                //Changed type of refrac_time from double to unsigned int
                {"refrac_time", params.refrac_time},
                //Changed type of h from int to unsigned int
                {"h", params.h}
            };
        }

        void setValue(std::string name, std::variant<unsigned int, double, size_t> value) 
        {
            lookUp.find(name)->second = value;
        }

        bool contains(std::string name)
        {
            return lookUp.find(name) != lookUp.end();
        }

        void getParameter(Parameters* params)
        {
            params->num_neurons = std::get<size_t>(lookUp.find("num_neurons")->second);
            params->simulation_time = std::get<size_t>(lookUp.find("simulation_time")->second);
            params->x_0 = std::get<double>(lookUp.find("x_0")->second);
            params->tau_x = std::get<double>(lookUp.find("tau_x")->second);
            params->k = std::get<double>(lookUp.find("k")->second);
            params->tau_C = std::get<double>(lookUp.find("tau_C")->second);
            params->beta = std::get<double>(lookUp.find("beta")->second);
            params->refrac_time = std::get<unsigned int>(lookUp.find("refrac_time")->second);
            params->h = std::get<unsigned int>(lookUp.find("h")->second);
        }
    };
    void setParameters(std::unique_ptr<NeuronModels>& model, Parameters* params);
    void setDefaultParameters(Parameters* params);
    void setSpecificParameters(Parameters* params, const std::vector<std::string> arguments);
    void printTimers();
    void printNeuronMonitor(const NeuronMonitor& nm, size_t neuron_id);

    //Variables
    MapStringToModelparameter lut;
};

#endif // SIMULATION_H