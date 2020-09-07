#include <mpi.h>

#include "LogFiles.h"

namespace Logs {
	std::string output_dir;

	std::map<std::string, LogFiles> logfiles;

	void init() {
		Logs::output_dir = "../output/";

		if (0 == MPIInfos::my_rank) {
			if (!std::filesystem::exists(Logs::output_dir)) {
				std::filesystem::create_directory(Logs::output_dir);
			}
		}

		// Wait until directory is created before any rank proceeds
		MPI_Barrier(MPI_COMM_WORLD);

		// Neurons to create log file for
		//size_t num_neurons_to_log = 3;
		//size_t neurons_to_log[num_neurons_to_log] = {0, 10, 19};

		// Create log files for neurons
		//LogFiles log_files(num_neurons_to_log, Logs::output_dir + "neuron_", neurons_to_log);

		// Create log file for neurons overview on rank 0
		Logs::addLogFile("neurons_overview", 0);

		// Create log file for sums on rank 0
		Logs::addLogFile("sums", 0);

		// Create log file for network on all ranks
		Logs::addLogFile("network_rank_" + MPIInfos::my_rank_str, -1);

		// Create log file for positions on all ranks
		Logs::addLogFile("positions_rank_" + MPIInfos::my_rank_str, -1);
	}

	void addLogFile(const std::string& name, int rank) {
		LogFiles lf(output_dir + name, rank);
		logfiles.insert(std::pair<const std::string, LogFiles>(name, std::move(lf)));
	}
}
