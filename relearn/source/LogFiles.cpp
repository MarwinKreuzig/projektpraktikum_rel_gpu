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

// One log file only at the MPI rank "on_rank"
// on_rank == -1 means all ranks
LogFiles::LogFiles(std::string file_name, int on_rank) {
	if (-1 == on_rank || MPIInfos::my_rank == on_rank) {
		num_files = 1;
		files = new std::ofstream[num_files];

		// Open file and overwrite if it already exists
		files[0].open(file_name, std::ios::trunc);
		if (files[0].fail()) {
			std::cout << __func__ << ": Opening file failed." << std::endl;
			exit(EXIT_FAILURE);
		}
	}
}

// Generate series of file name suffixes automatically
LogFiles::LogFiles(size_t num_files, std::string prefix) :
	num_files(num_files) {
	files = new std::ofstream[num_files];

	// Open "num_files" for writing
	for (size_t i = 0; i < num_files; i++) {
		// Open file and overwrite if it already exists
		files[i].open(prefix + std::to_string(i), std::ios::trunc);
		if (files[i].fail()) {
			std::cout << __func__ << ": Opening file failed." << std::endl;
			exit(EXIT_FAILURE);
		}
	}
}

LogFiles::LogFiles(LogFiles&& other) noexcept {
	std::swap(num_files, other.num_files);
	std::swap(files, other.files);
}

// Take array with file name suffixes

LogFiles::LogFiles(size_t num_files, std::string prefix, size_t* suffixes) :
	num_files(num_files) {
	files = new std::ofstream[num_files];

	// Open "num_files" for writing
	for (size_t i = 0; i < num_files; i++) {
		// Open file and overwrite if it already exists
		files[i].open(prefix + std::to_string(suffixes[i]), std::ios::trunc);
		if (files[i].fail()) {
			std::cout << __func__ << ": Opening file failed." << std::endl;
			exit(EXIT_FAILURE);
		}
	}
}

LogFiles::~LogFiles() noexcept(false) {
	// Close all files
	for (size_t i = 0; i < num_files; i++) {
		files[i].close();
	}
	delete[] files;
}

// Get pointer to file stream

std::ofstream* LogFiles::get_file(size_t file_id) {
	if (file_id < num_files) {
		return &files[file_id];
	}
	else {
		std::cout << __func__ << ": File id " << file_id << " too large." << std::endl;
		exit(EXIT_FAILURE);
	}
}
