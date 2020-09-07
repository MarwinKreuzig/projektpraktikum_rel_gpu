/*
 * File:   LogFiles.h
 * Author: rinke
 *
 * Created on Sep 18, 2015
 */

#ifndef LOGFILES_H
#define LOGFILES_H

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>

#include <map>

#include "MPIInfos.h"

class LogFiles {
public:
	// One log file only at the MPI rank "on_rank"
	// on_rank == -1 means all ranks
	LogFiles(std::string file_name, int on_rank) {
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
	LogFiles(size_t num_files, std::string prefix) :
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

	LogFiles(LogFiles&& other) noexcept {
		std::swap(num_files, other.num_files);
		std::swap(files, other.files);
	}

	LogFiles& operator=(LogFiles&& other) noexcept {
		std::swap(num_files, other.num_files);
		std::swap(files, other.files);

		return *this;
	}

	// Take array with file name suffixes
	LogFiles(size_t num_files, std::string prefix, size_t* suffixes) :
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

	~LogFiles() {
		// Close all files
		for (size_t i = 0; i < num_files; i++) {
			files[i].close();
		}
		delete[] files;
	}

	size_t get_num_files() { return num_files; }

	// Get pointer to file stream
	std::ofstream* get_file(size_t file_id) {
		if (file_id < num_files) {
			return &files[file_id];
		}
		else {
			std::cout << __func__ << ": File id " << file_id << " too large." << std::endl;
			exit(EXIT_FAILURE);
		}
	}

private:
	size_t num_files = 0;      // Number of files
	std::ofstream* files = nullptr;  // All file streams
};

namespace Logs {
	extern std::string output_dir;

	extern std::map<std::string, LogFiles> logfiles;

	void addLogFile(const std::string& name, int rank);

	void init();

	inline LogFiles& get(const std::string& name) { return logfiles.find(name)->second; }
}


#endif /* LOGFILES_H */
