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
#include <vector>

#include "MPIInfos.h"

class LogFiles {
public:
	// One log file only at the MPI rank "on_rank"
	// on_rank == -1 means all ranks
	LogFiles(const std::string& file_name, int on_rank);

	// Generate series of file name suffixes automatically
	LogFiles(size_t num_files, const std::string& prefix);

	LogFiles(const LogFiles& other) = delete;
	LogFiles& operator=(const LogFiles& other) = delete;

	LogFiles(LogFiles&& other) noexcept;

	LogFiles& operator=(LogFiles&& other) noexcept {
		std::swap(num_files, other.num_files);
		std::swap(files, other.files);

		return *this;
	}

	// Take array with file name suffixes
	LogFiles(size_t num_files, const std::string& prefix, std::vector<size_t> suffixes);

	~LogFiles() noexcept(false);

	size_t get_num_files() const noexcept { return num_files; }

	// Get pointer to file stream
	std::ofstream& get_file(size_t file_id);

private:
	size_t num_files = 0;      // Number of files
	std::vector<std::ofstream> files;  // All file streams
};

namespace Logs {
	extern std::string output_dir;

	extern std::map<std::string, LogFiles> logfiles;

	void addLogFile(const std::string& name, int rank);

	void init();

	inline LogFiles& get(const std::string& name) { return logfiles.find(name)->second; }
}


#endif /* LOGFILES_H */
