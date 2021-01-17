/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "LogFiles.h"
#include "MPIWrapper.h"
#include "RelearnException.h"

#include <filesystem>

namespace Logs {
std::string output_dir;

std::map<std::string, LogFiles> logfiles;

void init() {
    Logs::output_dir = "../output/";

    if (0 == MPIWrapper::my_rank) {
        std::filesystem::path output_path(Logs::output_dir);
        if (!std::filesystem::exists(output_path)) {
            std::filesystem::create_directory(output_path);
        }
    }

    // Wait until directory is created before any rank proceeds
    MPIWrapper::barrier(MPIWrapper::Scope::global);

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
    Logs::addLogFile("network_rank_" + MPIWrapper::my_rank_str, -1);

    // Create log file for positions on all ranks
    Logs::addLogFile("positions_rank_" + MPIWrapper::my_rank_str, -1);
}

void addLogFile(const std::string& name, int rank) {
    LogFiles lf(output_dir + name + ".txt", rank);
    logfiles.insert(std::pair<const std::string, LogFiles>(name, std::move(lf)));
}
} //namespace Logs

// One log file only at the MPI rank "on_rank"
// on_rank == -1 means all ranks
LogFiles::LogFiles(const std::string& file_name, int on_rank) {
    if (-1 == on_rank || MPIWrapper::my_rank == on_rank) {
        num_files = 1;
        files.resize(num_files);

        // Open file and overwrite if it already exists
        files[0].open(file_name, std::ios::trunc);
        if (files[0].fail()) {
            RelearnException::fail("Opening file failed");
        }
    }
}

// Generate series of file name suffixes automatically
LogFiles::LogFiles(size_t num_files, const std::string& prefix)
    : num_files(num_files) {
    files.resize(num_files);

    // Open "num_files" for writing
    for (size_t i = 0; i < num_files; i++) {
        // Open file and overwrite if it already exists
        files[i].open(prefix + std::to_string(i), std::ios::trunc);
        if (files[i].fail()) {
            RelearnException::fail("Opening file failed");
        }
    }
}

LogFiles::LogFiles(LogFiles&& other) noexcept {
    std::swap(num_files, other.num_files);
    std::swap(files, other.files);
}

LogFiles& LogFiles::operator=(LogFiles&& other) noexcept {
    std::swap(num_files, other.num_files);
    std::swap(files, other.files);

    return *this;
}

// Take array with file name suffixes
LogFiles::LogFiles(size_t num_files, const std::string& prefix, std::vector<size_t> suffixes)
    : num_files(num_files) {
    RelearnException::check(suffixes.size() == num_files, "Number of suffixes does not match number of files");

    files.resize(num_files);

    // Open "num_files" for writing
    for (size_t i = 0; i < num_files; i++) {
        // Open file and overwrite if it already exists
        files[i].open(prefix + std::to_string(suffixes[i]), std::ios::trunc);
        if (files[i].fail()) {
            RelearnException::fail("Opening file failed");
        }
    }
}

LogFiles::~LogFiles() noexcept(false) {
    // Close all files
    for (size_t i = 0; i < num_files; i++) {
        files[i].close();
    }
}

// Get pointer to file stream

std::ofstream& LogFiles::get_file(size_t file_id) {
    if (file_id < num_files) {
        return files[file_id];
    }

    RelearnException::fail("File id was too large");
    std::terminate();
}
