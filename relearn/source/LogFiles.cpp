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

std::map<LogFiles::EventType, LogFiles::LogFile> LogFiles::log_files;
std::string LogFiles::output_path{ "../output/" };
std::string LogFiles::general_prefix;

void LogFiles::init() {
    if (0 == MPIWrapper::get_my_rank()) {
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
    //LogFiles log_files(num_neurons_to_log, LogFiles::output_dir + "neuron_", neurons_to_log);

    // Create log file for neurons overview on rank 0
    LogFiles::add_logfile(EventType::NeuronsOverview, "neurons_overview");

    // Create log file for sums on rank 0
    LogFiles::add_logfile(EventType::Sums, "sums");

    // Create log file for network on all ranks
    LogFiles::add_logfile(EventType::Network, "network");

    // Create log file for positions on all ranks
    LogFiles::add_logfile(EventType::Positions, "positions");
}

void LogFiles::add_logfile(EventType type, const std::string& file_name) {
    auto complete_path = output_path + general_prefix + "_" + get_specific_file_prefix() + "_" + file_name + ".txt";
    log_files.emplace(type, std::move(complete_path));
}

void LogFiles::write_to_file(EventType type, const std::string& message, bool also_to_cout) {
    const auto iterator = log_files.find(type);
    RelearnException::check(iterator != log_files.end(), "The LogFiles don't contain the requested type");

    iterator->second.write(message);

    if (also_to_cout) {
        std::cout << message;
    }
}

void LogFiles::print_message(char const* string) {
    std::cout << "[INFO]  " << string << "\n";
}

void LogFiles::print_message_rank(char const* string, int rank) {
    if (rank == MPIWrapper::get_my_rank() || rank == -1) {
        std::cout << "[INFO:Rank " << MPIWrapper::get_my_rank() << "]  " << string << "\n";
    }
}

void LogFiles::print_error(char const* string) {
    std::cout << "[ERROR]  " << string << "\n";
}

void LogFiles::print_debug(char const* string) {
    std::cout << "[DEBUG]  " << string << "\n";
}
