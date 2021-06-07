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
#include "../mpi/MPIWrapper.h"
#include "../util/RelearnException.h"

#include <filesystem>
#include <iostream>

bool LogFiles::disable = false;

void LogFiles::init() {
    if (disable) {
        return;
    }

    if (0 == MPIWrapper::get_my_rank()) {
        if (!std::filesystem::exists(output_path)) {
            std::filesystem::create_directory(output_path);
        }
    }

    // Wait until directory is created before any rank proceeds
    MPIWrapper::barrier(MPIWrapper::Scope::global);

    // Create log file for neurons overview on rank 0
    LogFiles::add_logfile(EventType::NeuronsOverview, "neurons_overview", 0);

    // Create log file for sums on rank 0
    LogFiles::add_logfile(EventType::Sums, "sums", 0);

    // Create log file for network on all ranks
    LogFiles::add_logfile(EventType::Network, "network", -1);

    // Create log file for positions on all ranks
    LogFiles::add_logfile(EventType::Positions, "positions", -1);

    // Create log file for std::cout
    LogFiles::add_logfile(EventType::Cout, "stdcout", -1);

    // Create log file for the timers
    LogFiles::add_logfile(EventType::Timers, "timers", 0);

    // Create log file for the synapse creation and deletion
    LogFiles::add_logfile(EventType::PlasticityUpdate, "plasticity_changes", 0);

    // Create log file for the local synapse creation and deletion
    LogFiles::add_logfile(EventType::PlasticityUpdateLocal, "plasticity_changes_local", -1);
}

std::string LogFiles::get_specific_file_prefix() {
    return MPIWrapper::get_my_rank_str();
}

void LogFiles::add_logfile(EventType type, const std::string& file_name, int rank) {
    if (disable) {
        return;
    }

    if (rank == MPIWrapper::get_my_rank() || rank == -1) {
        auto complete_path = output_path + general_prefix + get_specific_file_prefix() + "_" + file_name + ".txt";
        log_files.emplace(type, std::move(complete_path));
    }
}

void LogFiles::write_to_file(EventType type, const std::string& message, bool also_to_cout) {
    if (disable) {
        return;
    }

    if (also_to_cout) {
        std::cout << message << std::flush;
    }

    // Not all ranks have all log files
    const auto iterator = log_files.find(type);
    if (iterator != log_files.end()) {
        iterator->second.write(message);
    }
}

void LogFiles::print_message_rank(char const* string, int rank) {
    if (disable) {
        return;
    }

    if (rank == MPIWrapper::get_my_rank() || rank == -1) {
        std::string txt = std::string("[INFO:Rank ") + MPIWrapper::get_my_rank_str() + "]\n" + string + "\n\n";
        write_to_file(LogFiles::EventType::Cout, txt, true);
    }
}
