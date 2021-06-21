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
#include "spdlog/sinks/basic_file_sink.h"
#include "../mpi/MPIWrapper.h"
#include "../util/RelearnException.h"

#include <filesystem>
#include <iostream>

bool LogFiles::do_i_print(int rank) {
    return rank == MPIWrapper::get_my_rank() || rank == -1;
}

std::string LogFiles::get_my_rank_str() {
    return MPIWrapper::get_my_rank_str();
}

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

    if (do_i_print(rank)) {
        auto complete_path = output_path + general_prefix + get_specific_file_prefix() + "_" + file_name + ".txt";
        auto logger = spdlog::basic_logger_mt(file_name, complete_path);
        logger->set_pattern("%v");
        log_files.emplace(type, std::move(logger));
    }
}
