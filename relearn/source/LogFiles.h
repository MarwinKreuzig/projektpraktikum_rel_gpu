/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#pragma once

#include "MPIWrapper.h"

#include <fstream>
#include <map>
#include <string>
#include <vector>

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

    LogFiles& operator=(LogFiles&& other) noexcept;

    // Take array with file name suffixes
    LogFiles(size_t num_files, const std::string& prefix, std::vector<size_t> suffixes);

    ~LogFiles() noexcept(false);

    [[nodiscard]] size_t get_num_files() const noexcept { return num_files; }

    // Get pointer to file stream
    [[nodiscard]] std::ofstream& get_file(size_t file_id);

    static std::string output_dir;
    static std::map<std::string, LogFiles> logfiles;

    static void addLogFile(const std::string& name, int rank);

    static void init();

    static LogFiles& get(const std::string& name) { return LogFiles::logfiles.find(name)->second; }

    /**
	 * Static functions for printing a tagged log message to std::cout
	 */
    static void print_message(char const* string);

    // Print tagged message only at MPI rank "rank"
    static void print_message_rank(char const* string, int rank);

    static void print_error(char const* string);

    static void print_debug(char const* string);

private:
    size_t num_files = 0; // Number of files
    std::vector<std::ofstream> files; // All file streams
};
