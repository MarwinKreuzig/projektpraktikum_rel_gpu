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

#include "RelearnException.h"

#include <filesystem>
#include <fstream>
#include <map>
#include <string>

class LogFiles {
    class LogFile {
        std::ofstream ofstream;

    public:
        explicit LogFile(const std::filesystem::path& path)
            : ofstream(path) { }

        LogFile(const LogFile& other) = delete;
        LogFile& operator=(const LogFile& other) = delete;

        LogFile(LogFile&& other) = default;
        LogFile& operator=(LogFile&& other) = default;

        ~LogFile() = default;

        void write(const std::string& message) {
            RelearnException::check(ofstream.is_open(), "The output stream is not open");
            RelearnException::check(ofstream.good(), "The output stream isn't good");
            ofstream << message;
        }
    };

public:
    enum class EventType : char {
        PlasticityUpdate,
        PlasticityUpdateLocal,
        NeuronsOverview,
        Sums,
        Network,
        Positions,
        Cout,
        Timers
    };

private:
    static std::map<EventType, LogFile> log_files;
    static std::string output_path;
    static std::string general_prefix;

    static std::string get_specific_file_prefix();

    static void add_logfile(EventType type, const std::string& file_name, int rank);

public:
    /**
     * Sets the folder path in which the log files will be generated. It should end with '/'.
     * Set before calling init()
     * Default is: "../output/"
     */
    static void set_output_path(const std::string& path_to_containing_folder) {
        output_path = path_to_containing_folder;
    }

    /**
     * Sets the general prefix for every log file.
     * Set before calling init()
     * Default is: "rank_"
     */
    static void set_general_prefix(const std::string& prefix) {
        general_prefix = prefix;
    }

    /**
     * Initializes all log files.
     * Set after setting the output path and the general prefix when they should be user defined
     */
    static void init();

    /**
     * Write the message into the file which is associated with the type.
     * Optionally prints the message also to std::cout
     */
    static void write_to_file(EventType type, const std::string& message, bool also_to_cout);

    // Print tagged message only at MPI rank "rank"
    static void print_message_rank(char const* string, int rank);
};
