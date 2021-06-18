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

#include "../util/RelearnException.h"

#include <filesystem>
#include <fstream>
#include <map>
#include <string>

/**
 * This class provides a static interface that allows for writing log messages to predefined files.
 * The path can be set and the filename's prefix can be chosen freely.
 * Some files are only created for the MPI rank 0, some for all.
 */
class LogFiles {
    friend class RelearnTest;

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
    /**
     * This enum classifies the different type of files that can be written to.
     * It also includes Cout, however, using this value does not automatically print to std::cout
     */
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
    static inline std::map<EventType, LogFile> log_files{};
    // NOLINTNEXTLINE
    static inline std::string output_path{ "../output/" };
    // NOLINTNEXTLINE
    static inline std::string general_prefix{ "rank_" };

    static std::string get_specific_file_prefix();

    static void add_logfile(EventType type, const std::string& file_name, int rank);

    static bool disable;

public:
    /**
     * @brief Sets the folder path in which the log files will be generated. It should end with '/'.
     *      Set before calling init()
     *      Default is: "../output/"
     * @parameter path_to_containing_folder The path to the folder in which the files should be generated
     */
    static void set_output_path(const std::string& path_to_containing_folder) {
        output_path = path_to_containing_folder;
    }

    /**
     * @brief Sets the general prefix for every log file.
     *      Set before calling init()
     *      Default is: "rank_"
     * @parameter prefix The prefix for every file
     */
    static void set_general_prefix(const std::string& prefix) {
        general_prefix = prefix;
    }

    /**
     * @brief Initializes all log files.
     *      Call this method after setting the output path and the general prefix when they should be user defined
     * @exception Throws a RelearnException if creating the files fails
     */
    static void init();

    /**
     * @brief Write the message into the file which is associated with the type.
     *      Optionally prints the message also to std::cout
     */
    static void write_to_file(EventType type, const std::string& message, bool also_to_cout);

    /**
     * @brief Print the message only at a certain MPI rank, and does nothing an all other ranks
     * @parameter string The message to print. Does not take ownership of the pointer
     * @parameter rank The rank that should print the message. If set to -1, all ranks print the message
     */
    static void print_message_rank(char const* string, int rank);
};
