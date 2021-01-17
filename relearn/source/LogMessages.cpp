/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "LogMessages.h"

#include "MPIWrapper.h"

#include <iostream>

void LogMessages::print_message(char const* string) {
    std::cout << "[INFO]  " << string << "\n";
}

// Print tagged message only at MPI rank "rank"
void LogMessages::print_message_rank(char const* string, int rank) {
    if (rank == MPIWrapper::my_rank || rank == -1) {
        std::cout << "[INFO:Rank " << MPIWrapper::my_rank << "]  " << string << "\n";
    }
}

void LogMessages::print_error(char const* string) {
    std::cout << "[ERROR]  " << string << "\n";
}

void LogMessages::print_debug(char const* string) {
    std::cout << "[DEBUG]  " << string << "\n";
}
