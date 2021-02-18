/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelearnException.h"

#include "MPIWrapper.h"

#include <iostream>
#include <sstream>

[[nodiscard]] const char* RelearnException::what() const noexcept {
    return message.c_str();
}

void RelearnException::fail(std::string&& message) {
    if (hide_messages) {
        throw RelearnException{};
    }

    const auto my_rank = MPIWrapper::get_my_rank();
    const auto num_ranks = MPIWrapper::get_num_ranks();

    std::stringstream sstream;

    sstream
        << "There was an error at rank: "
        << my_rank
        << " of "
        << num_ranks
        << "!\n"
        << message;

    std::cerr << sstream.str() << std::flush;

    throw RelearnException{ std::move(message) };
}
