/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "MultipleFilesSynapseLoader.h"

MultipleFilesSynapseLoader::MultipleFilesSynapseLoader(std::shared_ptr<Partition> partition, std::optional<std::filesystem::path> path_to_synapses)
    : SynapseLoader(std::move(partition)) {
}

MultipleFilesSynapseLoader::synapses_tuple_type MultipleFilesSynapseLoader::internal_load_synapses() {
    return synapses_tuple_type();
}
