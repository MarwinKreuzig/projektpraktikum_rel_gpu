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

#include "NeuronToSubdomainAssignment.h"
#include "Partition.h"
#include "Vec3.h"

#include <fstream>
#include <memory>
#include <set>
#include <string>

// This class reads the neurons with their positions from a file
// and, based on this, determines the size of the simulation box
// and the number of neurons in every individual subdomain.
class SubdomainFromFile : public NeuronToSubdomainAssignment {
public:
    SubdomainFromFile(const std::string& file_path, Partition& partition);

    SubdomainFromFile(const SubdomainFromFile& other) = delete;
    SubdomainFromFile(SubdomainFromFile&& other) = delete;

    SubdomainFromFile& operator=(const SubdomainFromFile& other) = delete;
    SubdomainFromFile& operator=(SubdomainFromFile&& other) = delete;

    ~SubdomainFromFile() override = default;

    void fill_subdomain(size_t subdomain_idx, size_t num_subdomains, const Position& min, const Position& max) override;

    std::vector<size_t> neuron_global_ids(size_t subdomain_idx, size_t num_subdomains,
        size_t local_id_start, size_t local_id_end) const override;

private:
    void read_dimensions_from_file(Partition& partition);

    void read_nodes_from_file(const Position& min, const Position& max, Nodes& nodes);

    std::ifstream file;
};
