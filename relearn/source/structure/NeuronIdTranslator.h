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

#include "../neurons/helper/RankNeuronId.h"
#include "../util/Vec3.h"

#include <filesystem>
#include <map>
#include <memory>
#include <vector>

class Partition;

class NeuronIdTranslator {
protected:
    using position_type = RelearnTypes::position_type;
    std::shared_ptr<Partition> partition{};

public:
    explicit NeuronIdTranslator(std::shared_ptr<Partition> partition)
        : partition(std::move(partition)) { }

    virtual std::map<size_t, RankNeuronId> translate_global_ids(const std::vector<size_t>& global_ids) = 0;

    virtual ~NeuronIdTranslator() = default;
};

class NeuronIdTranslatorFile : public NeuronIdTranslator {
protected:
    std::filesystem::path path_to_neurons{};

public:
    NeuronIdTranslatorFile(std::shared_ptr<Partition> partition, std::filesystem::path path_to_neurons)
        : NeuronIdTranslator(std::move(partition))
        , path_to_neurons(std::move(path_to_neurons)) { }

    std::map<size_t, RankNeuronId> translate_global_ids(const std::vector<size_t>& global_ids);

private:
    std::map<size_t, position_type> load_neuron_positions(const std::vector<size_t>& global_ids);
};
