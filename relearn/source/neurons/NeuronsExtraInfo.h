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

#include "helper/RankNeuronId.h"
#include "../util/RelearnException.h"
#include "../util/Vec3.h"

#include <optional>
#include <string>
#include <vector>

class NeuronsExtraInfo {
    size_t size{ 0 };

    std::vector<std::string> area_names;
    std::vector<double> x_dims;
    std::vector<double> y_dims;
    std::vector<double> z_dims;

    std::vector<size_t> mpi_rank_to_local_start_id;

public:
    void init(size_t number_neurons) noexcept;

    void create_neurons(size_t creation_count);

    void set_area_names(std::vector<std::string> names) {
        RelearnException::check(area_names.empty(), "Area names are not empty");
        RelearnException::check(size == names.size(), "Size does not match area names count");
        area_names = std::move(names);
    }

    void set_x_dims(std::vector<double> dims) {
        RelearnException::check(x_dims.empty(), "X dimensions are not empty");
        RelearnException::check(size == dims.size(), "Size does not match area names count");
        x_dims = std::move(dims);
    }

    void set_y_dims(std::vector<double> dims) {
        RelearnException::check(y_dims.empty(), "Y dimensions are not empty");
        RelearnException::check(size == dims.size(), "Size does not match area names count");
        y_dims = std::move(dims);
    }

    void set_z_dims(std::vector<double> dims) {
        RelearnException::check(z_dims.empty(), "Z dimensions are not empty");
        RelearnException::check(size == dims.size(), "Size does not match area names count");
        z_dims = std::move(dims);
    }

    [[nodiscard]] const std::vector<double>& get_x_dims() const noexcept {
        return x_dims;
    }

    [[nodiscard]] const std::vector<double>& get_y_dims() const noexcept {
        return y_dims;
    }

    [[nodiscard]] const std::vector<double>& get_z_dims() const noexcept {
        return z_dims;
    }

    [[nodiscard]] Vec3d get_position(size_t idx) const {
        RelearnException::check(idx < size, "Idx must be smaller than size in Positions");
        return Vec3d{ x_dims[idx], y_dims[idx], z_dims[idx] };
    }

    [[nodiscard]] const std::vector<std::string>& get_area_names() const noexcept {
        return area_names;
    }

    [[nodiscard]] double get_x(size_t neuron_id) const {
        RelearnException::check(neuron_id < x_dims.size(), "neuron_id must be smaller than size in NeuronsExtraInfo");
        return x_dims[neuron_id];
    }

    [[nodiscard]] double get_y(size_t neuron_id) const {
        RelearnException::check(neuron_id < y_dims.size(), "neuron_id must be smaller than size in NeuronsExtraInfo");
        return y_dims[neuron_id];
    }

    [[nodiscard]] double get_z(size_t neuron_id) const {
        RelearnException::check(neuron_id < z_dims.size(), "neuron_id must be smaller than size in NeuronsExtraInfo");
        return z_dims[neuron_id];
    }

    [[nodiscard]] const std::string& get_area_name(size_t neuron_id) const {
        RelearnException::check(neuron_id < area_names.size(), "neuron_id must be smaller than size in NeuronsExtraInfo");
        return area_names[neuron_id];
    }

    [[nodiscard]] std::optional<size_t> rank_neuron_id2glob_id(const RankNeuronId& rank_neuron_id) /*noexcept*/ {
        // Rank is not valid
        if (rank_neuron_id.get_rank() < 0 || rank_neuron_id.get_rank() > (mpi_rank_to_local_start_id.size() - 1)) {
            return {};
        }

        const auto glob_id = mpi_rank_to_local_start_id[rank_neuron_id.get_rank()] + rank_neuron_id.get_neuron_id();
        return glob_id;
    }
};
