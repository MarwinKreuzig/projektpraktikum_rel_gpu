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

#include "Cell.h"
#include "Config.h"
#include "MPIWrapper.h"

#include <array>
#include <cstddef>
#include <optional>

class OctreeNode {
    std::array<OctreeNode*, Constants::number_oct> children{ nullptr };
    Cell cell{};

    bool parent{ false };

    int rank{ -1 }; // MPI rank who owns this octree node
    size_t level{ 0 }; // Level in the tree [0 (= root) ... depth of tree]

public:
    [[nodiscard]] int get_rank() const noexcept {
        return rank;
    }

    [[nodiscard]] size_t get_level() const noexcept {
        return level;
    }

    [[nodiscard]] bool is_parent() const noexcept {
        return parent;
    }

    [[nodiscard]] const std::array<OctreeNode*, Constants::number_oct>& get_children() const noexcept {
        return children;
    }

    [[nodiscard]] const OctreeNode* get_child(size_t idx) const {
        RelearnException::check(idx < Constants::number_oct, "In OctreeNode::get_child const, idx was: %u", idx);
        // NOLINTNEXTLINE
        return children[idx];
    }

    [[nodiscard]] OctreeNode* get_child(size_t idx) {
        RelearnException::check(idx < Constants::number_oct, "In OctreeNode::get_child, idx was: %u", idx);
        // NOLINTNEXTLINE
        return children[idx];
    }

    [[nodiscard]] const Cell& get_cell() const noexcept {
        return cell;
    }

    [[nodiscard]] bool is_local() const noexcept {
        return rank == MPIWrapper::get_my_rank();
    }

    void set_rank(int new_rank) {
        RelearnException::check(new_rank >= 0, "In OctreeNode::set_rank, new_rank was: %u", new_rank);
        rank = new_rank;
    }

    void set_level(size_t new_level) {
        RelearnException::check(new_level < Constants::uninitialized, "In OctreeNode::set_level, new_level was: %u", new_level);
        level = new_level;
    }

    void set_parent() {
        this->parent = true;
    }

    void set_cell_neuron_id(size_t neuron_id) noexcept {
        cell.set_neuron_id(neuron_id);
    }

    void set_cell_size(const Vec3d& min, const Vec3d& max) noexcept {
        cell.set_size(min, max);
    }

    void set_cell_neuron_position(const std::optional<Vec3d>& opt_position) noexcept {
        cell.set_neuron_position(opt_position);
    }

    void set_cell_num_dendrites(unsigned int num_ex, unsigned int num_in) noexcept {
        cell.set_neuron_num_dendrites_exc(num_ex);
        cell.set_neuron_num_dendrites_inh(num_in);
    }

    void set_cell_neuron_pos_exc(const std::optional<Vec3d>& opt_position) noexcept {
        cell.set_neuron_position_exc(opt_position);
    }

    void set_cell_neuron_pos_inh(const std::optional<Vec3d>& opt_position) noexcept {
        cell.set_neuron_position_inh(opt_position);
    }

    void set_child(OctreeNode* node, size_t idx) {
        RelearnException::check(idx < Constants::number_oct, "In OctreeNode::set_child, idx was: %u", idx);
        // NOLINTNEXTLINE
        children[idx] = node;
    }

    void print() const;

    void reset() {
        cell = Cell{};
        children = std::array<OctreeNode*, Constants::number_oct>{ nullptr };
        parent = false;
        rank = -1;
        level = 0;
    }
};
