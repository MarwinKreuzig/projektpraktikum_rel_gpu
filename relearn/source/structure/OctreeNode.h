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

#include <array>
#include <optional>
#include <stack>
#include <vector>

class OctreeNode {
    std::array<OctreeNode*, Constants::number_oct> children{ nullptr };
    Cell<FastMultipoleMethodsCell> cell{};

    bool parent{ false };

    int rank{ -1 }; // MPI rank who owns this octree node
    size_t level{ Constants::uninitialized }; // Level in the tree [0 (= root) ... depth of tree]
    std::vector<const OctreeNode*> interaction_list;
    std::array<double, Constants::p3> hermite_coefficients_ex{ -1.0 };
    std::array<double, Constants::p3> hermite_coefficients_in{ -1.0 };

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

    [[nodiscard]] const Cell<FastMultipoleMethodsCell>& get_cell() const noexcept {
        return cell;
    }

    [[nodiscard]] bool is_local() const noexcept;

    OctreeNode* insert(const Vec3d& position, const size_t neuron_id, int rank);

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

    void set_cell_number_dendrites(unsigned int num_ex, unsigned int num_in) noexcept {
        cell.set_number_excitatory_dendrites(num_ex);
        cell.set_number_inhibitory_dendrites(num_in);
    }

    void set_cell_number_axons(unsigned int num_ex, unsigned int num_in) noexcept {
        cell.set_number_excitatory_axons(num_ex);
        cell.set_number_inhibitory_axons(num_in);
    }

    void set_cell_excitatory_dendrites_position(const std::optional<Vec3d>& opt_position) noexcept {
        cell.set_excitatory_dendrites_position(opt_position);
    }

    void set_cell_inhibitory_dendrites_position(const std::optional<Vec3d>& opt_position) noexcept {
        cell.set_inhibitory_dendrites_position(opt_position);
    }

    void set_cell_excitatory_axons_position(const std::optional<Vec3d>& opt_position) noexcept {
        cell.set_excitatory_axons_position(opt_position);
    }

    void set_cell_inhibitory_axons_position(const std::optional<Vec3d>& opt_position) noexcept {
        cell.set_inhibitory_axons_position(opt_position);
    }

    void set_child(OctreeNode* node, size_t idx) {
        RelearnException::check(idx < Constants::number_oct, "In OctreeNode::set_child, idx was: %u", idx);
        // NOLINTNEXTLINE
        children[idx] = node;
    }

    void print() const;

    void reset() {
        cell = Cell<FastMultipoleMethodsCell>{};
        children = std::array<OctreeNode*, Constants::number_oct>{ nullptr };
        parent = false;
        rank = -1;
        level = Constants::uninitialized;
    }

    void set_hermite_coef_ex(unsigned int x, double d) {
        hermite_coefficients_ex[x] = d;
    }

    void set_hermite_coef_in(unsigned int x, double d) {
        hermite_coefficients_in[x] = d;
    }

    void set_hermite_coef_for(unsigned int x, double d, SignalType needed) {
        if (needed == SignalType::EXCITATORY) {
            set_hermite_coef_ex(x, d);
        } else {
            set_hermite_coef_in(x, d);
        }
    }

    double get_hermite_coef_ex(unsigned int x) const {
        return hermite_coefficients_ex[x];
    }

    double get_hermite_coef_in(unsigned int x) const {
        return hermite_coefficients_in[x];
    }

    double get_hermite_coef_for(unsigned int x, SignalType needed) const {
        if (needed == SignalType::EXCITATORY) {
            return get_hermite_coef_ex(x);
        } else {
            return get_hermite_coef_in(x);
        }
    }

    void add_to_interactionlist(const OctreeNode* x) {
        interaction_list.push_back(x);
    }

    const OctreeNode* get_from_interactionlist(unsigned int x) const {
        if (x >= interaction_list.size()) {
            return nullptr;
        }
        return interaction_list[x];
    }

    size_t get_interactionlist_length() const {
        return interaction_list.size();
    }

    void reset_interactionlist() {
        interaction_list.clear();
    }

    std::vector<Vec3d> get_all_dendrite_positions_for(SignalType needed) const {
        std::vector<Vec3d> result{};

        std::stack<const OctreeNode*> stack{};
        stack.push(this);

        while (!stack.empty()) {
            const OctreeNode* current_node = stack.top();
            stack.pop();

            if (!current_node->is_parent()) {
                const auto& cell = current_node->get_cell();
                const auto num_of_ports = cell.get_number_dendrites_for(needed);
                if (num_of_ports > 0) {
                    const auto& opt_position = cell.get_neuron_position();
                    for (auto i = 0; i < num_of_ports; i++) {
                        result.emplace_back(opt_position.value());
                    }
                }
            } else {
                for (auto i = 0; i < 8; i++) {
                    const OctreeNode* children_node = current_node->get_child(i);
                    if (children_node != nullptr && children_node->get_cell().get_number_dendrites_for(needed) > 0) {
                        stack.push(children_node);
                    }
                }
            }
        }

        return result;
    }

    std::vector<Vec3d> get_all_axon_positions_for(SignalType needed) const {
        std::vector<Vec3d> result{};

        std::stack<const OctreeNode*> stack{};
        stack.push(this);

        while (!stack.empty()) {
            const OctreeNode* current_node = stack.top();
            stack.pop();

            if (!current_node->is_parent()) {
                const auto& cell = current_node->get_cell();
                const auto num_of_ports = cell.get_number_axons_for(needed);
                if (num_of_ports > 0) {
                    const auto& opt_position = cell.get_neuron_position();
                    for (auto i = 0; i < num_of_ports; i++) {
                        result.emplace_back(opt_position.value());
                    }
                }
            } else {
                for (auto i = 0; i < 8; i++) {
                    const OctreeNode* children_node = current_node->get_child(i);
                    if (children_node != nullptr && children_node->get_cell().get_number_axons_for(needed) > 0) {
                        stack.push(children_node);
                    }
                }
            }
        }

        return result;
    }

    void print_calculations(SignalType needed, double sigma);
};
