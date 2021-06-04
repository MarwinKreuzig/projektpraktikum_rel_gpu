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

#include "../Config.h"
#include "../neurons/SignalType.h"
#include "../util/RelearnException.h"
#include "../util/Vec3.h"

#include <optional>
#include <tuple>

class Cell {
public:
    Cell() = default;
    ~Cell() = default;

    Cell(const Cell& other) = default;
    Cell(Cell&& other) = default;

    Cell& operator=(const Cell& other) = default;
    Cell& operator=(Cell&& other) = default;

    void set_size(const Vec3d& min, const Vec3d& max) noexcept {
        xyz_min = min;
        xyz_max = max;
    }

    [[nodiscard]] std::tuple<Vec3d, Vec3d> get_size() const noexcept {
        return std::make_tuple(xyz_min, xyz_max);
    }

    /**
	 * Returns edge length of the cell
	 * Assumes that the cell is cubic, otherwise returns the maximum
	 */
    [[nodiscard]] double get_maximal_dimension_difference() const noexcept {
        const auto diff_vector = xyz_max - xyz_min;
        const auto diff = diff_vector.get_maximum();

        return diff;
    }

    [[nodiscard]] std::optional<Vec3d> get_neuron_position() const;

    [[nodiscard]] std::optional<Vec3d> get_neuron_position_dendrites_exc() const noexcept {
        return dendrites_ex.xyz_pos;
    }

    void set_neuron_position_dendrites_exc(const std::optional<Vec3d>& opt_position) {
        if (opt_position.has_value()) {
            const auto& position = opt_position.value();

            RelearnException::check(xyz_min.get_x() <= position.get_x() && position.get_x() <= xyz_max.get_x(), "In Cell::set_neuron_position_exc, x was not in the box");
            RelearnException::check(xyz_min.get_y() <= position.get_y() && position.get_y() <= xyz_max.get_y(), "In Cell::set_neuron_position_exc, y was not in the box");
            RelearnException::check(xyz_min.get_z() <= position.get_z() && position.get_z() <= xyz_max.get_z(), "In Cell::set_neuron_position_exc, z was not in the box");
        }

        dendrites_ex.xyz_pos = opt_position;
    }

    [[nodiscard]] std::optional<Vec3d> get_neuron_position_dendrites_inh() const noexcept {
        return dendrites_in.xyz_pos;
    }

    void set_neuron_position_dendrites_inh(const std::optional<Vec3d>& opt_position) {
        if (opt_position.has_value()) {
            const auto& position = opt_position.value();

            RelearnException::check(xyz_min.get_x() <= position.get_x() && position.get_x() <= xyz_max.get_x(), "In Cell::set_neuron_position_exc, x was not in the box");
            RelearnException::check(xyz_min.get_y() <= position.get_y() && position.get_y() <= xyz_max.get_y(), "In Cell::set_neuron_position_exc, y was not in the box");
            RelearnException::check(xyz_min.get_z() <= position.get_z() && position.get_z() <= xyz_max.get_z(), "In Cell::set_neuron_position_exc, z was not in the box");
        }

        dendrites_in.xyz_pos = opt_position;
    }

    [[nodiscard]] std::optional<Vec3d> get_neuron_position_axons_exc() const noexcept {
        return axons_ex.xyz_pos;
    }

    void set_neuron_position_axons_exc(const std::optional<Vec3d>& opt_position) noexcept {
        axons_ex.xyz_pos = opt_position;
    }

    void set_neuron_position_axons_inh(const std::optional<Vec3d>& opt_position) noexcept {
        axons_in.xyz_pos = opt_position;
    }

    [[nodiscard]] std::optional<Vec3d> get_neuron_position_axons_inh() const noexcept {
        return axons_in.xyz_pos;
    }

    void set_neuron_position(const std::optional<Vec3d>& opt_position) noexcept {
        set_neuron_position_dendrites_exc(opt_position);
        set_neuron_position_dendrites_inh(opt_position);
        set_neuron_position_axons_exc(opt_position);
        set_neuron_position_axons_inh(opt_position);
    }

    [[nodiscard]] std::optional<Vec3d> get_neuron_position_for(SignalType dendrite_type) const;

    [[nodiscard]] std::optional<Vec3d> get_neuron_dendrite_position_for(SignalType dendrite_type) const;

    [[nodiscard]] std::optional<Vec3d> get_neuron_axon_position_for(SignalType dendrite_type) const;

    void set_neuron_num_dendrites_exc(unsigned int num_dendrites) noexcept {
        dendrites_ex.num_dendrites = num_dendrites;
    }

    [[nodiscard]] unsigned int get_neuron_num_dendrites_exc() const noexcept {
        return dendrites_ex.num_dendrites;
    }

    void set_neuron_num_dendrites_inh(unsigned int num_dendrites) noexcept {
        dendrites_in.num_dendrites = num_dendrites;
    }

    [[nodiscard]] unsigned int get_neuron_num_dendrites_inh() const noexcept {
        return dendrites_in.num_dendrites;
    }

    void set_neuron_num_axons_exc(unsigned int num_axons) noexcept {
        axons_ex.num_axons = num_axons;
    }

    [[nodiscard]] unsigned int get_neuron_num_axons_exc() const noexcept {
        return axons_ex.num_axons;
    }

    void set_neuron_num_axons_inh(unsigned int num_axons) noexcept {
        axons_in.num_axons = num_axons;
    }

    [[nodiscard]] unsigned int get_neuron_num_axons_inh() const noexcept {
        return axons_in.num_axons;
    }

    [[nodiscard]] unsigned int get_neuron_num_dendrites_for(SignalType dendrite_type) const noexcept {
        if (dendrite_type == SignalType::EXCITATORY) {
            return dendrites_ex.num_dendrites;
        }

        return dendrites_in.num_dendrites;
    }

    [[nodiscard]] unsigned int get_neuron_num_axons_for(SignalType dendrite_type) const noexcept {
        if (dendrite_type == SignalType::EXCITATORY) {
            return axons_ex.num_axons;
        }

        return axons_in.num_axons;
    }

    [[nodiscard]] size_t get_neuron_id() const noexcept {
        return neuron_id;
    }

    void set_neuron_id(size_t neuron_id) noexcept {
        this->neuron_id = neuron_id;
    }

    [[nodiscard]] unsigned char get_neuron_octant() const {
        const std::optional<Vec3d>& pos = get_neuron_position();
        RelearnException::check(pos.has_value(), "position didn_t have a value");
        return get_octant_for_position(pos.value());
    }

    [[nodiscard]] unsigned char get_octant_for_position(const Vec3d& pos) const;

    [[nodiscard]] std::tuple<Vec3d, Vec3d> get_size_for_octant(unsigned char idx) const;

    void print() const;

private:
    struct Dendrites {
        // All dendrites have the same position
        std::optional<Vec3d> xyz_pos{};
        unsigned int num_dendrites{ 0 };
        // TODO(future)
        // List colliding_axons;
    };

    struct Axons {
        // All axons have the same position
        std::optional<Vec3d> xyz_pos{};
        unsigned int num_axons = { 0 };
    };

    // Two points describe size of cell
    Vec3d xyz_min{ Constants::uninitialized };
    Vec3d xyz_max{ Constants::uninitialized };

    /**
	 * Cell contains info for one neuron, which could be a "super" neuron
	 *
	 * Info about EXCITATORY dendrites: dendrites_ex
	 * Info about INHIBITORY dendrites: dendrites_in
	 */
    Dendrites dendrites_ex{};
    Dendrites dendrites_in{};

    /**
	 * Cell contains info for one neuron, which could be a "super" neuron
	 *
	 * Info about EXCITATORY axons: axons_ex
	 * Info about INHIBITORY axons: axons_in
	 */
    Axons axons_ex{};
    Axons axons_in{};

    /**
	 * ID of the neuron in the cell.
	 * This is only valid for cells that contain a normal neuron.
	 * For those with a super neuron, it has no meaning.
	 * This info is used to identify (return) the target neuron for a given axon
	 */
    size_t neuron_id{ Constants::uninitialized };
};
