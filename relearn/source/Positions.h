/*
 * File:   Positions.h
 * Author: naveau
 *
 * Created on September 26, 2014, 1:28 PM
 */

#ifndef POSITION_H
#define	POSITION_H

#include <cassert>
#include <utility>
#include <vector>

#include "Vec3.h"

class Positions {

public:
	Positions(size_t s) : size(s), x_dims(s), y_dims(s), z_dims(s) {

	}
	~Positions() = default;

	Positions(const Positions& other) = delete;
	Positions(Positions&& other) = default;

	Positions& operator = (const Positions& other) = delete;
	Positions& operator = (Positions&& other) = default;

	const std::vector<double>& get_x_dims() noexcept { return x_dims; };
	const std::vector<double>& get_y_dims() noexcept { return y_dims; };
	const std::vector<double>& get_z_dims() noexcept { return z_dims; };

	Vec3d get_position(size_t idx) const noexcept {
		assert(idx < size && "Idx must be smaller than size in Positions");
		return Vec3d{ x_dims[idx], y_dims[idx], z_dims[idx] };
	}

	void set_x(size_t neuron_id, double x) noexcept { x_dims[neuron_id] = x; };
	void set_y(size_t neuron_id, double y) noexcept { y_dims[neuron_id] = y; };
	void set_z(size_t neuron_id, double z) noexcept { z_dims[neuron_id] = z; };

	double get_x(size_t neuron_id) const noexcept { return x_dims[neuron_id]; };
	double get_y(size_t neuron_id) const noexcept { return y_dims[neuron_id]; };
	double get_z(size_t neuron_id) const noexcept { return z_dims[neuron_id]; };

private:
	size_t size;

	std::vector<double> x_dims;
	std::vector<double> y_dims;
	std::vector<double> z_dims;
};

#endif	/* POSITION_H */
