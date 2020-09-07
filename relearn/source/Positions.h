/*
 * File:   Positions.h
 * Author: naveau
 *
 * Created on September 26, 2014, 1:28 PM
 */

#ifndef POSITION_H
#define	POSITION_H

#include <assert.h>
#include <random>
#include <iostream>

class Positions {

public:
	Positions(size_t);
	~Positions();

	double* get_x_dims() const { return x_dims; };
	double* get_y_dims() const { return y_dims; };
	double* get_z_dims() const { return z_dims; };
	inline void set_x(size_t neuron_id, double x) { x_dims[neuron_id] = x; };
	inline void set_y(size_t neuron_id, double y) { y_dims[neuron_id] = y; };
	inline void set_z(size_t neuron_id, double z) { z_dims[neuron_id] = z; };
	inline double get_x(size_t neuron_id) { return x_dims[neuron_id]; };
	inline double get_y(size_t neuron_id) { return y_dims[neuron_id]; };
	inline double get_z(size_t neuron_id) { return z_dims[neuron_id]; };

private:
	size_t size;
	double* x_dims;
	double* y_dims;
	double* z_dims;
};

#endif	/* POSITION_H */
