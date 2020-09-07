/*
 * File:   NeuronModels.cpp
 * Author: naveau
 *
 * Created on September 26, 2014, 9:31 PM
 */

#include "NeuronModels.h"
#include "MPIInfos.h"
#include "Random.h"

NeuronModels::NeuronModels(size_t num_neurons, double x_0, double tau_x, double k, double tau_C, double beta, int h, double refrac_time) :
	my_num_neurons(num_neurons),
	x_0(x_0),
	tau_x(tau_x),
	k(k),
	tau_C(tau_C),
	beta(beta),
	refrac_time(refrac_time),
	h(h),
	random_number_distribution(0.0, nextafter(1.0, 2.0)),
	random_number_generator(RandomHolder<NeuronModels>::get_random_generator()) // Init random number distribution to [0,1]
{
	// Allocate variables for all neurons
	x = new double[my_num_neurons];
	fired = new int[my_num_neurons]();  // Initialize array to zero.
							// This is important for those neurons that I don't own
							// as they are only set when they fire and reset afterwards.
							// That is why, the default should be 0 (not firing)
	refrac = new int[my_num_neurons];
	I_syn = new double[my_num_neurons];

	// Init variables for my neurons only
	for (size_t i = 0; i < my_num_neurons; i++) {
		// Random init of the firing rate from interval [0,1]
		x[i] = random_number_distribution(random_number_generator);
		fired[i] = Theta(x[i]);
		refrac[i] = static_cast<int>(fired[i] * refrac_time);
	}
}

NeuronModels::~NeuronModels() {
	delete[] x;
	delete[] fired;
	delete[] refrac;
	delete[] I_syn;
}
