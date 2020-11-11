#pragma once

#include <vector>

#include "Neurons.h"

struct NeuronInformation {
	double calcium;
	double x;
	bool fired;
	bool refrac;
	double I_sync;

	double axons;
	double axons_connected;
	double dendrites_exc;
	double dendrites_exc_connected;
	double dendrites_inh;
	double dendrites_inh_connected;

	NeuronInformation(
		double c = 0.0, double x = 0.0, bool f = false, bool r = false, double i = 0.0,
		double ax = 0.0, double ax_c = 0.0, double de = 0.0, double de_c = 0.0, double di = 0.0, double di_c = 0.0) :
		calcium(c), x(x), fired(f), refrac(r), I_sync(i), axons(ax), axons_connected(ax_c),
		dendrites_exc(de), dendrites_exc_connected(de_c), dendrites_inh(di), dendrites_inh_connected(di_c) {
	}
};

class NeuronMonitor {
	const Neurons& neurons_to_monitor;
	size_t target_neuron_id;

	std::vector<NeuronInformation> informations;

public:
	static size_t max_steps;
	static size_t current_step;

	NeuronMonitor(size_t neuron_id, const Neurons& neurons)
		: neurons_to_monitor(neurons), target_neuron_id(neuron_id), informations(max_steps) {
	}

	size_t get_target_id() const /*noexcept*/ {
		return target_neuron_id;
	}

	void record_data() {
		if (current_step >= max_steps) {
			return;
		}

		const double& calcium = neurons_to_monitor.calcium[target_neuron_id];
		const double& x = neurons_to_monitor.neuron_models->x[target_neuron_id];
		const bool& fired = neurons_to_monitor.neuron_models->fired[target_neuron_id] > 0;
		const bool& refrac = neurons_to_monitor.neuron_models->get_refrac(target_neuron_id) > 0;
		const double& I_sync = neurons_to_monitor.neuron_models->I_syn[target_neuron_id];

		const double& axons = neurons_to_monitor.axons.cnts[target_neuron_id];
		const double& axons_connected = neurons_to_monitor.axons.connected_cnts[target_neuron_id];
		const double& dendrites_exc = neurons_to_monitor.dendrites_exc.cnts[target_neuron_id];
		const double& dendrites_exc_connected = neurons_to_monitor.dendrites_exc.connected_cnts[target_neuron_id];
		const double& dendrites_inh = neurons_to_monitor.dendrites_inh.cnts[target_neuron_id];
		const double& dendrites_inh_connected = neurons_to_monitor.dendrites_inh.connected_cnts[target_neuron_id];

		informations[current_step] = NeuronInformation(calcium, x, fired, refrac, I_sync, axons, axons_connected, dendrites_exc, dendrites_exc_connected, dendrites_inh, dendrites_inh_connected);
	}

	const std::vector<NeuronInformation>& get_informations() const /*noexcept*/ {
		return informations;
	}
};


