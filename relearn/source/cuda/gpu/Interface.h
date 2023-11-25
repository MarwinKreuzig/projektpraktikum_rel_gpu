#pragma once

#include "Macros.h"
#include "GpuTypes.h"
#include "enums/FiredStatus.h"

#include <memory>
#include <vector>

/**
 * This class defines an interface to access the gpu implementation from the host side.
 * It separates the cuda code from the host code.
 * Most methods create a handle to control the object on the gpu from the here defined methods from the host.
 * The target relearn_lib uses only the definition without cuda code as it can be compiled without cuda
 * The target relearn_lib_gpu implements this interface and handles all GPU calls
 */

// Macros to define methods as virtual/implemented somewhere else when compiled with cuda
#ifndef CUDA_DEFINITION
#define CUDA_DEFINITION ;
#define CUDA_PTR_DEFINITION CUDA_DEFINITION
#endif

namespace gpu::neurons {
class NeuronsExtraInfosHandle {
    /**
     * Virtual class that is the equivalent of its host class. Call the virtual methods from the corresponding cpu methods.
     * This class can only be created with the create() method.
     */
public:
    /**
     * Save neurons as disabled. Neurons must be enabled beforehand
     * @param neuron_ids Vector with neuron ids that we disable
     */
    virtual void disable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) = 0;

    /**
     * Save neurons as enabled. Neurons must be disabled beforehand
     * @param neuron_ids Vector with neuron ids that we enable
     */
    virtual void enable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) = 0;

    /**
     * Initialize the class when the number of neurons is known
     * @param Number Number local neurons
     */
    virtual void init(RelearnGPUTypes::number_neurons_type number_neurons) = 0;

    /**
     * Creates new neurons
     * @param num_created_neuron Number of newly created neurons
     */
    virtual void create_neurons(RelearnGPUTypes::number_neurons_type num_created_neurons) = 0;
};

/**
 * @return Pointer to the class that handles the NeuronExtraInfos on the gpu
 */
std::unique_ptr<NeuronsExtraInfosHandle> create() CUDA_PTR_DEFINITION
};

namespace gpu::background {
class BackgroundHandle {
    /**
     * Virtual class that is the equivalent of its host class. Call the virtual methods from the corresponding cpu methods.
     * This class can only be created with the create() method.
     */
public:
    /**
     * Initialize the class when the number of neurons is known
     * @param Number Number local neurons
     */
    virtual void init(RelearnGPUTypes::number_neurons_type num_neurons) = 0;

    /**
     * Notify the class over newly created neurons
     * @param creation_count Number of newly created neurons
     */
    virtual void create_neurons(RelearnGPUTypes::number_neurons_type num_neurons) = 0;

    /**
     * Copies the background activity from the gpu to the cpu.
     * Donot call this method unless it is absolutly necessary as it introduces additional work on memory and the NeuronModel can access the background activity directly from the gpu memory.
     */
    virtual std::vector<double> get_background_activity() = 0;

    /**
     * Calculates the background activity for each neuron.
     * DONOT call this method during the simulation. It is purely for test purposes as the background activity is calculated via the NeuronModel's update_activity(...) method on the fly
     * @param step Current step
     * @param number_local_neurons Number local neurons
     */
    virtual void update_input_for_all_neurons_on_gpu(RelearnGPUTypes::step_type step, RelearnGPUTypes::number_neurons_type number_local_neurons) = 0;

    /**
     * Set the NeuronsExtraInfos to the model on the gpu
     * @param extra_infos_handle Host handle to the NeuronsExtraInfo instance on the gpu
     */
    virtual void set_extra_infos(const std::unique_ptr<gpu::neurons::NeuronsExtraInfosHandle>& extra_infos_handle) = 0;
};

/**
 * Create constant background activity calculator on gpu.
 * @param c Constant background activity
 * @return Pointer to the class that handles the BackgroundActivitiy on the gpu
 */
std::shared_ptr<BackgroundHandle> set_constant_background(double c) CUDA_PTR_DEFINITION

    /**
     * Create normally distributed background activity calculator on gpu.
     * @param mean The mean of the normal distribution
     * @param stddev The standard deviation of the normal distribution
     * @return Pointer to the class that handles the BackgroundActivitiy on the gpu
     */
    std::shared_ptr<BackgroundHandle> set_normal_background(double mean, double stddev) CUDA_PTR_DEFINITION

    /**
     * Create normally distributed background activity calculator on gpu.
     * TODO Make it a really fast normal background activity calculator
     * @param mean The mean of the normal distribution
     * @param stddev The standard deviation of the normal distribution
     * @return Pointer to the class that handles the BackgroundActivitiy on the gpu
     */
    std::shared_ptr<BackgroundHandle> set_fast_normal_background(double mean, double stddev, size_t multiplier) CUDA_PTR_DEFINITION

};

namespace gpu::models {

class NeuronModelHandle {
public:
    virtual std::vector<FiredStatus> get_fired() = 0;
    /**
     * Notify the class over disabled neurond
     * @param neuron_ids List of disabled neuron ids
     */
    virtual void disable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) = 0;
    /**
     * Notify the class over enabled neurond
     * @param neuron_ids List of disabled neuron ids
     */
    virtual void enable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) = 0;

    /**
     * Initialize the class when the number of neurons is known
     * @param Number Number local neurons
     */
    virtual void init_neuron_model(RelearnGPUTypes::number_neurons_type number_neurons) = 0;

    virtual void init_neurons(const RelearnGPUTypes::number_neurons_type start_id, const RelearnGPUTypes::number_neurons_type end_id) = 0;

    /**
     * Update the activity for all neurons for a single step
     * @param step Current step
     * @param syn_input vector containing the synaptic input for each neuron
     * @param stimulation vector containing the stimulation for each neuron
     */
    virtual void update_activity(const RelearnGPUTypes::step_type step, const std::vector<double>& syn_input, const std::vector<double>& stimulation) = 0;

    /**
     * Notify the class over newly created neurons
     * @param creation_count Number of newly created neurons
     */
    virtual void create_neurons(const RelearnGPUTypes::number_neurons_type creation_count) = 0;

    /**
     * Set the NeuronsExtraInfos to the model on the gpu
     * @param extra_infos_handle Host handle to the NeuronsExtraInfo instance on the gpu
     */
    virtual void set_extra_infos(const std::unique_ptr<gpu::neurons::NeuronsExtraInfosHandle>& extra_infos_handle) = 0;
};
};

namespace gpu::models::izhikevich {

/**
 * Create Izhikevich neuron model on gpu.
 * @param background_handle The host handle to the background activity calculator
 * @param a The dampening factor for u(t)
 * @param b The dampening factor for v(t) inside the equation for d/dt u(t)
 * @param c The reset activity
 * @param d The additional dampening for u(t) in case of spiking
 * @param V_spike The spiking threshold
 * @param k1 The factor for v(t)^2 inside the equation for d/dt v(t)
 * @param k2 The factor for v(t) inside the equation for d/dt v(t)
 * @param k3 The constant inside the equation for d/dt v(t)
 * @return Pointer to the class that handles the NeuronModel on the gpu
 */
std::shared_ptr<NeuronModelHandle> construct_gpu(std::shared_ptr<gpu::background::BackgroundHandle> background_handle, const unsigned int h, double V_spike, double a, double b, double c, double d, double k1, double k2, double k3) CUDA_PTR_DEFINITION

};

namespace gpu::models::poisson {

/**
 * Create Poisson neuron model on gpu.
 * @param background_handle The host handle to the background activity calculator
 * @param synaptic_input_calculator See NeuronModel(...)
 * @param background_activity_calculator See NeuronModel(...)
 * @param stimulus_calculator See NeuronModel(...)
 * @param x_0 The resting membrane potential
 * @param tau_x The dampening factor by which the membrane potential decreases
 * @param refractory_time The number of steps a neuron doesn't spike after spiking
 * @return Pointer to the class that handles the NeuronModel on the gpu
 */
std::shared_ptr<NeuronModelHandle> construct_gpu(std::shared_ptr<gpu::background::BackgroundHandle> background_handle, const unsigned int h, const double x_0,
    const double tau_x,
    const unsigned int refractory_period) CUDA_PTR_DEFINITION

};

namespace gpu::models::aeif {

/**
 * Create AEIF neuron model on gpu.
 * @param background_handle The host handle to the background activity calculator
 * @param C The dampening factor for v(t) (membrane capacitance)
 * @param g_T The leak conductance
 * @param E_L The reset membrane potential (leak reversal potential)
 * @param V_T The spiking threshold in the equation
 * @param d_T The slope factor
 * @param tau_w The dampening factor for w(t)
 * @param a The sub-threshold adaptation
 * @param b The additional dampening for w(t) in case of spiking
 * @param V_spike The spiking threshold in the spiking check
 * @return Pointer to the class that handles the NeuronModel on the gpu
 */
std::shared_ptr<NeuronModelHandle> construct_gpu(std::shared_ptr<gpu::background::BackgroundHandle> background_handle, const unsigned int _h, double _C, double _g_L, double _E_L, double _V_T, double _d_T, double _tau_w, double _a, double _b, double _V_spike) CUDA_PTR_DEFINITION

};

namespace gpu::models::fitz_hugh_nagumo {
/**
 * Create Fitz Hugh Nagumo neuron model on gpu.
 * @param background_handle The host handle to the background activity calculator
 * @param a The constant inside the equation for d/dt w(t)
 * @param b The dampening factor for w(t) inside the equation for d/dt w(t)
 * @param phi The dampening factor for w(t)
 * @return Pointer to the class that handles the NeuronModel on the gpu
 */
std::shared_ptr<NeuronModelHandle> construct_gpu(std::shared_ptr<gpu::background::BackgroundHandle> background_handle, const unsigned int _h, double _a, double _b, double _phi, double _init_w, double _init_x) CUDA_PTR_DEFINITION
};
