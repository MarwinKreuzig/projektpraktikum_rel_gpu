#pragma once

#include "Macros.h"
#include "GpuTypes.h"
#include "../../shared/enums/FiredStatus.h"
#include "../../shared/enums/ElementType.h"
#include "enums/SignalType.h"

#include "../structure/OctreeCPUCopy.h"
#include "../structure/GpuDataStructures.h"

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

namespace gpu::algorithm {
class OctreeHandle {
public:
    /**
     * @brief Copies the GPU data structure version of the octree, which was constructed on the CPU, to the GPU
     * @param octree_cpu_copy Struct which holds the octree data to be copied to the GPU
     */
    virtual void copy_to_device(OctreeCPUCopy&& octree_cpu_copy) = 0;

    /**
     * @brief Returns the number of virtual neurons in the octree on the GPU
     * @return The number of virtual neurons in the tree
     */
    [[nodiscard]] virtual RelearnGPUTypes::number_neurons_type get_number_virtual_neurons() const = 0;

    /**
     * @brief Returns the number of neurons in the octree on the GPU
     * @return The number of neurons in the tree
     */
    [[nodiscard]] virtual RelearnGPUTypes::number_neurons_type get_number_neurons() const = 0;

    /**
     * @brief Copies the GPU data structure version of the octree to the CPU
     * @param number_neurons The number of leaf nodes
     * @param number_virtual_neurons The number of virtual neurons
     */
    virtual OctreeCPUCopy copy_to_host(
        const RelearnGPUTypes::number_neurons_type number_neurons,
        const RelearnGPUTypes::number_neurons_type number_virtual_neurons)
        = 0;

    /**
     * @brief Calls the kernel that updates the octree
     */
    virtual void update_virtual_neurons() = 0;

    /**
     * @brief Updates the octree leaf nodes
     */
    virtual void update_leaf_nodes(std::vector<gpu::Vec3d> position_excitatory_element,
        std::vector<gpu::Vec3d> position_inhibitory_element,
        std::vector<RelearnGPUTypes::number_elements_type> num_free_elements_excitatory,
        std::vector<RelearnGPUTypes::number_elements_type> num_free_elements_inhibitory)
        = 0;

    /**
     * @brief Getter for octree_dev_ptr
     * @return octree_dev_ptr
     */
    [[nodiscard]] virtual void* get_device_pointer() = 0;

    /**
     * @brief Getter for Neuron IDs
     * @return Neuron IDs
     */
    [[nodiscard]] virtual std::vector<RelearnGPUTypes::neuron_id_type> get_neuron_ids() = 0;

    /**
     * @brief Returns the total excitatory elements in the tree through the root node
     * @return The total excitatory elements in the tree
     */
    [[nodiscard]] virtual RelearnGPUTypes::number_elements_type get_total_excitatory_elements() = 0;

    /**
     * @brief Returns the total inhibitory elements in the tree through the root node
     * @return The total inhibitory elements in the tree
     */
    [[nodiscard]] virtual RelearnGPUTypes::number_elements_type get_total_inhibitory_elements() = 0;

    /**
     * @brief Returns the position of a node for a given signal type
     * @param node_index The index of the node
     * @param signal_type The signal type
     * @return The position of the node
     */
    [[nodiscard]] virtual gpu::Vec3d get_node_position(RelearnGPUTypes::neuron_index_type node_index, SignalType signal_type) = 0;

    /**
     * @brief Returns the bounding box of the given cell of the node index given
     * @param node_index The index of the node
     * @return The bounding box of the given cell
     */
    [[nodiscard]] virtual std::pair<gpu::Vec3d, gpu::Vec3d> get_bounding_box(RelearnGPUTypes::neuron_index_type node_index) = 0;
};

/**
 * @brief Returns a shared pointer to a newly created handle to the Octree on the GPU
 * @param number_neurons Number of neurons, influences how much memory will be allocated on the GPU
 * @param number_virtual_neurons Number of virtual neurons, influences how much memory will be allocated on the GPU
 * @param stored_element_type Type of elements (Axon or Dendrites)
 */
std::shared_ptr<OctreeHandle> create_octree(RelearnGPUTypes::number_neurons_type number_neurons, RelearnGPUTypes::number_neurons_type number_virtual_neurons, ElementType stored_element_type) CUDA_PTR_DEFINITION
};

namespace gpu::neurons {
class NeuronsExtraInfosHandle {
    /**
     * Virtual class that is the equivalent of its host class. Call the virtual methods from the corresponding cpu methods.
     * This class can only be created with the create() method.
     */
public:
    /**
     * @brief Save neurons as disabled. Neurons must be enabled beforehand
     * @param neuron_ids Vector with neuron ids that we disable
     */
    virtual void disable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) = 0;

    /**
     * @brief Save neurons as enabled. Neurons must be disabled beforehand
     * @param neuron_ids Vector with neuron ids that we enable
     */
    virtual void enable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) = 0;

    /**
     * @brief Initialize the class when the number of neurons is known
     * @param number_neurons Number local neurons
     */
    virtual void init(RelearnGPUTypes::number_neurons_type number_neurons) = 0;

    /**
     * @brief Creates new neurons
     * @param new_size The new number of neurons
     * @param positions The positions of all neurons, including the new ones
     */
    virtual void create_neurons(RelearnGPUTypes::number_neurons_type new_size, const std::vector<gpu::Vec3d>& positions) = 0;

    /**
     * @brief Overwrites the current positions with the supplied ones
     * @param pos The new positions, must have the same size as neurons are stored
     */
    virtual void set_positions(const std::vector<gpu::Vec3d>& pos) = 0;

    /**
     * @brief Returns a pointer to the data on the GPU
     */
    [[nodiscard]] virtual void* get_device_pointer() = 0;
};

/**
 * @return Pointer to the class that handles the NeuronExtraInfos on the gpu
 */
std::unique_ptr<NeuronsExtraInfosHandle> create() CUDA_PTR_DEFINITION

};

namespace gpu::models {
class SynapticElementsHandle {
    /**
     * Virtual class that is the equivalent of its host class. Call the virtual methods from the corresponding cpu methods.
     * This class can only be created with the create() method.
     */

public:
    /**
     * @brief Copies the initial values from the CPU version of the class
     * @param number_neurons The number of neurons that should be stored
     * @param grown_elements The grown elements generated in the cpu version of init()
     */
    virtual void init(RelearnGPUTypes::number_neurons_type number_neurons, const std::vector<double>& grown_elements) = 0;

    /**
     * @brief Copies the on the CPU created neurons on to the GPU
     * @param new_size The new number of neurons
     * @param grown_elements All grown elements of all neurons, including the new ones
     */
    virtual void create_neurons(const RelearnGPUTypes::number_neurons_type new_size, const std::vector<double>& grown_elements) = 0;

    /**
     * @brief Returns a pointer to the data on the GPU
     */
    [[nodiscard]] virtual void* get_device_pointer() = 0;

    /**
     * @brief Updates the counts the grown elements of the specified neuron by the specified delta, should not be called since it skips the commit step
     * @param neuron_id The local neuron id
     * @param delta The delta by which the number of elements changes (can be positive and negative)
     */
    virtual void update_grown_elements(const RelearnGPUTypes::neuron_id_type neuron_id, const double delta) = 0;

    /**
     * @brief Updates the connected elements for the specified neuron by the specified delta
     * @param neuron_id The local neuron id
     * @param delta The delta by which the number of elements changes (can be positive and negative)
     */
    virtual void update_connected_elements(const RelearnGPUTypes::neuron_id_type neuron_id, const int delta) = 0;

    /**
     * @brief Sets the signal types on the GPU
     * @param types The signal types to copy over to the GPU
     */
    virtual void set_signal_types(const std::vector<SignalType>& types) = 0;
};

/**
 * @return Pointer to the class that handles the SynapticElements on the gpu
 */
std::unique_ptr<SynapticElementsHandle> create_synaptic_elements(const ElementType type) CUDA_PTR_DEFINITION

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
