#pragma once

#include "../../utils/GpuTypes.h"
#include "../../utils/Interface.h"
#include "../../structure/CudaArray.cuh"
#include "enums/ElementType.h"
#include "enums/SignalType.h"

namespace gpu::models {

struct SynapticElements {
    /**
     * Struct representing SynapticElements on the gpu. Contains most of the data contained by the original cpu class
     */

    ElementType type;
    RelearnGPUTypes::number_neurons_type size;

    gpu::Vector::CudaArray<double> grown_elements;
    gpu::Vector::CudaArray<unsigned int> connected_elements;
    gpu::Vector::CudaArray<SignalType> signal_types;

    /**
     * @brief Constructs the SynapticElements on the GPU
     * @param type The element type of these synaptic elements
     */
    __device__ SynapticElements(const ElementType type)
        : type(type) { }

    /**
     * @brief Returns the number of free elements for the specified neuron id
     * @param neuron_id The neuron
     * @return The number of free elements for the neuron
     */
    __device__ unsigned int get_free_elements(const RelearnGPUTypes::neuron_id_type neuron_id) const {

        return static_cast<unsigned int>(grown_elements.data[neuron_id] - connected_elements.data[neuron_id]);
    }
};

class SynapticElementsHandleImpl : public SynapticElementsHandle {
    /**
     * Implementation of the handle for the cpu that controls the gpu object
     */

public:
    /**
     * @brief Constructs the SynapticElementsHandle Implementation
     * @param _dev_ptr The pointer to the SynapticElements object on the GPU
     * @param type The element type the SynapticElements belong to
     */
    SynapticElementsHandleImpl(SynapticElements* _dev_ptr, const ElementType type);

    /**
     * @brief Init function called by the constructor, has to be public in order to be allowed to use device lamdas in it, do not call from outside
     */
    void _init();

    /**
     * @brief Copies the initial values from the CPU version of the class
     * @param number_neurons The number of neurons that should be stored
     * @param grown_elements The grown elements generated in the cpu version of init()
     */
    void init(RelearnGPUTypes::number_neurons_type number_neurons, const std::vector<double>& grown_elements) override;

    /**
     * @brief Copies the on the CPU created neurons on to the GPU
     * @param new_size The new number of neurons
     * @param grown_elements All grown elements of all neurons, including the new ones
     */
    void create_neurons(const RelearnGPUTypes::number_neurons_type new_size, const std::vector<double>& grown_elements) override;

    /**
     * @brief Returns a pointer to the data on the GPU
     */
    [[nodiscard]] void* get_device_pointer() override;

    /**
     * @brief Updates the counts the grown elements of the specified neuron by the specified delta, should not be called since it skips the commit step
     * @param neuron_id The local neuron id
     * @param delta The delta by which the number of elements changes (can be positive and negative)
     */
    void update_grown_elements(const RelearnGPUTypes::neuron_id_type neuron_id, const double delta) override;

    /**
     * @brief Updates the connected elements for the specified neuron by the specified delta
     * @param neuron_id The local neuron id
     * @param delta The delta by which the number of elements changes (can be positive and negative)
     */
    void update_connected_elements(const RelearnGPUTypes::neuron_id_type neuron_id, const int delta) override;

    /**
     * @brief Sets the signal types on the GPU
     * @param types The signal types to copy over to the GPU
     */
    void set_signal_types(const std::vector<SignalType>& types) override;

private:
    SynapticElements* device_ptr;

    ElementType type;

    RelearnGPUTypes::number_neurons_type* handle_size;
    gpu::Vector::CudaArrayDeviceHandle<double> handle_grown_elements;
    gpu::Vector::CudaArrayDeviceHandle<unsigned int> handle_connected_elements;
    gpu::Vector::CudaArrayDeviceHandle<SignalType> handle_signal_types;
};
};