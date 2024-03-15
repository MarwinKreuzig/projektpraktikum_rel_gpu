#pragma once

#include "enums/UpdateStatus.h"
#include "../utils/GpuTypes.h"
#include "../utils/Interface.h"
#include "../structure/CudaArray.cuh"

namespace gpu::neurons {

struct NeuronsExtraInfos {
    /**
     * Struct representing NeuronsExtraInfos on the gpu. Contains most of the data contained by the original cpu class
     */

    RelearnGPUTypes::number_neurons_type num_neurons{0};

    gpu::Vector::CudaArray<UpdateStatus> disable_flags;

public:
    /**
     * @return Return the number of local neurons
     */
    inline __device__ size_t get_number_local_neurons() {
        return number_local_neurons_device;
    }
    gpu::Vector::CudaArray<double3> positions;
};

class NeuronsExtraInfosHandleImpl : public NeuronsExtraInfosHandle {
    /**
     * Implementation of the handle for the cpu that controls the gpu object
     */
public:
    /**
     * @brief Constructs the NeuronsExtraInfosHandle Implementation
     * @param _dev_ptr The pointer to the NeuronsExtraInfos object on the GPU
     */
    NeuronsExtraInfosHandleImpl(NeuronsExtraInfos* _dev_ptr);

    /**
    * @brief Init function called by the constructor, has to be public in order to be allowed to use device lamdas in it, do not call from outside
    */
    void _init();

    /**
     * @brief Returns a pointer to the data on the GPU
     */
    void* get_device_pointer() override;

    /**
     * @brief Save neurons as disabled. Neurons must be enabled beforehand
     * @param neuron_ids Vector with neuron ids that we disable
     */
    void disable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) override;

    /**
     * @brief Save neurons as enabled. Neurons must be disabled beforehand
     * @param neuron_ids Vector with neuron ids that we enable
     */
    void enable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) override;

    /**
     * @brief Initialize the class when the number of neurons is known
     * @param Number Number local neurons
     */
    void init(const RelearnGPUTypes::number_neurons_type _num_neurons) override;

    /**
     * @brief Creates new neurons
     * @param new_size The new number of neurons
     * @param positions The positions of all neurons, including the new ones
     */
    void create_neurons(RelearnGPUTypes::number_neurons_type new_size, const std::vector<gpu::Vec3d>& positions) override;

    /**
     * @brief Overwrites the current positions with the supplied ones
     * @param pos The new positions, must have the same size as neurons are stored
     */
    void set_positions(const std::vector<gpu::Vec3d>& pos) override;

private:
    void set_num_neurons(RelearnGPUTypes::number_neurons_type _num_neurons);

private:
    /**
     * Pointer to the NeuronsExtraInfos instance on the gpu
     */
    NeuronsExtraInfos* device_ptr;

    RelearnGPUTypes::number_neurons_type* handle_num_neurons;
    gpu::Vector::CudaArrayDeviceHandle<UpdateStatus> handle_disable_flags;
    gpu::Vector::CudaArrayDeviceHandle<double3> handle_positions;
};

};
