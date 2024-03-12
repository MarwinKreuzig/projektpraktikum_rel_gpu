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
     * @brief Gets called by constructor to init class, do not call from outside
     */
    void _init();

    void* get_device_pointer() override;

    // Generally, functionality like this can be done on the cpu here, but somtimes we will need in on a kernel on the GPU
    // In this case, outsource it as a device function into the above struct and call it indirictly over a global function here, if it is also needed on the GPU
    void disable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) override;

    void enable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) override;

    void init(const RelearnGPUTypes::number_neurons_type _num_neurons) override;

    // currently only updates disable_flags, size and positions. Add more parameters as needed
    void create_neurons(RelearnGPUTypes::number_neurons_type new_size, const std::vector<gpu::Vec3d>& positions) override;

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