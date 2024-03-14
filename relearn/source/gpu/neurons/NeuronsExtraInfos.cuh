#pragma once

#include "enums/UpdateStatus.h"
#include "../utils/GpuTypes.h"
#include "../utils/Interface.h"
#include "../structure/CudaArray.cuh"
#include "../structure/CudaVector.cuh"
#include <cuda.h>

namespace gpu::neurons {

class NeuronsExtraInfos {
    /**
     * Class representing NeuronsExtraInfos on the gpu. Contains the disable flags and number of local neurons
     */

public:
    size_t number_local_neurons_device = 0;

    gpu::Vector::CudaArray<UpdateStatus> disable_flags;

public:
    /**
     * @return Return the number of local neurons
     */
    inline __device__ size_t get_number_local_neurons() {
        return number_local_neurons_device;
    }
};

class NeuronsExtraInfosHandleImpl : public NeuronsExtraInfosHandle {
    /**
     * Implementation of the handle for the cpu that controls the gpu object
     */
public:
    NeuronsExtraInfosHandleImpl(void* _dev_ptr);

    void _init();

    void* get_device_pointer();

    void disable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) override;

    void enable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) override;

    void init(const RelearnGPUTypes::number_neurons_type _num_neurons) override;

    void set_num_neurons(size_t _num_neurons);

    void create_neurons(size_t creation_count);

private:
    /**
     * Pointer to the NeuronsExtraInfos instance on the gpu
     */
    void* device_ptr;

    size_t num_neurons;

    gpu::Vector::CudaArrayDeviceHandle<UpdateStatus> handle_disable_flags;
};

std::unique_ptr<NeuronsExtraInfosHandle> create();

};