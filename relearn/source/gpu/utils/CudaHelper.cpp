#include "CudaHelper.h"

void CudaHelper::set_use_cuda(bool enable) {
    CudaHelper::use_cuda = enable;
}

bool CudaHelper::is_cuda_available() {
    return RELEARN_CUDA_FOUND && CudaHelper::use_cuda;
}

std::vector<size_t> CudaHelper::convert_neuron_ids_to_primitives(const std::span<const NeuronID>& ids) {
    std::vector<size_t> prims{};
    prims.reserve(ids.size());
    for (const auto& neuron_id : ids) {
        prims.push_back(neuron_id.get_neuron_id());
    }
    return prims;
}