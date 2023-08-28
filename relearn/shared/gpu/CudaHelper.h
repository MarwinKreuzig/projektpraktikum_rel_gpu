#pragma once

#include "gpu/Macros.h"

#include <span>
#include "Types.h"


class CudaHelper {
public:

    static void set_use_cuda(bool u) {
        CudaHelper::use_cuda = u;
    }

    static bool is_cuda_available() {
        return RELEARN_CUDA_FOUND && CudaHelper::use_cuda;
    }

    static std::vector<size_t> convert_neuron_ids_to_primitives(const std::span<const NeuronID> ids) {
        std::vector<size_t> prims{};
        prims.reserve(ids.size());
        for(const auto & neuron_id : ids) {
            prims.push_back(neuron_id.get_neuron_id());
        }
        return prims;
    }

    private:
    static inline bool use_cuda = true;
};