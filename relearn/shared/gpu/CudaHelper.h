#pragma once

#include "gpu/Macros.h"

#include <span>
#include "Types.h"


class CudaHelper {
public:

    /**
     * Enable/Disable cuda support. Default is enabled when compiled wth cuda
     * @param enable Enable
    */
    static void set_use_cuda(bool enable) {
        CudaHelper::use_cuda = enable;
    }

    /**
     * @return True if the code shall be executed with cuda
    */
    static bool is_cuda_available() {
        return RELEARN_CUDA_FOUND && CudaHelper::use_cuda;
    }

    /**
     * Converts vector of NeuronIDs to size_t primitives
    */
    static std::vector<size_t> convert_neuron_ids_to_primitives(const std::span<const NeuronID>& ids) {
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