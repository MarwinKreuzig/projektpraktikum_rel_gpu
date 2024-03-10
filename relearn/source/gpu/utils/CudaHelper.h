#pragma once

#include "Macros.h"

#include <span>
#include "Types.h"

class CudaHelper {
public:
    /**
     * Enable/Disable utils support. Default is enabled when compiled with utils
     * @param enable Enable
     */
    static void set_use_cuda(bool enable);

    /**
     * @return True if the code shall be executed with utils
     */
    static bool is_cuda_available();

    /**
     * Converts vector of NeuronIDs to size_t primitives
     */
    static std::vector<size_t> convert_neuron_ids_to_primitives(const std::span<const NeuronID>& ids);

private:
    static inline bool use_cuda = true;
};