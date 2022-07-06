#include "MultipleFilesSynapseLoader.h"

MultipleFilesSynapseLoader::MultipleFilesSynapseLoader(std::shared_ptr<Partition> partition, std::optional<std::filesystem::path> path_to_synapses)
    : SynapseLoader(std::move(partition)) {
}

MultipleFilesSynapseLoader::synapses_tuple_type MultipleFilesSynapseLoader::internal_load_synapses() {
    return synapses_tuple_type();
}
