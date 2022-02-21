#include "MPINoWrapper.h"

#if !RELEARN_MPI_FOUND

#include "algorithm/BarnesHutCell.h"
#include "io/LogFiles.h"
#include "structure/OctreeNode.h"
#include "util/MemoryHolder.h"
#include "util/RelearnException.h"
#include "util/Utility.h"

#include <bitset>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>

void MPINoWrapper::init(int argc, char** argv) {
    LogFiles::print_message_rank(0, "I'm using the MPINoWrapper");
}

void MPINoWrapper::barrier() {
}

[[nodiscard]] double MPINoWrapper::reduce(double value, ReduceFunction /*function*/, int /*root_rank*/) {
    return value;
}

[[nodiscard]] double MPINoWrapper::all_reduce_double(double value, ReduceFunction /*function*/) {
    return value;
}

[[nodiscard]] uint64_t MPINoWrapper::all_reduce_uint64(uint64_t value, ReduceFunction /*function*/) {
    return value;
}

std::vector<size_t> MPINoWrapper::all_to_all(const std::vector<size_t>& src) {
    return src;
}

void MPINoWrapper::reduce(const void* src, void* dst, int size, ReduceFunction /*function*/, int /*root_rank*/) {
    std::memcpy(dst, src, size);
}

void MPINoWrapper::all_gather(const void* own_data, void* buffer, int size) {
    std::memcpy(buffer, own_data, size);
}

[[nodiscard]] int MPINoWrapper::get_num_ranks() {
    return num_ranks;
}

[[nodiscard]] int MPINoWrapper::get_my_rank() {
    return my_rank;
}

[[nodiscard]] size_t MPINoWrapper::get_num_neurons() {
    return num_neurons;
}

[[nodiscard]] size_t MPINoWrapper::get_my_num_neurons() {
    return num_neurons;
}

[[nodiscard]] size_t MPINoWrapper::get_my_neuron_id_start() {
    return 0;
}

[[nodiscard]] size_t MPINoWrapper::get_my_neuron_id_end() {
    return num_neurons;
}

[[nodiscard]] std::string MPINoWrapper::get_my_rank_str() {
    return my_rank_str;
}

void MPINoWrapper::lock_window(int rank, MPI_Locktype /*lock_type*/) {
    RelearnException::check(rank >= 0, "rank was: %d", rank);
}

void MPINoWrapper::unlock_window(int rank) {
    RelearnException::check(rank >= 0, "rank was: %d", rank);
}

void MPINoWrapper::finalize() {
}

#endif
