#include "MPINoWrapper.h"

#if !MPI_FOUND

#include "../Config.h"
#include "../io/LogFiles.h"
#include "MPINo_RMA_MemAllocator.h"
#include "../util/RelearnException.h"
#include "../util/Utility.h"

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
}

void MPINoWrapper::init_neurons(size_t num_neurons) {
    /**
	 * Sanity check for use of MPI
	 *
	 * Check if num_neurons fits in int value (see IMPORTANT notice above)
	 */
    RelearnException::check(num_neurons < std::numeric_limits<int>::max(), "init_neurons: num_neurons does not fit in \"int\" data type");

    MPINoWrapper::num_neurons = num_neurons;
}

void MPINoWrapper::init_buffer_octree(size_t num_partitions) {
    MPINo_RMA_MemAllocator::init(Constants::mpi_alloc_mem, num_partitions);

    rma_buffer_branch_nodes.num_nodes = num_partitions;
    rma_buffer_branch_nodes.ptr = MPINo_RMA_MemAllocator::get_branch_nodes();
}

void MPINoWrapper::barrier(Scope scope) {
}

[[nodiscard]] double MPINoWrapper::reduce(double value, ReduceFunction /*function*/, int /*root_rank*/, Scope /*scope*/) {
    return value;
}

[[nodiscard]] double MPINoWrapper::all_reduce(double value, ReduceFunction /*function*/, Scope /*scope*/) {
    return value;
}

void MPINoWrapper::all_to_all(const std::vector<size_t>& src, std::vector<size_t>& dst, Scope /*scope*/) {
    dst = src;
}

void MPINoWrapper::async_s(const void* buffer, int count, int /*rank*/, Scope /*scope*/, AsyncToken& token) {
    if (const auto it = tuple_map.find(token); it != tuple_map.end()) {
        auto [_, dest, async_count] = it->second;
        RelearnException::check(async_count == count, "MPINoWrapper::async_s count mismatch");
        std::memcpy(dest, buffer, count);
        tuple_map.erase(it);
    } else {
        const auto [_, success] = tuple_map.insert({ token, { buffer, nullptr, count } });
        RelearnException::check(success, "MPINoWrapper::async_s insertion to map failed");
    }
}

void MPINoWrapper::async_recv(void* buffer, int count, int /*rank*/, Scope /*scope*/, AsyncToken& token) {
    if (const auto it = tuple_map.find(token); it != tuple_map.end()) {
        auto [src, _, async_count] = it->second;
        RelearnException::check(async_count == count, "MPINoWrapper::async_s count mismatch");
        std::memcpy(buffer, src, count);
        tuple_map.erase(it);
    } else {
        const auto [_, success] = tuple_map.insert({ token, { nullptr, buffer, count } });
        RelearnException::check(success, "MPINoWrapper::async_s insertion to map failed");
    }
}

void MPINoWrapper::reduce(const void* src, void* dst, int size, ReduceFunction /*function*/, int /*root_rank*/, Scope /*scope*/) {
    std::memcpy(dst, src, size);
}

void MPINoWrapper::all_gather(const void* own_data, void* buffer, int size, Scope /*scope*/) {
    std::memcpy(buffer, own_data, size);
}

void MPINoWrapper::download_octree_node(OctreeNode* dst, int /*target_rank*/, const OctreeNode* src) {
    *dst = *src;
}

[[nodiscard]] OctreeNode* MPINoWrapper::new_octree_node() {
    return MPINo_RMA_MemAllocator::new_octree_node();
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

[[nodiscard]] size_t MPINoWrapper::get_num_avail_objects() {
    return MPINo_RMA_MemAllocator::get_num_avail_objects();
}

[[nodiscard]] OctreeNode* MPINoWrapper::get_buffer_octree_nodes() {
    return rma_buffer_branch_nodes.ptr;
}

[[nodiscard]] size_t MPINoWrapper::get_num_buffer_octree_nodes() {
    return rma_buffer_branch_nodes.num_nodes;
}

[[nodiscard]] std::string MPINoWrapper::get_my_rank_str() {
    return my_rank_str;
}

void MPINoWrapper::delete_octree_node(OctreeNode* ptr) {
    MPINo_RMA_MemAllocator::delete_octree_node(ptr);
}

void MPINoWrapper::wait_request(AsyncToken& /*request*/) {
}

void MPINoWrapper::wait_all_tokens(std::vector<AsyncToken>& /*tokens*/) {
}

void MPINoWrapper::lock_window(int rank, MPI_Locktype /*lock_type*/) {
    RelearnException::check(rank >= 0, "rank was: %d", rank);
}

void MPINoWrapper::unlock_window(int rank) {
    RelearnException::check(rank >= 0, "rank was: %d", rank);
}

void MPINoWrapper::finalize() /*noexcept*/ {
    MPINo_RMA_MemAllocator::finalize();
}

#endif
