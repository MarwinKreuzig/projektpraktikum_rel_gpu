#pragma once

#if !MPI_FOUND

#include "MPITypes.h"
#include "../util/RelearnException.h"

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <future>

class Octree;
class OctreeNode;

enum class MPI_Locktype : int {
    exclusive = MPI_LOCK_EXCLUSIVE,
    shared = MPI_LOCK_SHARED,
};

class MPINoWrapper {
    struct RMABufferOctreeNodes {
        OctreeNode* ptr;
        size_t num_nodes;
    };

public:
    enum class Scope : char {
        global = 0,
        none = 1
    };

    enum class ReduceFunction : char {
        min = 0,
        max = 1,
        avg = 2,
        sum = 3,
        none = 4,
        minsummax = 100
    };

    using AsyncToken = MPI_Request;

private:
    MPINoWrapper() = default;

    static inline RMABufferOctreeNodes rma_buffer_branch_nodes{};

    static inline const int num_ranks{ 1 }; // Number of ranks in MPI_COMM_WORLD
    static inline const int my_rank{ 0 }; // My rank in MPI_COMM_WORLD

    static inline size_t num_neurons{}; // Total number of neurons

    using async_type = std::tuple<const void*, void*, int>;

    static inline std::map<AsyncToken, async_type> tuple_map{};

    static inline std::string my_rank_str{ '0' };

    static void all_gather(const void* own_data, void* buffer, int size, Scope scope);

    static void reduce(const void* src, void* dst, int size, ReduceFunction function, int root_rank, Scope scope);

    static void async_s(const void* buffer, int count, int rank, Scope scope, AsyncToken& token);

    static void async_recv(void* buffer, int count, int rank, Scope scope, AsyncToken& token);

public:
    static void init(int argc, char** argv);

    static void init_neurons(size_t num_neurons);

    static void init_buffer_octree(size_t num_partitions);

    static void barrier(Scope scope);

    [[nodiscard]] static double reduce(double value, ReduceFunction function, int root_rank, Scope scope);

    [[nodiscard]] static double all_reduce(double value, ReduceFunction function, Scope scope);

    // NOLINTNEXTLINE
    static void all_to_all(const std::vector<size_t>& src, std::vector<size_t>& dst, Scope scope);

    template <typename T>
    // NOLINTNEXTLINE
    static void async_send(const T* buffer, size_t size_in_bytes, int rank, Scope scope, AsyncToken& token) {
        async_s(buffer, static_cast<int>(size_in_bytes), rank, scope, token);
    }

    template <typename T>
    // NOLINTNEXTLINE
    static void async_receive(T* buffer, size_t size_in_bytes, int rank, Scope scope, AsyncToken& token) {
        async_recv(buffer, static_cast<int>(size_in_bytes), rank, scope, token);
    }

    template <typename T, size_t size>
    static void reduce(const std::array<T, size>& src, std::array<T, size>& dst, ReduceFunction function, int root_rank, Scope scope) {
        RelearnException::check(src.size() == dst.size(), "Sizes of vectors don't match");

        reduce(src.data(), dst.data(), src.size() * sizeof(T), function, root_rank, scope);
    }

    template <typename T>
    static void all_gather(T own_data, std::vector<T>& results, Scope scope) {
        all_gather(&own_data, results.data(), sizeof(T), scope);
    }

    template <typename T>
    static void get(T* ptr, int target_rank, int64_t target_display) {
    }

    template <typename T>
    static void all_gather_inline(T* ptr, int count, Scope scope) {
    }

    [[nodiscard]] static int64_t get_ptr_displacement(int target_rank, const OctreeNode* ptr);

    [[nodiscard]] static OctreeNode* new_octree_node();

    [[nodiscard]] static int get_num_ranks();

    [[nodiscard]] static int get_my_rank();

    [[nodiscard]] static size_t get_num_neurons();

    [[nodiscard]] static size_t get_my_num_neurons();

    [[nodiscard]] static size_t get_my_neuron_id_start();

    [[nodiscard]] static size_t get_my_neuron_id_end();

    [[nodiscard]] static size_t get_num_avail_objects();

    [[nodiscard]] static OctreeNode* get_buffer_octree_nodes();

    [[nodiscard]] static size_t get_num_buffer_octree_nodes();

    [[nodiscard]] static std::string get_my_rank_str();

    static void delete_octree_node(OctreeNode* ptr);

    // NOLINTNEXTLINE
    static void wait_request(AsyncToken& request);

    [[nodiscard]] static AsyncToken get_non_null_request();

    [[nodiscard]] static AsyncToken get_null_request();

    // NOLINTNEXTLINE
    static void all_gather_v(size_t total_num_neurons, std::vector<double>& xyz_pos, std::vector<int>& recvcounts, std::vector<int>& displs);

    // NOLINTNEXTLINE
    static void wait_all_tokens(std::vector<AsyncToken>& tokens);

    static void lock_window(int rank, MPI_Locktype lock_type);

    static void unlock_window(int rank);

    static void finalize() /*noexcept*/;
};

#endif
