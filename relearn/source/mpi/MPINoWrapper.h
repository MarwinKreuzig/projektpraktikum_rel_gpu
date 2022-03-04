#pragma once

#include "Config.h"

#if !RELEARN_MPI_FOUND

#include "CommunicationMap.h"
#include "io/LogFiles.h"
#include "util/MemoryHolder.h"
#include "util/RelearnException.h"

#include <array>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

using MPI_Request = int;

constexpr inline auto MPI_LOCK_EXCLUSIVE = 0;
constexpr inline auto MPI_LOCK_SHARED = 1;

template <typename T>
class OctreeNode;
class RelearnTest;

enum class MPI_Locktype : int {
    Exclusive = MPI_LOCK_EXCLUSIVE,
    Shared = MPI_LOCK_SHARED,
};

class MPINoWrapper {
    friend class RelearnTest;

public:
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

    static inline const int num_ranks{ 1 }; // Number of ranks in MPI_COMM_WORLD
    static inline const int my_rank{ 0 }; // My rank in MPI_COMM_WORLD

    static inline size_t num_neurons{}; // Total number of neurons

    template <typename AdditionalCellAttributes>
    static inline std::vector<OctreeNode<AdditionalCellAttributes>> base_ptr{ 0 }; // Start address of MPI-allocated memory

    static inline int64_t base_pointers{}; // RMA window base pointers of all procs

    static inline std::string my_rank_str{ '0' };

    static void all_gather(const void* own_data, void* buffer, int size);

    static void reduce(const void* src, void* dst, int size, ReduceFunction function, int root_rank);

public:
    static void init(int argc, char** argv);

    template <typename AdditionalCellAttributes>
    static void init_buffer_octree() {
        const auto max_num_objects = Constants::mpi_alloc_mem / sizeof(OctreeNode<AdditionalCellAttributes>);

        base_ptr<AdditionalCellAttributes>.resize(max_num_objects, OctreeNode<AdditionalCellAttributes>());

        // create_rma_window();
        // NOLINTNEXTLINE
        base_pointers = reinterpret_cast<int64_t>(base_ptr<AdditionalCellAttributes>.data());

        // NOLINTNEXTLINE
        auto cast = reinterpret_cast<OctreeNode<AdditionalCellAttributes>*>(base_ptr<AdditionalCellAttributes>.data());

        MemoryHolder<AdditionalCellAttributes>::init(cast, max_num_objects);

        LogFiles::print_message_rank(0, "MPI RMA MemAllocator: max_num_objects: {}  sizeof(OctreeNode): {}", max_num_objects, sizeof(OctreeNode<AdditionalCellAttributes>));
    }

    static void barrier();

    [[nodiscard]] static double reduce(double value, ReduceFunction function, int root_rank);

    [[nodiscard]] static double all_reduce_double(double value, ReduceFunction function);

    [[nodiscard]] static uint64_t all_reduce_uint64(uint64_t value, ReduceFunction function);

    [[nodiscard]] static std::vector<size_t> all_to_all(const std::vector<size_t>& src);

    template <typename T, size_t size>
    [[nodiscard]] static std::array<T, size> reduce(const std::array<T, size>& src, ReduceFunction function, int root_rank) {
        RelearnException::check(root_rank >= 0, "In MPIWrapper::reduce, root_rank was negative");

        std::array<T, size> dst{};
        reduce(src.data(), dst.data(), src.size() * sizeof(T), function, root_rank);

        return dst;
    }

    template <typename RequestType>
    [[nodiscard]] static CommunicationMap<RequestType> exchange_requests(const CommunicationMap<RequestType>& outgoing_requests) {
        return outgoing_requests;
    }

    template <typename T>
    static std::vector<T> all_gather(T own_data) {
        std::vector<T> results(1);
        all_gather(&own_data, results.data(), sizeof(T));
        return results;
    }

    template <typename T>
    static void all_gather_inline(std::span<T> buffer) {
    }

    template <typename AdditionalCellAttributes>
    static void download_octree_node(OctreeNode<AdditionalCellAttributes>* dst, [[maybe_unused]] int target_rank, const OctreeNode<AdditionalCellAttributes>* src) {
        *dst = *src;
    }

    [[nodiscard]] static int64_t get_base_pointers() noexcept {
        return base_pointers;
    }

    [[nodiscard]] static int get_num_ranks();

    [[nodiscard]] static std::vector<int> get_ranks() {
        return { 0 };
    }

    [[nodiscard]] static int get_my_rank();

    [[nodiscard]] static size_t get_num_neurons();

    [[nodiscard]] static size_t get_my_num_neurons();

    [[nodiscard]] static size_t get_my_neuron_id_start();

    [[nodiscard]] static size_t get_my_neuron_id_end();

    [[nodiscard]] static std::string get_my_rank_str();

    template <typename T>
    static std::vector<std::vector<T>> exchange_values(const std::vector<std::vector<T>>& values) {
        RelearnException::check(values.size() == 1 && values[0].size() == 0, "MPINoWrapper::exchange_values: There were values!");
        std::vector<std::vector<T>> return_value(1, std::vector<T>(0));
        return return_value;
    }

    static void lock_window(int rank, MPI_Locktype lock_type);

    static void unlock_window(int rank);

    static uint64_t get_number_bytes_sent() noexcept {
        return 0;
    }

    static uint64_t get_number_bytes_received() noexcept {
        return 0;
    }

    static uint64_t get_number_bytes_remote_accessed() noexcept {
        return 0;
    }

    static void finalize();
};

#endif
