#pragma once

#include "Config.h"

#if !RELEARN_MPI_FOUND

#include "io/LogFiles.h"
#include "mpi/CommunicationMap.h"
#include "util/MemoryHolder.h"
#include "util/MPIRank.h"
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
        Min = 0,
        Max = 1,
        Sum = 2,
        None = 3,
        MinSumMax = 100
    };

    using AsyncToken = MPI_Request;

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

        std::span<OctreeNode<AdditionalCellAttributes>> span{ cast, max_num_objects };
        MemoryHolder<AdditionalCellAttributes>::init(span);

        LogFiles::print_message_rank(0, "MPI RMA MemAllocator: max_num_objects: {}  sizeof(OctreeNode): {}", max_num_objects, sizeof(OctreeNode<AdditionalCellAttributes>));
    }

    static void barrier();

    [[nodiscard]] static double reduce(double value, ReduceFunction function, MPIRank root_rank);

    [[nodiscard]] static double all_reduce_double(double value, ReduceFunction function);

    [[nodiscard]] static uint64_t all_reduce_uint64(uint64_t value, ReduceFunction function);

    [[nodiscard]] static std::vector<size_t> all_to_all(const std::vector<size_t>& src);

    template <typename T, size_t size>
    [[nodiscard]] static std::array<T, size> reduce(const std::array<T, size>& src, ReduceFunction function, MPIRank root_rank) {
        RelearnException::check(root_rank.is_initialized(), "In MPIWrapper::reduce, root_rank was negative");

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
    static void download_octree_node(OctreeNode<AdditionalCellAttributes>* dst, const MPIRank target_rank, const OctreeNode<AdditionalCellAttributes>* src, const int number_elements) {
        for (auto i = 0; i < number_elements; i++) {
            dst[i] = src[i];
        }
    }

    template <typename AdditionalCellAttributes>
    static void download_octree_node(OctreeNode<AdditionalCellAttributes>* dst, const MPIRank target_rank, const uint64_t offset, const int number_elements) {
        RelearnException::fail("MPINoWrapper::download_octree_node: Cannot perform the offset version without MPI.");
    }

    [[nodiscard]] static int64_t get_base_pointers() noexcept {
        return base_pointers;
    }

    [[nodiscard]] static size_t get_num_ranks();

    [[nodiscard]] static MPIRank get_my_rank();

    [[nodiscard]] static std::string get_my_rank_str();

    template <typename T>
    static std::vector<std::vector<T>> exchange_values(const std::vector<std::vector<T>>& values) {
        RelearnException::check(values.size() == 1 && values[0].size() == 0, "MPINoWrapper::exchange_values: There were values!");
        std::vector<std::vector<T>> return_value(1, std::vector<T>(0));
        return return_value;
    }

    static void lock_window(MPIRank rank, MPI_Locktype lock_type);

    static void unlock_window(MPIRank rank);

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

private:
    MPINoWrapper() = default;

    static inline size_t num_neurons{}; // Total number of neurons

    template <typename AdditionalCellAttributes>
    static inline std::vector<OctreeNode<AdditionalCellAttributes>> base_ptr{ 0 }; // Start address of MPI-allocated memory

    static inline int64_t base_pointers{}; // RMA window base pointers of all procs

    static inline std::string my_rank_str{ '0' };

    static void all_gather(const void* own_data, void* buffer, int size);

    static void reduce(const void* src, void* dst, int size, ReduceFunction function, int root_rank);
};

#endif
