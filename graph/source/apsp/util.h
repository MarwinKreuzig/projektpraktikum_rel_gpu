#ifndef graph_util_h
#define graph_util_h

#include <iostream> // cout
#include <sstream> // stringstream
#include <string> // string
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h> // NOLINT(llvm-include-order)
#include <curand_kernel.h> // Ordering curand_kernel.h before device_launch_parameters.h causes redefinition with C linkage error

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define gpuErrchkNoTermination(ans) \
    { gpuAssert((ans), __FILE__, __LINE__, false); }
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define cudaCheckError() \
    { gpuAssert((cudaPeekAtLastError()), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line) {
#ifndef NDEBUG
	if (code != cudaSuccess) {
		const char* const err_str = cudaGetErrorString(code);
		std::cout << "CUDA Error: " << err_str << " (" << code << ") at: " << file << ":" << line << "\n";
	}
#endif
}

#define __modifier__ __device__ __host__

namespace apsp {

	/**
	 * @brief Wrapper for pointer and size pairs denoting a range
	 *
	 * @tparam T value type of the data
	 */
	template <typename T>
	struct View { // NOLINT(cppcoreguidelines-pro-type-member-init, hicpp-member-init)
		using value_type = T;

		View() = default;

		template <typename Container>
		explicit View(Container& container)
			: data_{ container.data() }
			, size_{ container.size() } { }

		View(T* data, size_t size)
			: data_{ data }
			, size_{ size } { }

		[[nodiscard]] __modifier__ T* data() { return data_; }
		[[nodiscard]] __modifier__ const T* data() const { return data_; }
		[[nodiscard]] __modifier__ size_t size() const { return size_; }
		[[nodiscard]] __modifier__ size_t size_bytes() const { return size() * sizeof(T); }

		[[nodiscard]] __modifier__ T& operator[](size_t i) { return data()[i]; }
		[[nodiscard]] __modifier__ const T& operator[](size_t i) const { return data()[i]; }

		T* data_;
		size_t size_;
	};

	template <typename Container>
	View(Container& container)->View<typename Container::value_type>;

	/**
	 * @brief RAII type for dynamic memory allocation with CUDA
	 *
	 * @tparam T value type of the data
	 */
	template <typename T>
	class RAIIDeviceMemory {
	public:
		using value_type = T;

		RAIIDeviceMemory() = default;

		explicit RAIIDeviceMemory(size_t size)
			: size_{ size } {
			gpuErrchk(cudaMalloc(&data_, size_bytes()));
		}

		RAIIDeviceMemory(const T* in_data, size_t size, cudaMemcpyKind kind = cudaMemcpyHostToDevice)
			: RAIIDeviceMemory{ size } {
			gpuErrchk(cudaMemcpy(data_, in_data, size_bytes(), kind));
		}

		explicit RAIIDeviceMemory(const std::vector<T>& vec, cudaMemcpyKind kind = cudaMemcpyHostToDevice)
			: RAIIDeviceMemory{ vec.data(), vec.size(), kind } {
		}

		RAIIDeviceMemory(const RAIIDeviceMemory&) = delete;
		RAIIDeviceMemory& operator=(const RAIIDeviceMemory&) = delete;

		RAIIDeviceMemory(RAIIDeviceMemory&&) = delete;
		RAIIDeviceMemory& operator=(RAIIDeviceMemory&&) = delete;

		~RAIIDeviceMemory() {
			if (data_)
				gpuErrchk(cudaFree(data_));
		}

		[[nodiscard]] T* data() { return data_; }
		[[nodiscard]] const T* data() const { return data_; }
		[[nodiscard]] size_t size() const { return size_; }
		[[nodiscard]] size_t size_bytes() const { return size_ * sizeof(T); }

		[[nodiscard]] T& operator[](size_t i) { return data()[i]; }
		[[nodiscard]] const T& operator[](size_t i) const { return data()[i]; }

		[[nodiscard]] explicit operator View<T>() { return { data(), size() }; }

		void reallocate(size_t num) {
			if (size() == num) {
				return;
			}
			gpuErrchk(cudaFree(data_));
			size_ = num;
			gpuErrchk(cudaMalloc(&data_, size_bytes()));
		}

		void resize(size_t num, const T& /* unused */) {
			reallocate(num);
		}

		void resize(size_t num) {
			reallocate(num);
		}

	private:
		T* data_{};
		size_t size_{};
	};

	/**
	 * @brief Copy the elements from dst to src. Wrapper for cudaMemcpy
	 *
	 * @tparam T destination container type
	 * @tparam U soruce container type
	 * @param dst destination
	 * @param src source
	 * @param kind direction to copy
	 */
	template <typename T, typename U>
	void copy(T& dst, const U& src, cudaMemcpyKind kind) {
		using dst_type = typename T::value_type;
		using src_type = typename U::value_type;
		static_assert(std::is_same_v<dst_type, src_type>);
		static_assert(!std::is_same_v<src_type, bool>);
		gpuErrchk(cudaMemcpy(dst.data(), src.data(), dst.size() * sizeof(dst_type), kind));
	}

} // namespace apsp

#endif
