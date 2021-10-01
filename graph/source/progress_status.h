#pragma once

#include <atomic>

/**
 * @brief Progress status object.
 *
 * Has atomic members for the progress, total steps, if the operation has started, and if it has finished.
 *
 */
struct progress_status {
    std::atomic_size_t progress = 0U;
    std::atomic_size_t total = 0U;
    std::atomic_bool started{ false };
    std::atomic_bool done{ false };

    /**
     * @brief Get the progress percentage
     *
     * @return float percentage [0.0, 1.0]
     */
    [[nodiscard]] float get_percentage() const { return static_cast<float>(static_cast<double>(progress.load()) / static_cast<double>(total.load())); }
};
