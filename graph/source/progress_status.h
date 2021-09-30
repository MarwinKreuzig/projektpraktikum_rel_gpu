#pragma once

#include <atomic>

struct progress_status {
    std::atomic_size_t progress = 0U;
    std::atomic_size_t total = 0U;
    std::atomic_bool started{ false };
    std::atomic_bool done{ false };

    [[nodiscard]] float get_percentage() const { return static_cast<float>(static_cast<double>(progress.load()) / static_cast<double>(total.load())); }
};
