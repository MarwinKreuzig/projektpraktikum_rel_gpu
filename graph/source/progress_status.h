#pragma once

#include <atomic>
#include <cstddef>

/**
 * @brief Progress status object.
 *
 * Has atomic members for the progress, total steps, if the operation has started, and if it has finished.
 *
 */
struct progress_status {
    progress_status() = default;

    /**
     * @brief Construct a new progress status object
     *
     * @param total_steps total steps
     */
    explicit progress_status(size_t total_steps)
        : total{ total_steps } { }

    /**
     * @brief Construct a new progress status object
     *
     * @param total_steps total steps
     * @param start start on oncstruction
     */
    progress_status(size_t total_steps, bool start)
        : total{ total_steps }
        , started{ start } { }

    progress_status(const progress_status&) = delete;

    progress_status& operator=(const progress_status&) = delete;

    progress_status(progress_status&&) = delete;

    progress_status& operator=(progress_status&&) = delete;

    /**
     * @brief Destroy the progress status object and mark it as done
     *
     */
    ~progress_status() {
        started.store(true, std::memory_order_relaxed);
        set_done();
    }

    virtual void set_start() {
        started = true;
    }

    void set_done() {
        done.store(true, std::memory_order_relaxed);
    }

    progress_status& operator++() {
        progress.fetch_add(1, std::memory_order_relaxed);
        return *this;
    }

    /**
     * @brief Get the progress percentage
     *
     * @return float percentage [0.0, 1.0]
     */
    [[nodiscard]] float get_percentage() const { return static_cast<float>(static_cast<double>(progress.load(std::memory_order_relaxed)) / static_cast<double>(total.load(std::memory_order_relaxed))); }

    std::atomic_size_t progress = 0U;
    std::atomic_size_t total = 0U;
    std::atomic_bool started{ false };
    std::atomic_bool done{ false };
};
