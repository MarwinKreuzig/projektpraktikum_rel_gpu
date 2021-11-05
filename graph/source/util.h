#pragma once

#include <chrono>
#include <cstdint>
#include <future>
#include <concepts>

#include <iostream>
#include <ostream>
#include <spdlog/fmt/bundled/core.h>
#include <spdlog/fmt/bundled/chrono.h>

#include "progress_status.h"

/**
 * @brief Timed progress status object
 *
 * Construction of the object or calling set_start() sets the start time
 *
 */
struct timed_progress_status : public progress_status {
    using progress_status::progress_status;

    /**
     * @brief Construct a new timed progress status object
     *
     * @param total_steps total steps
     * @param start start on construction
     */
    timed_progress_status(size_t total_steps, bool start)
        : progress_status{ total_steps, start } {
    }

    /**
     * @brief Start the progress measurement
     *
     * Sets the start flag to true and sets the start time to now
     *
     */
    void set_start() override {
        progress_status::set_start();
        start_time = std::chrono::steady_clock::now();
    }

    /**
     * @brief Get the estimated time remaining
     *
     * @return std::chrono::seconds estimated remaining time
     */
    [[nodiscard]] std::chrono::seconds get_eta() const {
        const auto perc = get_percentage();
        if (perc == 0.0) {
            return std::chrono::seconds{ 0 };
        }
        return std::chrono::duration_cast<std::chrono::seconds>(((std::chrono::steady_clock::now() - start_time) / perc) * (1 - perc));
    }

    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
};

/**
 * @brief Print a progress bar with perc progress and a specific width in console.
 *
 * Has an additional format string, that gets appended to the end of the progress bar.
 *
 * @tparam FormatString type of the additional format string
 * @tparam Ts type for the additional format string
 * @param perc percentage [0.0, 1.0]
 * @param width width in characters for the bar
 * @param format extra format string
 * @param args format args
 */
template <typename FormatString = std::string_view, typename... Ts>
void print_progress_bar(float perc, std::uint16_t width, const FormatString& format = std::string_view{}, Ts&&... args) {
    const auto progress_width = static_cast<std::uint16_t>(perc * static_cast<float>(width));
    const auto remaining_width = static_cast<std::uint16_t>((1 - perc) * static_cast<float>(width));
    fmt::print("[{0:=^{1}}{0: ^{2}}] {3:5.1f} % \t{4}\r",
        "",
        progress_width,
        remaining_width,
        perc * 100.0f,
        fmt::format(fmt::runtime(format), std::forward<Ts>(args)...));
}

/**
 * @brief Print a progress bar with the status and a specific width in console.
 *
 * Has an additional format string, that gets appended to the end of the progress bar.
 *
 * @tparam FormatString type of the additional format string
 * @tparam Ts type for the additional format string
 * @param status progress status object
 * @param width width in characters for the bar
 * @param format extra format string
 * @param args format args
 */
template <typename FormatString = std::string_view, typename... Ts>
void print_progress_bar(const progress_status& status, std::uint16_t width, const FormatString& format = std::string_view{}, Ts&&... args) {
    print_progress_bar(status.get_percentage(), width, format, std::forward<Ts>(args)...);
}

/**
 * @brief Print a progress bar with the estimated remaining time using the status and a specific width in console.
 *
 * Has an additional format string, that gets appended to the end of the progress bar.
 *
 * @tparam FormatString type of the additional format string
 * @tparam Ts type for the additional format string
 * @param status timed progress status object
 * @param width width in characters for the bar
 * @param format extra format string
 * @param args format args
 */
template <typename FormatString = std::string_view, typename... Ts>
void print_timed_progress_bar(const timed_progress_status& status, std::uint16_t width, FormatString&& format = std::string_view{}, Ts&&... args) {
    const auto perc = status.get_percentage();
    const auto progress_width = static_cast<std::uint64_t>(perc * static_cast<double>(width));
    const auto remaining_width = width - progress_width;
    fmt::print("[{0:=^{1}}{0: ^{2}}] {3:5.1f} % ETA: {4:%Hh:%Mm:%Ss} \t{5}\r",
        "",
        progress_width,
        remaining_width,
        perc * 100.0,
        status.get_eta(),
        fmt::format(fmt::runtime(format), std::forward<Ts>(args)...));
}

namespace detail {
/**
 * @brief Empty string getter for the default behavior of no extra string in the printed progress bar
 *
 * @return std::string empty string
 */
[[nodiscard]] inline std::string get_empty_string(const progress_status& /* unused */) { return ""; }

/**
 * @brief Call the print function asynchronously using the status
 *
 * Waits until status.started is true, then prints the progress bar until status.done is true
 *
 * @tparam PrintFunction type of the function printing the progress bar
 * @param print function pringing the progress bar
 * @param status progress status
 * @return std::future<void> future from the async call
 */
template <typename PrintFunction>
[[nodiscard]] std::future<void> print_progress_bar_async(
    PrintFunction print,
    const progress_status& status) {

    static constexpr auto milliseconds_sleep = 10;
    auto sleep = []() {
        std::this_thread::sleep_for(std::chrono::milliseconds{ milliseconds_sleep });
    };

    return std::async([print, sleep, &status]() {
        while (!status.started) {
            sleep();
        }

        print();

        auto prev_perc = 0.0f;
        auto prev_time = std::chrono::steady_clock::now();

        static constexpr auto seconds_to_pass = 30;
        const auto seconds_passed = [](const auto& duration) {
            return std::chrono::duration_cast<std::chrono::seconds>(duration).count() > seconds_to_pass;
        };

        static constexpr auto perc_threshold = 0.001f;
        const auto perc_diff_threshold_passed = [](const auto perc_diff) {
            return perc_diff > perc_threshold;
        };

        while (!status.done) {
            const auto next_perc = status.get_percentage();
            const auto next_time = std::chrono::steady_clock::now();
            if (perc_diff_threshold_passed(next_perc - prev_perc) || seconds_passed(next_time - prev_time)) {
                print();
                std::cout << std::flush;
                prev_time = next_time;
                prev_perc = next_perc;
            }
            sleep();
        }
        print();
        fmt::print("\n");
        std::cout << std::flush;
    });
}
}

/**
 * @brief Call print_progress_bar_async with the corresponding print function for the ProgressStatusType
 *
 * Waits until status.started is true, then prints the progress bar until status.done is true
 *
 * @tparam ProgressStatusType progress status type
 * @tparam ExtraStringGetterFunctionType type of the extra string getter. Signature: std::string(const ProgressStatusType&)
 * @param status progress status
 * @param width progress bar print width
 * @param extra_string_getter extra string getter
 * @return std::future<void> future for the async call
 */
template <typename ProgressStatusType, typename ExtraStringGetterFunctionType = decltype(detail::get_empty_string)>
[[nodiscard]] std::future<void> print_progress_bar_async(const ProgressStatusType& status, std::uint16_t width,
    ExtraStringGetterFunctionType extra_string_getter = detail::get_empty_string) {
    auto print = [width, &status, extra_string_getter = std::move(extra_string_getter)]() {
        if constexpr (std::same_as<ProgressStatusType, progress_status>) {
            print_progress_bar(status, width, extra_string_getter(status));
        } else if constexpr (std::same_as<ProgressStatusType, timed_progress_status>) {
            print_timed_progress_bar(status, width, extra_string_getter(status));
        }
    };
    return detail::print_progress_bar_async(print, status);
}

/**
 * @brief Prints the progress bar continuously and asynchronously until the status indicates the operation has finished.
 *
 * Waits until status.started is true, then prints the progress bar until status.done is true
 *
 * @tparam ExtraStringGetterFunctionType type of the extra string getter. Signature: std::string(const progress_status&)
 * @param status progress status object
 * @param width width in characters for the bar
 * @param extra_string_getter extra string getter
 * @return std::future<void> future for the async call
 */
template <typename ExtraStringGetterFunctionType = decltype(detail::get_empty_string)>
[[nodiscard]] std::future<void> print_progress_bar_until_finished_async(
    const progress_status& status,
    std::uint16_t width,
    ExtraStringGetterFunctionType extra_string_getter = detail::get_empty_string) {
    return print_progress_bar_async(status, width, extra_string_getter);
}

/**
 * @brief Prints the timed progress bar continuously and asynchronously until the status indicates the operation has finished.
 *
 * Waits until status.started is true, then prints the progress bar until status.done is true
 *
 * @tparam ExtraStringGetterFunctionType type of the extra string getter. Signature: std::string(const timed_progress_status&)
 * @param status timed progress status object
 * @param width width in characters for the bar
 * @param extra_string_getter extra string callable
 * @return std::future<void> future for the async call
 */
template <typename ExtraStringGetterFunctionType = decltype(detail::get_empty_string)>
[[nodiscard]] std::future<void> print_timed_progress_bar_until_finished_async(
    const timed_progress_status& status,
    std::uint16_t width,
    ExtraStringGetterFunctionType extra_string_getter = detail::get_empty_string) {
    return print_progress_bar_async(status, width, extra_string_getter);
}

/**
 * @brief Class to handle a progress bar and printing it asynchronously with RAII
 *
 * Can be used to print a progress bar within a scoped region.
 * When using multiple progress bars within a scope or otherwise, manual management of when
 * to start and when the progress bar can stop is needed via the variable member status.
 * Otherwise multiple progress bars would print over each other.
 *
 * @tparam ProgressStatusType type of the progress state
 */
template <typename ProgressStatusType>
requires std::derived_from<ProgressStatusType, progress_status>
class RAII_progress_status {
public:
    /**
     * @brief Construct a new raii progress status object
     *
     * @tparam ExtraStringGetterFunctionType type of the extra string getter. Signature: std::string(const ProgressStatusType&)
     * @param total_steps total steps
     * @param width progress bar print width
     * @param start start printing the progress bar
     * @param extra_string_getter extra string getter
     */
    template <typename ExtraStringGetterFunctionType = decltype(detail::get_empty_string)>
    RAII_progress_status(size_t total_steps, std::uint16_t width, bool start = true, ExtraStringGetterFunctionType extra_string_getter = detail::get_empty_string)
        : status{ total_steps, start }
        , async_print_future{ print_progress_bar_async(status, width, extra_string_getter) } { }

    RAII_progress_status(const RAII_progress_status&) = delete;

    RAII_progress_status& operator=(const RAII_progress_status&) = delete;

    RAII_progress_status(RAII_progress_status&&) = delete;

    RAII_progress_status& operator=(RAII_progress_status&&) = delete;

    ~RAII_progress_status() {
        status.done.store(true);
    }

    ProgressStatusType status;

private:
    std::future<void> async_print_future;
};
