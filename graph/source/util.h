#pragma once

#include <future>
#include <spdlog/fmt/bundled/core.h>

#include "progress_status.h"

template <typename FormatString = std::string_view, typename... Ts>
void print_progress_bar(float perc, std::uint16_t width, const FormatString& format = std::string_view{}, Ts&&... args) {
    const auto progress = static_cast<std::uint16_t>(perc * static_cast<float>(width));
    const auto left = static_cast<std::uint16_t>((1 - perc) * static_cast<float>(width));
    fmt::print("[{0:=^{1}}{0: ^{2}}] {3:5.1f} % \t{4}\r",
        "",
        progress,
        left,
        perc * 100,
        fmt::format(fmt::runtime(format), std::forward<Ts>(args)...));
}

template <typename FormatString = std::string_view, typename... Ts>
void print_progress_bar(const progress_status& status, std::uint16_t width, const FormatString& format = std::string_view{}, Ts&&... args) {
    const auto perc = status.get_percentage();
    const auto progress = static_cast<std::uint16_t>(perc * static_cast<float>(width));
    auto left = static_cast<std::uint16_t>((1 - perc) * static_cast<float>(width));
    if (progress + left < width) {
        ++left;
    }

    fmt::print("[{0:=^{1}}{0: ^{2}}] {3:5.1f} % \t{4}\r",
        "",
        progress,
        left,
        perc * 100,
        fmt::format(fmt::runtime(format), std::forward<Ts>(args)...));
}

template <typename Callable>
[[nodiscard]] std::future<void> print_progress_bar_untill_finished_async(
    const progress_status& status,
    std::uint16_t width,
    Callable&& extra_string_getter = [](progress_status&) -> std::string { return ""; }) {
    return std::async([width, &status, extra_string_getter]() {
        while (!status.started) {
            std::this_thread::sleep_for(std::chrono::milliseconds{ 10 });
        }
        while (!status.done) {
            print_progress_bar(status, width, extra_string_getter(status));
            std::this_thread::sleep_for(std::chrono::milliseconds{ 10 });
        }
        print_progress_bar(status, width, extra_string_getter(status));
        fmt::print("\n");
    });
}
