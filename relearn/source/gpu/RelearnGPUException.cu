#include "RelearnGPUException.h"

const char* RelearnGPUException::what() const noexcept {
    return message.c_str();
}

RelearnGPUException::RelearnGPUException(std::string mes)
    : message(std::move(mes)) {
}

void RelearnGPUException::log_message(const std::string& message) {

    std::cerr << message << std::flush;
    fflush(stderr);
}