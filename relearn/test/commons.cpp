#include "commons.h"

#include "../source/LogFiles.h"
#include "../source/MPIWrapper.h"
#include "../source/RelearnException.h"

#include <mutex>

std::mt19937 mt;
std::once_flag some_flag;

void setup() {

    auto lambda = []() {
        char* argument = (char*)"./runTests";
        MPIWrapper::init(1, &argument);
        LogFiles::init();
    };

    std::call_once(some_flag, lambda);

    RelearnException::hide_messages = true;
}
